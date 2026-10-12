"""Train Nequix with Muon or BMuon.

Usage:  python -m bmuon.train bmuon/configs/<config>.yml

The loop follows nequix/train.py (data, loss, EMA, evaluation are imported from Nequix)
with these additions:
  * optimizer "bmuon" (budgeted Muon on the type-I channels, Muon/AdamW elsewhere);
  * a budget schedule annealed from the full cost to ``bmuon.budget``;
  * gradient accumulation (``grad_accum_steps``) to keep Nequix's global batch on one GPU;
  * ``base_config`` inheritance, optional training subset, step-based lr schedule;
  * automatic resume from ``out_dir/state.pkl`` and local JSONL logs (wandb optional);
  * the allocation (one value per channel) is saved every epoch.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import yaml

from bmuon.channels import build_channel_spec
from bmuon.optimizer import bmuon, find_bmuon_state, weight_decay_keys
from nequix.data import (
    AseDBDataset,
    ConcatDataset,
    DataLoader,
    IndexDataset,
    ParallelLoader,
    average_atom_energies,
    dataset_stats,
    prefetch,
)
from nequix.model import Nequix, save_model, weight_decay_mask
from nequix.train import evaluate, load_training_state, loss, save_training_state


# ------------------------------------------------------------------------ config


def _expand(x):
    if isinstance(x, str):
        return os.path.expandvars(x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def apply_overrides(config: dict, overrides: list) -> dict:
    """``key=value`` or ``a.b=value`` overrides; values are parsed as YAML."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def load_config(path: str) -> dict:
    """YAML config; ``base_config`` (relative to this file or absolute) is merged first.
    Environment variables in string values ($USER, $TMPDIR, ...) are expanded."""
    with open(path) as f:
        config = yaml.safe_load(f)
    base = config.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = (Path(path).parent / base_path).resolve()
            if not base_path.exists():
                base_path = Path(base)
        merged = load_config(str(base_path))
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        config = merged
    return _expand(config)


def budget_schedule(cfg: dict, total_steps: int):
    """Fraction of the total cost available, linear from 1 to ``budget`` over the first
    ``anneal_fraction`` of training, then constant (as in the BMuon paper)."""
    target = float(cfg.get("budget", 1.0))
    n = max(1, int(cfg.get("anneal_fraction", 0.25) * total_steps))

    def frac(step):
        t = jnp.clip(step / n, 0.0, 1.0)
        return 1.0 + (target - 1.0) * t

    return frac


def build_optimizer(config: dict, model, schedule, total_steps: int):
    wd = config["weight_decay"] if config["weight_decay"] != 0.0 else None
    clip = optax.clip_by_global_norm(config["grad_clip_norm"])
    name = config["optimizer"]
    spec = None
    if name == "muon":
        inner = optax.contrib.muon(
            learning_rate=schedule, weight_decay=wd, weight_decay_mask=weight_decay_mask(model)
        )
    elif name == "adamw":
        inner = optax.adamw(
            learning_rate=schedule,
            weight_decay=config["weight_decay"],
            mask=weight_decay_mask(model),
        )
    elif name == "bmuon":
        cfg = config.get("bmuon", {})
        spec = build_channel_spec(model, cost_model=cfg.get("cost_model", "cg"))
        inner = bmuon(
            learning_rate=schedule,
            spec=spec,
            budget_fraction=budget_schedule(cfg, total_steps),
            alpha=cfg.get("alpha", 0.5),
            refresh_every=cfg.get("refresh_every", 5),
            eps=cfg.get("eps", 1e-6),
            weight_decay=wd,
            weight_decay_keys=weight_decay_keys(model),
        )
    else:
        raise ValueError(f"optimizer {name} not supported")
    optim = optax.chain(clip, inner)
    accum = int(config.get("grad_accum_steps", 1))
    if accum > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=accum)
    return optim, spec


# ------------------------------------------------------------------------- data


def load_datasets(config):
    def ase(path):
        return AseDBDataset(
            file_path=path, atomic_numbers=config["atomic_numbers"], cutoff=config["cutoff"],
            backend="jax",
        )

    paths = config["train_path"]
    train_dataset = ConcatDataset([ase(p) for p in paths]) if isinstance(paths, list) else ase(paths)
    atomic_indices = getattr(train_dataset, "atomic_indices", None)
    if "valid_frac" in config:
        train_dataset, val_dataset = train_dataset.split(valid_frac=config["valid_frac"])
    else:
        val_dataset = ase(config["valid_path"])
    frac = config.get("train_subset_frac")
    if frac and frac < 1.0:
        rng = np.random.default_rng(config.get("subset_seed", 0))
        idx = np.sort(rng.choice(len(train_dataset), int(frac * len(train_dataset)), replace=False))
        train_dataset = IndexDataset(train_dataset, idx)
    vfrac = config.get("valid_subset_frac")
    if vfrac and vfrac < 1.0:
        rng = np.random.default_rng(1)
        idx = np.sort(rng.choice(len(val_dataset), int(vfrac * len(val_dataset)), replace=False))
        val_dataset = IndexDataset(val_dataset, idx)
    if atomic_indices is not None:  # needed by nequix.data.average_atom_energies after a split
        train_dataset.atomic_indices = atomic_indices
    return train_dataset, val_dataset


# --------------------------------------------------------------------- logging


class Logger:
    def __init__(self, out_dir: Path, config: dict, resume_id=None):
        self.path = out_dir / "log.jsonl"
        self.wandb = None
        if config.get("wandb", False):
            import wandb

            kw = {"project": config.get("wandb_project", "bmuon-nequix"), "config": config,
                  "name": config.get("run_name"), "dir": str(out_dir)}
            if resume_id:
                kw.update(id=resume_id, resume="allow")
            wandb.init(**kw)
            self.wandb = wandb

    @property
    def run_id(self):
        return getattr(getattr(self.wandb, "run", None), "id", None)

    def log(self, logs: dict, step: int):
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": int(step), **logs}) + "\n")
        if self.wandb:
            self.wandb.log(logs, step=int(step))
        print(f"step {int(step)}: {logs}", flush=True)


def allocation_logs(spec, alloc) -> dict:
    if spec is None or alloc is None:
        return {}
    alloc = np.asarray(alloc)
    logs = {
        "alloc/spent": float(spec.weights @ alloc / spec.weights.sum()),
        "alloc/frac_above_half": float((alloc > 0.5).mean()),
    }
    groups = defaultdict(list)
    for i, c in enumerate(spec.channels):
        groups[(c.layer, c.ir)].append(alloc[i])
    for (t, ir), v in groups.items():
        logs[f"alloc/b{t}_{ir}_mean"] = float(np.mean(v))
        logs[f"alloc/b{t}_{ir}_active"] = int(np.sum(np.asarray(v) > 0.5))
    return logs


def save_allocation(path: Path, spec, state):
    if spec is None or state is None:
        return
    np.savez(
        path,
        alloc=np.asarray(state.alloc),
        scores=np.asarray(state.scores),
        weights=spec.weights,
        layer=np.array([c.layer for c in spec.channels]),
        ir=np.array([c.ir for c in spec.channels]),
        copy=np.array([c.copy for c in spec.channels]),
    )


# ----------------------------------------------------------------------- train


def train(config_path: str, overrides: list | None = None):
    config = apply_overrides(load_config(config_path), overrides)
    out_dir = Path(config.get("out_dir", f"bmuon/runs/{Path(config_path).stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yml", "w") as f:
        yaml.safe_dump(config, f)
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    train_dataset, val_dataset = load_datasets(config)
    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        atom_energies = average_atom_energies(train_dataset)
    keys = ["shift", "scale", "avg_n_neighbors", "max_n_edges", "max_n_nodes", "avg_n_nodes",
            "avg_n_edges"]
    stats = ({k: config[k] for k in keys} if all(k in config for k in keys)
             else dataset_stats(train_dataset, atom_energies))

    num_devices = len(jax.devices())
    workers = config.get("num_workers", 16)
    loader_kw = dict(
        batch_size=config["batch_size"], max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"], avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"], num_workers=workers,
    )
    train_loader = ParallelLoader(DataLoader(train_dataset, shuffle=True, **loader_kw), num_devices)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

    model = Nequix(
        jax.random.key(config.get("seed", 0)),
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
    )
    config_to_save = {**config, **stats}

    accum = int(config.get("grad_accum_steps", 1))
    micro_per_epoch = len(train_dataset) // (config["batch_size"] * num_devices)
    steps_per_epoch = max(1, micro_per_epoch // accum)  # optimizer updates per epoch
    total_steps = config["n_epochs"] * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["learning_rate"] * config["warmup_factor"],
        peak_value=config["learning_rate"],
        end_value=1e-6,
        warmup_steps=int(config["warmup_epochs"] * steps_per_epoch),
        decay_steps=total_steps,
    )
    optim, spec = build_optimizer(config, model, schedule, total_steps)
    if spec is not None:
        print(f"BMuon channels ({spec.num_channels}):\n{spec.summary()}", flush=True)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model = jax.device_put_replicated(model, list(jax.devices()))
    opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))
    ema_model = jax.tree.map(lambda x: x.copy(), model)
    step, start_epoch, best_val_loss, run_id = jnp.array(0), 0, float("inf"), None

    state_path = Path(config.get("state_path") or out_dir / "state.pkl")
    if state_path.exists():
        (model, ema_model, _, opt_state, step, start_epoch, best_val_loss,
         run_id) = load_training_state(state_path)
        print(f"resumed from {state_path} at epoch {start_epoch}", flush=True)
    logger = Logger(out_dir, config, run_id)

    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch):
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, batch, config["energy_weight"], config["force_weight"],
            config["stress_weight"], config["loss_type"],
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        # EMA once per optimizer update (with accumulation, updates are zero in between)
        ema_step = step // accum
        decay = jnp.minimum(config["ema_decay"], (1 + ema_step) / (10 + ema_step))
        decay = jnp.where((step + 1) % accum == 0, decay, 1.0)
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        new_ema = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, eqx.filter(model, eqx.is_array)
        )
        return model, eqx.combine(ema_static, new_ema), opt_state, total_loss, metrics

    def current_bmuon_state():
        s = find_bmuon_state(opt_state)
        return None if s is None else jax.tree.map(lambda x: x[0], s)

    for epoch in range(start_epoch, config["n_epochs"]):
        t0 = time.time()
        train_loader.loader.set_epoch(epoch)
        for batch in prefetch(train_loader):
            model, ema_model, opt_state, total_loss, metrics = train_step(
                model, ema_model, step, opt_state, batch
            )
            step = step + 1
            if step % config["log_every"] == 0:
                logs = {"train/loss": total_loss.mean().item(),
                        "learning_rate": float(schedule(step // accum)),
                        "time/elapsed": time.time() - t0}
                logs.update({f"train/{k}": v.mean().item() for k, v in metrics.items()})
                bs = current_bmuon_state()
                if bs is not None:
                    logs.update(allocation_logs(spec, bs.alloc))
                logger.log(logs, step)

        ema_single = jax.tree.map(lambda x: x[0], ema_model)
        val = evaluate(ema_single, val_loader, config["energy_weight"], config["force_weight"],
                       config["stress_weight"], config["loss_type"])
        if val["loss"] < best_val_loss:
            best_val_loss = float(val["loss"])
            save_model(out_dir / "checkpoint.nqx", ema_single, config_to_save)
        save_model(out_dir / "last.nqx", ema_single, config_to_save)
        bs = current_bmuon_state()
        save_allocation(out_dir / f"alloc_epoch{epoch:03d}.npz", spec, bs)
        save_training_state(state_path, model, ema_model, optim, opt_state, step, epoch + 1,
                            best_val_loss, wandb_run_id=logger.run_id)
        logs = {f"val/{k}": float(v) for k, v in val.items()}
        logs.update({"epoch": epoch, "time/epoch": time.time() - t0})
        if bs is not None:
            logs.update(allocation_logs(spec, bs.alloc))
        logger.log(logs, step)

    (out_dir / "DONE").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. bmuon.budget=0.3")
    args = parser.parse_args()
    train(args.config_path, args.set)


if __name__ == "__main__":
    main()
