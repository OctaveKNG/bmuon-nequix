"""Section 6.2: inference time of Nequix vs the multiplicities entering the tensor product.

    python -m bmuon.experiments.speed --model models/nequix-mp-1.nqx --kernel \
        --out bmuon/results/speed_3090 --cost-json bmuon/configs/costs_3090.json

Times jitted energy + forces on bulk supercells of increasing size, for
  * the base model,
  * one variant per (block, irrep) with half of its copies removed  -> marginal cost per copy,
  * uniform shrinks of every multiplicity (25/50/75 %)               -> linearity check.
Weights do not matter for timing; rebuilt blocks keep random weights. Writes a CSV of raw
timings and a JSON {"channel_costs": {"layer:irrep": cost}} usable as bmuon.cost_model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import e3nn_jax as e3nn
import equinox as eqx
import jax
import numpy as np
from ase.build import bulk

from bmuon.model import with_tp_irreps
from bmuon.prune import tp_irreps_summary
from bmuon.experiments.common import bucket, pad, platform, to_graph
from nequix.model import load_model


def structures(config, sizes):
    out = []
    for n in sizes:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
        atoms.rattle(0.02, seed=n)
        g = to_graph(atoms, config["atomic_numbers"], config["cutoff"])
        out.append((len(atoms), pad(g, bucket(len(atoms) + 1, 32), bucket(int(g.n_edge[0]) + 1, 256))))
    return out


def time_model(model, graphs, reps: int, warmup: int = 3, stress: bool = False):
    @eqx.filter_jit
    def f(m, g):
        e, forces, s = m(g)
        return (e, forces, s) if stress else (e, forces)

    times = []
    for _, g in graphs:
        for _ in range(warmup):
            jax.block_until_ready(f(model, g))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(model, g))
            ts.append(time.perf_counter() - t0)
        times.append(float(np.median(ts)))
    return times


def scaled(irreps: e3nn.Irreps, factor: float) -> e3nn.Irreps:
    return e3nn.Irreps([(max(1, int(round(m * factor))), ir) for m, ir in irreps])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nequix-mp-1.nqx")
    p.add_argument("--sizes", default="2,3,4", help="Si cubic supercell repeats (8 n^3 atoms)")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--kernel", action="store_true", help="use OpenEquivariance kernels (GPU)")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--out", default="bmuon/results/speed")
    p.add_argument("--cost-json", default=None)
    args = p.parse_args()

    model, config = load_model(args.model, kernel=False)
    base_irreps = [e3nn.Irreps(s) for s in tp_irreps_summary(model)]
    model = with_tp_irreps(model, base_irreps, kernel=args.kernel)
    graphs = structures(config, [int(s) for s in args.sizes.split(",")])
    atoms = [n for n, _ in graphs]
    print(f"{platform()}  structures: {atoms} atoms", flush=True)

    rows = []

    def run(name, irreps, removed=None):
        m = with_tp_irreps(model, irreps, kernel=args.kernel)
        ts = time_model(m, graphs, args.reps, stress=args.stress)
        row = {"variant": name, "tp_irreps": " | ".join(map(str, irreps))}
        row.update({f"t_{n}_atoms_ms": 1e3 * t for n, t in zip(atoms, ts)})
        if removed:
            row["removed_layer"], row["removed_ir"], row["removed_copies"] = removed
        rows.append(row)
        print(name, [f"{1e3 * t:.2f} ms" for t in ts], flush=True)
        return np.array(ts)

    t_base = run("base", base_irreps)
    marginal = {}
    for t, irreps in enumerate(base_irreps):
        for i, (mul, ir) in enumerate(irreps):
            if mul < 2:
                continue
            k = mul // 2
            new = [x for x in base_irreps]
            new[t] = e3nn.Irreps([(m - k if j == i else m, r) for j, (m, r) in enumerate(irreps)])
            ts = run(f"b{t}_{ir}_half", new, (t, str(ir), k))
            # per copy, per atom, averaged over the structure sizes
            marginal[f"{t}:{ir}"] = float(np.mean((t_base - ts) / k / np.array(atoms)))
    for f in (0.75, 0.5, 0.25):
        run(f"uniform_{f}", [scaled(x, f) for x in base_irreps])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    pos = {k: max(v, 0.0) for k, v in marginal.items()}
    floor = max(min([v for v in pos.values() if v > 0], default=1.0) * 0.1, 1e-12)
    costs = {k: max(v, floor) for k, v in pos.items()}
    mean = float(np.mean(list(costs.values())))
    result = {
        "platform": platform(),
        "kernel": args.kernel,
        "atoms": atoms,
        "base_ms": (1e3 * t_base).tolist(),
        "channel_costs": {k: v / mean for k, v in costs.items()},
        "raw_seconds_per_copy_per_atom": marginal,
    }
    with open(f"{out}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    if args.cost_json:
        with open(args.cost_json, "w") as fh:
            json.dump({"channel_costs": result["channel_costs"], "platform": platform()}, fh,
                      indent=2)
    print(json.dumps(result["channel_costs"], indent=2))


if __name__ == "__main__":
    main()
