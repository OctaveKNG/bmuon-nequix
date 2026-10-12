"""Rank type-I channels, choose which to keep, and export a smaller dense Nequix."""

from __future__ import annotations

from collections import defaultdict

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bmuon.channels import ChannelSpec, build_channel_spec
from bmuon.model import with_tp_irreps


def _leaves(tree) -> dict:
    return {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _path_weights(model) -> dict:
    """keystr -> runtime multiplier of every e3nn Linear weight in the interaction blocks."""
    out = {}
    for t, conv in enumerate(model.layers):
        for attr in ("linear_1", "linear_2", "skip"):
            lin = getattr(conv, attr)._linear
            for ins in lin.instructions:
                if ins.i_in == -1:
                    continue
                name = f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}"
                out[f".layers[{t}].{attr}._weights['{name}']"] = float(ins.path_weight)
    return out


def magnitude_scores(model, spec: ChannelSpec) -> np.ndarray:
    """Coupled magnitude s_c = sqrt(sum of squared norms of all slices of channel c)."""
    leaves = _leaves(eqx.filter(model, eqx.is_array))
    sq = np.zeros(spec.num_channels)
    for s in spec.slices:
        w = np.asarray(leaves[s.keypath])
        per = (w**2).sum(axis=1 - s.axis)
        np.add.at(sq, s.channel_ids, per)
    return np.sqrt(sq)


def select_channels(
    scores: np.ndarray,
    spec: ChannelSpec,
    remove_fraction: float,
    by: str = "count",
    min_keep: int | dict = 1,
    round_to: int = 1,
) -> np.ndarray:
    """Global ranking: drop the lowest-scored channels until ``remove_fraction`` is reached.

    ``by``: "count" (fraction of channels) or "cost" (fraction of the total cost w).
    ``min_keep``: minimum copies kept per (block, irrep); an int or a dict irrep -> int.
    ``round_to``: kept copies per (block, irrep) rounded up to a multiple (kernel friendly).
    Returns a boolean keep mask over channels.
    """
    n = spec.num_channels
    w = spec.weights if by == "cost" else np.ones(n)
    target = remove_fraction * w.sum()
    groups = defaultdict(list)
    for i, c in enumerate(spec.channels):
        groups[(c.layer, c.ir)].append(i)

    def floor(ir):
        return min_keep.get(ir, 1) if isinstance(min_keep, dict) else min_keep

    keep = np.ones(n, bool)
    left = {g: len(ids) for g, ids in groups.items()}
    removed = 0.0
    for i in np.argsort(scores, kind="stable"):
        if removed >= target - 1e-12:
            break
        c = spec.channels[i]
        g = (c.layer, c.ir)
        if left[g] <= min(floor(c.ir), len(groups[g])):
            continue
        keep[i] = False
        left[g] -= 1
        removed += w[i]

    if round_to > 1:
        for g, ids in groups.items():
            ids = np.array(ids)
            m = int(keep[ids].sum())
            want = min(len(ids), int(np.ceil(m / round_to) * round_to)) if m else 0
            best = ids[np.argsort(-scores[ids], kind="stable")][:want]
            keep[ids] = False
            keep[best] = True
    return keep


def prune(model, spec: ChannelSpec, keep: np.ndarray, kernel: bool | None = None):
    """Export a dense model without the dropped channels. Exactly equal to ``model`` with
    the slices of the dropped channels set to zero."""
    kept = defaultdict(list)  # (layer, ir) -> old copy indices, ascending
    for i, c in enumerate(spec.channels):
        if keep[i]:
            kept[(c.layer, c.ir)].append(c.copy)

    new_irreps = []
    for t, conv in enumerate(model.layers):
        old = e3nn.Irreps(getattr(conv, "tp_input_irreps", conv.linear_1.irreps_out))
        new_irreps.append(
            e3nn.Irreps([(len(kept[(t, str(ir))]), ir) for _, ir in old if kept[(t, str(ir))]])
        )
    new_model = with_tp_irreps(model, new_irreps, kernel)
    new_spec = build_channel_spec(new_model, cost_model="uniform")

    old_leaves = _leaves(eqx.filter(model, eqx.is_array))
    old_slice = {(s.layer, s.role, s.ir): s for s in spec.slices}
    pw_old, pw_new = _path_weights(model), _path_weights(new_model)
    gathered = {}
    for s in new_spec.slices:
        o = old_slice[(s.layer, s.role, s.ir)]
        pos = {lab: k for k, lab in enumerate(o.labels)}
        idx = [pos[(pk, ir1, kept[(s.layer, ir1)][j])] for pk, ir1, j in s.labels]
        w = jnp.take(old_leaves[o.keypath], jnp.array(idx), axis=s.axis)
        # e3nn Linear multiplies each weight by a constant that depends on the fan-in
        # (number of input copies); keep the product weight * constant unchanged.
        if o.keypath in pw_old:
            w = w * (pw_old[o.keypath] / pw_new[s.keypath])
        gathered[s.keypath] = w

    params, static = eqx.partition(new_model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    out = []
    for p, x in flat:
        k = jax.tree_util.keystr(p)
        y = gathered.get(k, old_leaves.get(k))
        if y is None or y.shape != x.shape:
            raise ValueError(f"cannot fill {k}: {None if y is None else y.shape} vs {x.shape}")
        out.append(y)
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), static)


def zero_channels(model, spec: ChannelSpec, keep: np.ndarray):
    """Same model with every slice of a dropped channel set to zero (no shape change)."""
    params, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    by_key = {s.keypath: s for s in spec.slices}
    out = []
    for p, x in flat:
        s = by_key.get(jax.tree_util.keystr(p))
        if s is not None:
            m = jnp.asarray(keep[s.channel_ids], x.dtype)
            x = x * (m[:, None] if s.axis == 0 else m[None, :])
        out.append(x)
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), static)


def tp_irreps_summary(model) -> list:
    return [str(getattr(c, "tp_input_irreps", c.linear_1.irreps_out)) for c in model.layers]
