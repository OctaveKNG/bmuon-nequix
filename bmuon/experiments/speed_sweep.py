"""How does inference time depend on the multiplicities entering the tensor product?

    python -m bmuon.experiments.speed_sweep --kernel --out bmuon/results/sweep_3090

The first benchmark showed a step-like behaviour (25 % fewer copies: ~no gain; 50 %: -35 %),
so the cost of a copy is not linear in the number of copies. This script maps it:
  * uniform: every multiplicity scaled by f, f in 1, 15/16, ..., 1/16;
  * per irrep: the multiplicity of one irrep set to m in all interaction blocks at once
    (others full), m from 4 to its full value in steps of 4.
Writes <out>.csv (one row per variant, times in ms per structure size).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import e3nn_jax as e3nn

from bmuon.experiments.common import platform
from bmuon.experiments.speed import structures, time_model
from bmuon.model import with_tp_irreps
from bmuon.prune import tp_irreps_summary
from nequix.model import load_model


def set_mul(irreps: e3nn.Irreps, ir_name: str, mul: int) -> e3nn.Irreps:
    return e3nn.Irreps([(mul if str(ir) == ir_name else m, ir) for m, ir in irreps])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nequix-mp-1.nqx")
    p.add_argument("--sizes", default="2,3,4")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--kernel", action="store_true")
    p.add_argument("--only", choices=["uniform", "irrep", "all"], default="all")
    p.add_argument("--out", default="bmuon/results/sweep")
    args = p.parse_args()

    model, config = load_model(args.model, kernel=False)
    base = [e3nn.Irreps(s) for s in tp_irreps_summary(model)]
    graphs = structures(config, [int(s) for s in args.sizes.split(",")])
    atoms = [n for n, _ in graphs]
    print(f"{platform()}  structures: {atoms} atoms", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "irrep", "value", "tp_irreps"] + [f"t_{n}_atoms_ms" for n in atoms]
    fh = open(f"{out}.csv", "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    def run(kind, irrep, value, irreps):
        m = with_tp_irreps(model, irreps, kernel=args.kernel)
        ts = time_model(m, graphs, args.reps)
        row = {"kind": kind, "irrep": irrep, "value": value,
               "tp_irreps": " | ".join(map(str, irreps))}
        row.update({f"t_{n}_atoms_ms": round(1e3 * t, 3) for n, t in zip(atoms, ts)})
        writer.writerow(row)
        fh.flush()
        print(kind, irrep, value, [f"{1e3 * t:.2f}" for t in ts], flush=True)

    run("base", "", 1.0, base)
    if args.only in ("uniform", "all"):
        for k in range(15, 0, -1):
            f = k / 16
            run("uniform", "", f, [e3nn.Irreps([(max(1, round(m * f)), ir) for m, ir in x])
                                   for x in base])
    if args.only in ("irrep", "all"):
        full = {str(ir): m for m, ir in base[1]}  # hidden multiplicities (blocks >= 1)
        for ir_name, m_full in full.items():
            for mul in range(args.step, m_full, args.step):
                irreps = [x if t == 0 else set_mul(x, ir_name, mul) for t, x in enumerate(base)]
                run("irrep", ir_name, mul, irreps)
    fh.close()
    print(f"wrote {out}.csv")


if __name__ == "__main__":
    main()
