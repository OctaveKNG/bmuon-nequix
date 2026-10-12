# BMuon for Nequix — what is new in this repository

Everything outside the folders below is the upstream Nequix code
(github.com/atomicarchitects/nequix, commit da0fb24, MIT license) and is **unmodified**.

| Path | Content |
| --- | --- |
| `bmuon/channels.py` | map of the type-I channels: for each copy entering the tensor product, its slices in `linear_1`, `linear_2` and the radial MLP; channel costs |
| `bmuon/optimizer.py` | BMuon as optax transformations (budgeted matrices) + optax Muon/AdamW for the rest |
| `bmuon/model.py` | `PrunableConvolution` (tensor product fed by fewer copies than the block input), save/load of pruned models |
| `bmuon/prune.py` | channel scores, global selection, export to a smaller dense model |
| `bmuon/train.py` | training loop (Muon or BMuon), gradient accumulation, auto-resume, allocation logging |
| `bmuon/configs/` | full model (`nequix-mp-1-*`), dev model (`small-*`), CPU smoke test (`toy-bmuon`) |
| `experiments/fragility.py` | §6.1/6.3 accuracy after pruning, no fine-tuning |
| `experiments/speed.py` | §6.2 timing and measured channel costs (`costs_<gpu>.json`) |
| `experiments/launch.py`, `run_forever.sh` | spread runs over the lab machines, resume on failure |
| `experiments/machines.txt`, `runs_*.yml` | machine list and run plans (pilot, wave 3) |
| `tests_bmuon/` | 14 tests (see below) |

## Install (pinned like Nequix)

    uv sync --extra oeq && uv pip install openequivariance_extjax --no-build-isolation
    uv pip install ase-db-backends scipy pytest

The code was tested with jax 0.5.3, e3nn-jax 0.20.8, equinox 0.13.1, optax 0.2.6, lmdb 1.x
(the versions of `uv.lock`). lmdb 3.x breaks Nequix's forked data workers; jax ≥ 0.10 removed
`jax.device_put_replicated`, used by the training loop.

## Tests

    python -m pytest tests_bmuon

* the slice map is exact: zeroing a channel's `linear_1` column zeroes the gradient of all its
  other slices and only them;
* with the full budget, BMuon reproduces Nequix's `optax.contrib.muon` updates step by step;
* the allocation solver satisfies the cost budget and beats random feasible allocations;
* low-allocation channels move by at most ~mu^alpha;
* the exported pruned model equals the model with zeroed slices (energy, forces), whole irreps
  can disappear, the pruned model is O(3)-equivariant (rotations and inversion), save/load works.

## Typical commands

    python -m experiments.speed --kernel --out results/speed_3090 --cost-json bmuon/configs/costs_3090.json
    python -m experiments.fragility --model models/nequix-mp-1.nqx --data $BMUON_DATA/mptrj-val --n-structures 2000
    python -m bmuon.train bmuon/configs/small-bmuon.yml --set bmuon.budget=0.3 out_dir=runs/test
    python -m experiments.launch plan  experiments/runs_wave3.yml experiments/machines.txt
    python -m experiments.launch start experiments/runs_wave3.yml experiments/machines.txt
    python -m experiments.launch status experiments/runs_wave3.yml

## Implementation notes

* e3nn-jax regroups the input of `linear_2`: there is **one matrix per output irrep of the
  tensor product** (rows = all (path, copy) pairs producing that irrep), not one per path.
  BMuon orthogonalizes that whole matrix with the channel scaling on its rows.
* e3nn `Linear` multiplies each weight by a constant depending on its fan-in; the export
  rescales `linear_2` rows so that the pruned model is exactly the zeroed one.
* Weights are stored `(in, out)` and applied as `x @ W`: a channel is a **column** of
  `linear_1` and of the radial layer, and a **row** of `linear_2`.
