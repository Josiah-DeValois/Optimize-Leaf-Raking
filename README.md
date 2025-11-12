# Optimize_Leaf_Raking
Optimization and visualization of leaf‑raking strategies with a solver baseline, for illustrating MILP and heuristics.

## What’s here
- Modular package in `src/optimize_leaf_raking/`:
  - `core/`: configuration, calibration, yard grid, raking physics, bagging/walking, center heuristics, front‑sweep dynamics
  - `solvers/`: MILP p‑median solver (PuLP)
  - `viz/`: plotting helpers and a 2×2 animated comparison with controls
- Scripts in `scripts/`:
  - `solve_centers_mip.py`: solve pile‑center selection via MILP (PuLP)
  - `run_viz.py`: run the interactive 2×2 animation (and optionally save it)

## Quickstart
First time: install the package (editable) so scripts can import it (this also installs numpy, matplotlib, and PuLP):

```
pip install -e .
```
### Solve centers (preferred MILP, PuLP)
Compute pile locations by solving the MILP with a practical stopping policy (5% relative gap and at least 5 minutes), and save the chosen centers to `results/optimal_centers.csv`:

```
python3 scripts/solve_centers_mip.py \
  --K-max 5 --candidate-spacing 10 --grid-step 1 \
  --rel-gap 0.05 --min-seconds 300
```

What you’ll see printed:
- MILP objective (raking + bagging seconds)
- Nearest-assignment eval (for transparency)
- Chosen pile centers (x, y) and a CSV at `results/optimal_centers.csv`

### Visualization
Run the 2×2 animation; if `results/optimal_centers.csv` exists, the Optimization panel uses those centers and its total time matches the MILP objective.

```
python3 scripts/run_viz.py --show
```

Optional saves (run one of these as a separate command) — writes to `results/figures/`:

```
python3 scripts/run_viz.py --save --format mp4
python3 scripts/run_viz.py --save --format gif
```

Useful flags:
- `--fps 2` and `--spf 60` (seconds per frame) to control playback
- `--style interactive` for the original UI with sliders

### Tests
Run the unit tests:

```
pytest -q
```

Notes:
- Solver‑dependent tests are skipped if PuLP is not installed. Installing this package via `pip install -e .` will install PuLP.

## Package outline
- `core/config.py` — dataclasses for parameters (yard, rake/bag models, UI, viz)
- `core/calibration.py` — fit power‑law raking, bagging time model
- `core/yard.py` — grid and mass distribution
- `core/raking.py` — outside‑in arrivals and pile deposition helpers
- `core/bagging.py` — bagging time simulation, walking between piles
- `core/centers.py` — BF centers, micro‑piles, discrete search
- `core/front_sweep.py` — strip timing, column‑aware spillage, band bagging
- `solvers/mip.py` — p‑median MILP using PuLP
- `viz/plotting.py` — colormap + density rendering
- `viz/animate.py` — build the 2×2 figure, UI controls, and animation

## Notes
- By default, the “Optimization (discrete K≤5)” panel can fall back to a discrete search when no MILP centers are available. I prefer running the MILP first so the Optimization panel reflects the solver’s centers and time.
- Outputs are saved under `results/`.
- If you cannot install packages, you can also run with `PYTHONPATH=src` to make the package importable: `PYTHONPATH=src python3 scripts/run_viz.py --show`.
- I changed the solver so that x is continuous instead of binary to speed the algorithm up (ie, relaxed the binary constraint, making it so that part of a cell could be raked to one pile, and part to another)
