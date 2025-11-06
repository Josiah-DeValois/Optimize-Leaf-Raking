# Optimize_Leaf_Raking
Optimization and visualization of leaf‑raking strategies (outside‑in piles vs. front‑sweep), with a solver baseline.

## What’s here
- Modular package in `src/optimize_leaf_raking/`:
  - `core/`: configuration, calibration, yard grid, raking physics, bagging/walking, center heuristics, front‑sweep dynamics
  - `solvers/`: MILP p‑median solver (PuLP)
  - `viz/`: plotting helpers and a 2×2 animated comparison with controls
- Scripts in `scripts/`:
  - `solve_centers_mip.py`: solve pile‑center selection via MILP (PuLP)
  - `run_viz.py`: run the interactive 2×2 animation (and optionally save it)

## Quickstart
Create optimal centers via MILP and save to CSV:

```
python3 scripts/solve_centers_mip.py --K-max 5 --candidate-spacing 10 --grid-step 1
```

Run the 2×2 animation with front‑sweep controls:

```
python3 scripts/run_viz.py --show
# Optional: save an MP4 (requires ffmpeg)
python3 scripts/run_viz.py --save --format mp4
# Or save a GIF (pillow writer)
python3 scripts/run_viz.py --save --format gif
```

Useful flags:
- `--fps 2` and `--spf 60` (seconds per frame) to control playback.

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
- By default, the “Optimization (discrete K≤5)” panel uses a discrete search over candidate sites for parity with the original. The MILP script (`solve_centers_mip.py`) provides a stronger baseline you can wire in if desired.
- Outputs are saved under `results/`.

