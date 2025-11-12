#!/usr/bin/env python3
"""
Solve pile-center selection with a MILP (p-median style) using PuLP.

- Replicates the yard model and calibration from original.py
- Builds a candidate grid of potential pile sites
- Minimizes total raking time: sum_i alpha * mass_i * (dist(i, assigned_center))**beta
  using assignment variables (x_ij) and binary open variables (y_j).
  This linearizes the "min over centers" via assignment constraints.

Outputs:
- Prints chosen centers and objective value (seconds)
- Saves chosen centers to results/optimal_centers.csv

Note: Requires PuLP. If not installed, install via: pip install pulp
"""

from pathlib import Path
import sys
import argparse
import numpy as np

# Ensure `src/` is on PYTHONPATH when running from repo
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from optimize_leaf_raking.core.config import YardParams, CalibrationData
from optimize_leaf_raking.core.calibration import calibrate_rake_model, calibrate_bag_model
from optimize_leaf_raking.core.yard import build_yard_and_masses
from optimize_leaf_raking.solvers.mip import build_candidates, solve_pmedian_mip


def main():
    parser = argparse.ArgumentParser(description="Solve pile-center selection via MILP (PuLP)")
    parser.add_argument("--K-max", type=int, default=5, help="Maximum number of centers to open (<=K)")
    parser.add_argument("--candidate-spacing", type=float, default=10.0, help="Candidate grid spacing (ft)")
    parser.add_argument("--grid-step", type=float, default=1.0, help="Cell grid spacing (ft)")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional hard solver time limit (seconds)")
    parser.add_argument("--rel-gap", type=float, default=0.05, help="Relative optimality gap target (e.g., 0.05 for 5%)")
    parser.add_argument("--min-seconds", type=int, default=300, help="Minimum wall time to run before stopping (seconds)")
    args = parser.parse_args()

    yard = YardParams(s=args.grid_step)
    yd = build_yard_and_masses(yard)
    cells = yd["cells"]; masses = yd["masses"]
    calib = CalibrationData()
    alpha, beta = calibrate_rake_model(calib.raking_splits)
    b0, b1 = calibrate_bag_model(calib.bagging_times, bag_capacity_lb=35.0)
    Psites = build_candidates(yard.L, yard.W, args.candidate_spacing)

    print(f"Grid cells: {cells.shape[0]}, candidates: {Psites.shape[0]}, K_max={args.K_max}")
    print(f"Calibrated alpha={alpha:.6g}, beta={beta:.3f}")

    chosen, obj, y_sol = solve_pmedian_mip(
        cells, masses, Psites, alpha, beta, args.K_max, time_limit=args.time_limit,
        bag_b0=b0, bag_b1=b1, bag_capacity_lb=35.0, rel_gap=args.rel_gap, min_seconds=args.min_seconds,
    )

    print("\nChosen centers (x, y) in feet:")
    for i, (x, y) in enumerate(chosen):
        print(f"  {i+1:2d}: ({x:6.2f}, {y:6.2f})")
    print(f"\nMILP objective (raking + bagging seconds): {obj:.2f}")

    # Also evaluate the same objective using nearest-center assignment for transparency
    import numpy as np
    D = np.sqrt(((cells[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2))
    idx = np.argmin(D, axis=1)
    dmin = D.min(axis=1)
    raking_eval = float(np.sum(alpha * masses * (dmin ** beta)))
    M_per = np.array([masses[idx == j].sum() for j in range(chosen.shape[0])], dtype=float)
    bags = np.ceil(M_per / 35.0).astype(int)
    bag_eval = float(b1 * M_per.sum() + b0 * bags.sum())
    obj_eval = raking_eval + bag_eval
    print(f"Nearest-assignment eval (raking + bagging seconds): {obj_eval:.2f}")
    print(f"Gap vs MILP eval: {(obj - obj_eval):.2f} sec ({(obj - obj_eval)/(obj_eval+1e-12)*100:.2f}%)")

    # Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "optimal_centers.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in chosen:
            f.write(f"{x:.6f},{y:.6f}\n")
    print(f"Saved chosen centers to {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        sys.exit(1)
