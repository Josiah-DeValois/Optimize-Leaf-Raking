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


from optimize_leaf_raking.core.config import YardParams, CalibrationData
from optimize_leaf_raking.core.calibration import calibrate_rake_model
from optimize_leaf_raking.core.yard import build_yard_and_masses
from optimize_leaf_raking.solvers.mip import build_candidates, solve_pmedian_mip


def main():
    parser = argparse.ArgumentParser(description="Solve pile-center selection via MILP (PuLP)")
    parser.add_argument("--K-max", type=int, default=5, help="Maximum number of centers to open (<=K)")
    parser.add_argument("--candidate-spacing", type=float, default=10.0, help="Candidate grid spacing (ft)")
    parser.add_argument("--grid-step", type=float, default=1.0, help="Cell grid spacing (ft)")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional solver time limit (seconds)")
    args = parser.parse_args()

    yard = YardParams(s=args.grid_step)
    yd = build_yard_and_masses(yard)
    cells = yd["cells"]; masses = yd["masses"]
    alpha, beta = calibrate_rake_model(CalibrationData().raking_splits)
    Psites = build_candidates(yard.L, yard.W, args.candidate_spacing)

    print(f"Grid cells: {cells.shape[0]}, candidates: {Psites.shape[0]}, K_max={args.K_max}")
    print(f"Calibrated alpha={alpha:.6g}, beta={beta:.3f}")

    chosen, obj, y_sol = solve_pmedian_mip(
        cells, masses, Psites, alpha, beta, args.K_max, time_limit=args.time_limit
    )

    print("\nChosen centers (x, y) in feet:")
    for i, (x, y) in enumerate(chosen):
        print(f"  {i+1:2d}: ({x:6.2f}, {y:6.2f})")
    print(f"\nObjective (total raking seconds): {obj:.2f}")

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
