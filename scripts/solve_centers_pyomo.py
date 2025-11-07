#!/usr/bin/env python3
"""
Solve pile-center selection with Pyomo using the same objective as our MILP:
  total = raking + bag stuffing + bag setups

Variables
  y_j ∈ {0,1}   open candidate site j
  x_ij ∈ [0,1]  assignment of cell i to site j
  t_j  ≥ 0      total mass assigned to site j
  B_j  ∈ Z+     number of bags used at site j

Objective
  min sum_{i,j} alpha * m_i * d_ij^beta * x_ij
      + b1 * sum_j t_j
      + b0 * sum_j B_j

Constraints
  ∑_j x_ij = 1                ∀ i
  x_ij ≤ y_j                  ∀ i,j
  t_j = ∑_i m_i x_ij          ∀ j
  t_j ≤ C * B_j               ∀ j
  ∑_j y_j ≤ K_max

Outputs
  - Prints objective (seconds and minutes)
  - Prints chosen centers (coordinates), mass per center, and bags per center
  - Saves chosen centers to results/optimal_centers_pyomo.csv

Run from repo root (after optional editable install):
  python3 scripts/solve_centers_pyomo.py \
    --K-max 5 --candidate-spacing 10 --grid-step 1 \
    --rel-gap 0.05 --min-seconds 300

If imports fail, ensure Pyomo is installed: pip install pyomo
Install a solver such as CBC (recommended) or GLPK.
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimize_leaf_raking.core.config import YardParams, CalibrationData
from optimize_leaf_raking.core.yard import build_yard_and_masses
from optimize_leaf_raking.core.calibration import calibrate_rake_model, calibrate_bag_model
from optimize_leaf_raking.solvers.mip import build_candidates


def build_pyomo_model(cells, masses, Psites, alpha, beta, K_max, b0, b1, capacity):
    from pyomo.environ import (
        ConcreteModel, Set, RangeSet, Var, Objective, Constraint, NonNegativeReals,
        Binary, NonNegativeIntegers, summation, value, minimize
    )

    n = cells.shape[0]
    m = Psites.shape[0]

    # Precompute raking costs c_ij
    diff = cells[:, None, :] - Psites[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))  # (n, m)
    C = alpha * (masses[:, None]) * (D ** beta)

    M = ConcreteModel()
    M.I = RangeSet(0, n - 1)
    M.J = RangeSet(0, m - 1)

    M.x = Var(M.I, M.J, bounds=(0.0, 1.0))
    M.y = Var(M.J, within=Binary)
    M.B = Var(M.J, within=NonNegativeIntegers)
    M.t = Var(M.J, within=NonNegativeReals)

    # Assignment
    def assign_rule(M, i):
        return sum(M.x[i, j] for j in M.J) == 1.0
    M.Assign = Constraint(M.I, rule=assign_rule)

    # Link
    def link_rule(M, i, j):
        return M.x[i, j] <= M.y[j]
    M.Link = Constraint(M.I, M.J, rule=link_rule)

    # Mass aggregation per site
    def mass_rule(M, j):
        return M.t[j] == sum(float(masses[i]) * M.x[i, j] for i in M.I)
    M.Mass = Constraint(M.J, rule=mass_rule)

    # Bag capacity
    def cap_rule(M, j):
        return M.t[j] <= float(capacity) * M.B[j]
    M.Cap = Constraint(M.J, rule=cap_rule)

    # Cardinality
    M.Card = Constraint(expr=sum(M.y[j] for j in M.J) <= int(K_max))

    # Objective
    M.Obj = Objective(
        expr=sum(float(C[i, j]) * M.x[i, j] for i in M.I for j in M.J)
        + float(b1) * sum(M.t[j] for j in M.J)
        + float(b0) * sum(M.B[j] for j in M.J),
        sense=minimize,
    )

    return M, D


def pick_solver(rel_gap: float, time_limit: int | None):
    from pyomo.opt import SolverFactory
    # Try CBC → GLPK → HiGHS
    for name in ("cbc", "glpk", "highs"):
        opt = SolverFactory(name)
        if opt is not None and opt.available():
            # Apply common gap/time options where supported
            if name == "cbc":
                opt.options["ratioGap"] = rel_gap
                if time_limit:
                    opt.options["seconds"] = int(time_limit)
            elif name == "glpk":
                # GLPK accepts mipgap (relative) and tmlim (seconds)
                opt.options["mipgap"] = rel_gap
                if time_limit:
                    opt.options["tmlim"] = int(time_limit)
            elif name == "highs":
                opt.options["mip_rel_gap"] = rel_gap
                if time_limit:
                    opt.options["time_limit"] = float(time_limit)
            return opt, name
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Solve pile-center selection via Pyomo (MILP)")
    ap.add_argument("--K-max", type=int, default=5, help="Maximum number of centers to open (<=K)")
    ap.add_argument("--candidate-spacing", type=float, default=10.0, help="Candidate grid spacing (ft)")
    ap.add_argument("--grid-step", type=float, default=1.0, help="Cell grid spacing (ft)")
    ap.add_argument("--rel-gap", type=float, default=0.05, help="Relative optimality gap target")
    ap.add_argument("--min-seconds", type=int, default=300, help="Minimum wall time before stopping")
    ap.add_argument("--time-limit", type=int, default=None, help="Optional hard time limit (seconds)")
    args = ap.parse_args()

    yard = YardParams(s=args.grid_step)
    yd = build_yard_and_masses(yard)

    calib = CalibrationData()
    alpha, beta = calibrate_rake_model(calib.raking_splits)
    b0, b1 = calibrate_bag_model(calib.bagging_times, bag_capacity_lb=35.0)

    Psites = build_candidates(yard.L, yard.W, spacing=args.candidate_spacing)

    print(f"Grid cells: {yd['cells'].shape[0]}, candidates: {Psites.shape[0]}, K_max={args.K_max}")
    print(f"Calibrated alpha={alpha:.6g}, beta={beta:.3f}; bag b0={b0:.2f}s, b1={b1:.4f}s/lb")

    try:
        from pyomo.environ import value
    except Exception:
        print("Pyomo not available. Install with: pip install pyomo", file=sys.stderr)
        sys.exit(1)

    M, D = build_pyomo_model(
        yd["cells"], yd["masses"], Psites, alpha, beta, args.K_max, b0, b1, 35.0
    )
    opt, name = pick_solver(args.rel_gap, args.time_limit)
    if opt is None:
        print("No MILP solver available for Pyomo (tried cbc, glpk, highs).", file=sys.stderr)
        sys.exit(2)
    print(f"Using solver: {name}")

    t0 = time.time()
    res = opt.solve(M)
    elapsed = time.time() - t0
    # Enforce minimum wall time (align with other script semantics)
    if elapsed < float(args.min_seconds) and (args.time_limit is None or args.time_limit >= args.min_seconds):
        time.sleep(float(args.min_seconds) - elapsed)

    obj = float(M.Obj())
    print(f"\nPyomo objective (raking + bagging seconds): {obj:.2f}  ({obj/60.0:.2f} min)")

    # Extract chosen centers
    y_vals = np.array([float(M.y[j]()) for j in M.J])
    open_idx = np.where(y_vals > 0.5)[0]
    chosen = Psites[open_idx]
    print("\nChosen centers (x, y) in feet:")
    for k, (x, y) in enumerate(chosen):
        print(f"  {k+1:2d}: ({x:6.2f}, {y:6.2f})")

    # Mass per open site and bags used (from t_j, B_j)
    t_vals = np.array([float(M.t[j]()) for j in M.J])
    B_vals = np.array([int(round(float(M.B[j]()))) for j in M.J])
    if open_idx.size:
        print("\nPer-site totals:")
        for j in open_idx:
            print(f"  j={j:2d}: mass={t_vals[j]:8.2f} lb   bags={B_vals[j]:3d}")

    # Save centers
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "optimal_centers_pyomo.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in chosen:
            f.write(f"{x:.6f},{y:.6f}\n")
    print(f"\nSaved chosen centers to {out_csv}")


if __name__ == "__main__":
    main()

