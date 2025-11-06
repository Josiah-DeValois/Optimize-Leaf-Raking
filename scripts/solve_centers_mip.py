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
from math import radians, sqrt
import sys
import argparse
import numpy as np


def mmss_to_seconds(t: str) -> int:
    t = t.strip()
    if ":" in t:
        mm, ss = t.split(":")
        return int(mm) * 60 + int(ss)
    return int(float(t))


def fit_power_time(distances, times):
    d = np.array(distances, dtype=float)
    T = np.array(times, dtype=float)
    mask = (d > 0) & (T > 0)
    d, T = d[mask], T[mask]
    x = np.log(d)
    y = np.log(T)
    A = np.vstack([np.ones_like(x), x]).T
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_k, beta = theta
    k = np.exp(ln_k)
    return k, beta


def build_yard_and_masses(s: float = 1.0):
    """Replicate the mock yard and mass distribution from original.py."""
    # Yard + tree/distribution params (match original.py defaults)
    L, W = 60.0, 40.0
    tree = (15.0, 20.0)
    trunk_radius = 1.5
    phi_deg = 90.0
    axis_ratio = 1.5
    sigma = 10.0
    rho0, A_amp, p_pow = 0.03, 0.28, 2.0

    # Grid
    nx, ny = int(L / s), int(W / s)
    xs = np.linspace(s / 2, L - s / 2, nx)
    ys = np.linspace(s / 2, W - s / 2, ny)  # y: 0(front) → W(back)
    X, Y = np.meshgrid(xs, ys)
    Acell = s * s

    # Anisotropic bump + trunk mask (match original)
    phi = radians(phi_deg)
    sigma_par = axis_ratio * sigma
    sigma_perp = sigma
    dx = X - tree[0]
    dy = Y - tree[1]
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    R_aniso = np.sqrt((u / sigma_par) ** 2 + (v / sigma_perp) ** 2)
    rho_init = rho0 + A_amp * np.exp(-(R_aniso ** p_pow))
    R_circ = np.sqrt(dx ** 2 + dy ** 2)
    rho_init = np.where(R_circ < trunk_radius, rho0, rho_init)

    mass_grid_init = rho_init * Acell
    masses = mass_grid_init.ravel().astype(float)
    cells = np.stack([X.ravel(), Y.ravel()], axis=1)
    return (L, W, s), cells, masses


def calibrate_alpha_beta():
    # Same mock calibration as original.py
    raking_splits = {10: 7, 20: 9, 30: 13, 40: 17, 50: 21, 60: 28, 70: 35, 80: 42, 90: 49, 100: 51}
    distances = sorted(raking_splits.keys())
    times = [raking_splits[d] for d in distances]
    alpha, beta = fit_power_time(distances, times)
    return float(alpha), float(beta)


def build_candidates(L, W, spacing):
    px = np.arange(spacing / 2, L, spacing)
    py = np.arange(spacing / 2, W, spacing)
    Psites = np.array([(x, y) for x in px for y in py], dtype=float)
    return Psites


def solve_pmedian_mip(cells, masses, Psites, alpha, beta, K_max, time_limit=None):
    try:
        import pulp
    except Exception as e:
        print("PuLP not available. Please install with: pip install pulp", file=sys.stderr)
        raise

    n = cells.shape[0]
    m = Psites.shape[0]

    # Precompute costs c_ij = alpha * mass_i * (distance(i,j))**beta
    # Using Euclidean distance
    diff = cells[:, None, :] - Psites[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))  # (n, m)
    C = alpha * (masses[:, None]) * (D ** beta)

    prob = pulp.LpProblem("leaf_raking_pmedian", pulp.LpMinimize)

    # Variables
    y = pulp.LpVariable.dicts("y", (range(m),), lowBound=0, upBound=1, cat=pulp.LpBinary)
    # Assignments as continuous in [0,1] (integrality not required for p-median with fixed y)
    x = pulp.LpVariable.dicts("x", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous)

    # Objective: sum_i sum_j C_ij x_ij
    prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(n) for j in range(m))

    # Each cell assigned to exactly one open center
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(m)) == 1, f"assign_{i}"

    # Cannot assign to a site unless it is open
    for i in range(n):
        for j in range(m):
            prob += x[i][j] <= y[j], f"link_{i}_{j}"

    # Open at most K_max centers (matches original search over K=1..K_max)
    prob += pulp.lpSum(y[j] for j in range(m)) <= int(K_max), "cardinality"

    # Solve
    if time_limit is None:
        solver = pulp.PULP_CBC_CMD(msg=True)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=int(time_limit))
    status = prob.solve(solver)

    # Extract solution
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved", "Feasible"):
        print(f"Solver status: {pulp.LpStatus[status]}")
    y_sol = np.array([pulp.value(y[j]) for j in range(m)])
    open_idx = np.where(y_sol > 0.5)[0]
    chosen = Psites[open_idx]
    obj = pulp.value(prob.objective)
    return chosen, obj, y_sol


def main():
    parser = argparse.ArgumentParser(description="Solve pile-center selection via MILP (PuLP)")
    parser.add_argument("--K-max", type=int, default=5, help="Maximum number of centers to open (<=K)")
    parser.add_argument("--candidate-spacing", type=float, default=10.0, help="Candidate grid spacing (ft)")
    parser.add_argument("--grid-step", type=float, default=1.0, help="Cell grid spacing (ft)")
    parser.add_argument("--time-limit", type=int, default=None, help="Optional solver time limit (seconds)")
    args = parser.parse_args()

    (L, W, s), cells, masses = build_yard_and_masses(args.grid_step)
    alpha, beta = calibrate_alpha_beta()
    Psites = build_candidates(L, W, args.candidate_spacing)

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
