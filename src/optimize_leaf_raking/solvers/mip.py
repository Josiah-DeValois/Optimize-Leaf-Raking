from __future__ import annotations

from typing import Tuple
import numpy as np
import time


def build_candidates(L: float, W: float, spacing: float) -> np.ndarray:
    px = np.arange(spacing / 2.0, L, spacing)
    py = np.arange(spacing / 2.0, W, spacing)
    return np.array([(x, y) for x in px for y in py], dtype=float)


def solve_pmedian_mip(
    cells: np.ndarray,
    masses: np.ndarray,
    Psites: np.ndarray,
    alpha: float,
    beta: float,
    K_max: int,
    time_limit: int | None = None,
    *,
    bag_b0: float = 0.0,
    bag_b1: float = 0.0,
    bag_capacity_lb: float = 35.0,
    rel_gap: float = 0.05,
    min_seconds: int = 300,
) -> Tuple[np.ndarray, float, np.ndarray]:
    try:
        import pulp
    except Exception:
        raise ImportError("PuLP not available. Install with: pip install pulp")

    n = cells.shape[0]
    m = Psites.shape[0]

    # Costs c_ij = alpha * mass_i * dist(i,j)^beta  (raking)
    # Bagging adds: sum_j [ bag_b1 * (sum_i mass_i * x_ij) + bag_b0 * B_j ]
    # with integer B_j and capacity constraint: sum_i mass_i * x_ij <= bag_capacity_lb * B_j
    diff = cells[:, None, :] - Psites[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))  # (n, m)
    C = alpha * (masses[:, None]) * (D ** beta)

    prob = pulp.LpProblem("leaf_raking_pmedian", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", (range(m),), lowBound=0, upBound=1, cat=pulp.LpBinary)
    # Changed to binary variables for assignmentS, changed back to continuous for speed solving
    x = pulp.LpVariable.dicts("x", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    # Integer number of bags per open site
    B = pulp.LpVariable.dicts("B", (range(m),), lowBound=0, cat=pulp.LpInteger)

    # Raking + bagging stuffing time (bag_b1 * mass) + bag setup per bag (bag_b0 * B_j)
    prob += (
        pulp.lpSum((C[i, j] + bag_b1 * masses[i]) * x[i][j] for i in range(n) for j in range(m))
        + bag_b0 * pulp.lpSum(B[j] for j in range(m))
    )

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(m)) == 1, f"assign_{i}"
    for i in range(n):
        for j in range(m):
            prob += x[i][j] <= y[j], f"link_{i}_{j}"
    # Capacity-link for bags: sum_i mass_i * x_ij <= bag_capacity_lb * B_j
    for j in range(m):
        prob += pulp.lpSum(masses[i] * x[i][j] for i in range(n)) <= bag_capacity_lb * B[j], f"capacity_{j}"
    # Optional tightening: B_j == 0 when y_j == 0
    M_total = float(masses.sum())
    B_ub = int(np.ceil(M_total / max(1e-9, bag_capacity_lb)))
    for j in range(m):
        prob += B[j] <= B_ub * y[j], f"bags_open_{j}"
    prob += pulp.lpSum(y[j] for j in range(m)) <= int(K_max), "cardinality"

    # Configure CBC to target a relative gap. Use API if available, otherwise pass CLI options.
    solver_kwargs = {"msg": True}
    if time_limit:  # if explicit, honor it; otherwise let gap control termination
        solver_kwargs["timeLimit"] = int(time_limit)
    # Try gapRel argument first
    try:
        solver = pulp.PULP_CBC_CMD(gapRel=rel_gap, **solver_kwargs)
    except TypeError:
        # Fallback via options
        solver = pulp.PULP_CBC_CMD(options=["-ratioGap", str(rel_gap)], **solver_kwargs)

    t0 = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0
    # Enforce minimum wall time if requested and we finished early (gap achieved fast)
    if (min_seconds or 0) and elapsed < float(min_seconds) and (time_limit is None or float(time_limit) >= float(min_seconds)):
        time.sleep(float(min_seconds) - elapsed)

    y_sol = np.array([pulp.value(y[j]) for j in range(m)])
    open_idx = np.where(y_sol > 0.5)[0]
    chosen = Psites[open_idx]
    obj = float(pulp.value(prob.objective))
    return chosen, obj, y_sol


def compute_fixed_centers_objective(
    cells: np.ndarray,
    masses: np.ndarray,
    centers: np.ndarray,
    alpha: float,
    beta: float,
    bag_b0: float,
    bag_b1: float,
    bag_capacity_lb: float,
    rel_gap: float = 0.0,
) -> float:
    """Compute MILP objective for best assignment to fixed centers.

    Minimizes raking + bag stuffing + bag setup given open centers.
    Returns objective value in seconds. Requires PuLP/CBC installed.
    """
    try:
        import pulp
    except Exception:
        raise ImportError("PuLP not available. Install with: pip install pulp")

    n = cells.shape[0]
    K = centers.shape[0]
    if K == 0:
        return 0.0

    diff = cells[:, None, :] - centers[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    Ccost = alpha * (masses[:, None]) * (D ** beta)

    x = pulp.LpVariable.dicts("x", (range(n), range(K)), lowBound=0, upBound=1, cat=pulp.LpContinuous)
    B = pulp.LpVariable.dicts("B", (range(K),), lowBound=0, cat=pulp.LpInteger)
    prob = pulp.LpProblem("fixed_centers_assign", pulp.LpMinimize)

    M_total = float(masses.sum())
    prob += (
        pulp.lpSum(Ccost[i, j] * x[i][j] for i in range(n) for j in range(K))
        + bag_b1 * M_total
        + bag_b0 * pulp.lpSum(B[j] for j in range(K))
    )
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(K)) == 1, f"assign_{i}"
    for j in range(K):
        prob += pulp.lpSum(masses[i] * x[i][j] for i in range(n)) <= bag_capacity_lb * B[j], f"cap_{j}"

    try:
        solver = pulp.PULP_CBC_CMD(gapRel=rel_gap, msg=True)
    except TypeError:
        solver = pulp.PULP_CBC_CMD(options=["-ratioGap", str(rel_gap)], msg=True)
    prob.solve(solver)
    return float(pulp.value(prob.objective))
