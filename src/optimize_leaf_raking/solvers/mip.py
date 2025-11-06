from __future__ import annotations

from typing import Tuple
import numpy as np


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
) -> Tuple[np.ndarray, float, np.ndarray]:
    try:
        import pulp
    except Exception:
        raise ImportError("PuLP not available. Install with: pip install pulp")

    n = cells.shape[0]
    m = Psites.shape[0]

    # Costs c_ij = alpha * mass_i * dist(i,j)^beta
    diff = cells[:, None, :] - Psites[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))  # (n, m)
    C = alpha * (masses[:, None]) * (D ** beta)

    prob = pulp.LpProblem("leaf_raking_pmedian", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", (range(m),), lowBound=0, upBound=1, cat=pulp.LpBinary)
    x = pulp.LpVariable.dicts("x", (range(n), range(m)), lowBound=0, upBound=1, cat=pulp.LpContinuous)

    prob += pulp.lpSum(C[i, j] * x[i][j] for i in range(n) for j in range(m))

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(m)) == 1, f"assign_{i}"
    for i in range(n):
        for j in range(m):
            prob += x[i][j] <= y[j], f"link_{i}_{j}"
    prob += pulp.lpSum(y[j] for j in range(m)) <= int(K_max), "cardinality"

    solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=int(time_limit) if time_limit else None)
    status = prob.solve(solver)

    y_sol = np.array([pulp.value(y[j]) for j in range(m)])
    open_idx = np.where(y_sol > 0.5)[0]
    chosen = Psites[open_idx]
    obj = float(pulp.value(prob.objective))
    return chosen, obj, y_sol

