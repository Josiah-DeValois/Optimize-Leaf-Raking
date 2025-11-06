from __future__ import annotations

import numpy as np


def bag_mass_removed(M_total: float, t: float, capacity_lb: float, b0: float, b1: float) -> float:
    """Simulate bagging with setup time b0 per bag and fill rate b1 sec/lb.

    Returns mass removed within time t.
    """
    if M_total <= 1e-12 or t <= 0:
        return 0.0
    M_rem = float(M_total)
    t_rem = float(t)
    removed = 0.0
    while M_rem > 1e-12 and t_rem > 1e-12:
        cap = min(capacity_lb, M_rem)
        if t_rem <= b0:
            break
        t_rem -= b0
        fill_mass = min(cap, t_rem / b1)
        removed += fill_mass
        M_rem -= fill_mass
        t_rem -= fill_mass * b1
        if fill_mass < cap:
            break
    return float(removed)


def compute_pile_order(centers: np.ndarray, method: str = "left_to_right") -> np.ndarray:
    n = centers.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if method == "nn":
        left = int(np.argmin(centers[:, 0]))
        order = [left]
        used = {left}
        cur = left
        for _ in range(n - 1):
            remaining = [i for i in range(n) if i not in used]
            d = np.linalg.norm(centers[remaining] - centers[cur], axis=1)
            nxt = remaining[int(np.argmin(d))]
            order.append(nxt)
            used.add(nxt)
            cur = nxt
        return np.array(order, dtype=int)
    return np.argsort(centers[:, 0])


def walk_times_from_order(centers: np.ndarray, order: np.ndarray, v_walk_ftps: float) -> np.ndarray:
    if centers.size == 0 or order.size == 0:
        return np.array([])
    times = np.zeros(len(order))
    for j in range(1, len(order)):
        a = centers[order[j - 1]]
        b = centers[order[j]]
        dist = float(np.linalg.norm(a - b))
        times[j] = dist / max(1e-6, v_walk_ftps)
    return times

