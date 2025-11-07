from __future__ import annotations

import itertools
from typing import Tuple
import numpy as np

from .config import YardParams
from .raking import euclid


def centers_bf(yard: YardParams, masses: np.ndarray, bag_capacity_lb: float) -> np.ndarray:
    M_total = float(masses.sum())
    k = max(1, int(round(M_total / (2 * bag_capacity_lb))))
    x_bounds = np.linspace(0, yard.L, k + 1)
    return np.array([[(x_bounds[j] + x_bounds[j + 1]) / 2.0, yard.W / 2.0] for j in range(k)], dtype=float)


def centers_micro(cells: np.ndarray, masses: np.ndarray, bag_capacity_lb: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M_total = float(masses.sum())
    k0 = max(1, int(round(M_total / (2 * bag_capacity_lb))))
    probs = masses / (M_total + 1e-12)
    idx0 = rng.choice(len(masses), size=k0, replace=False, p=probs)
    centers = cells[idx0].copy()

    for _ in range(8):
        D2 = ((cells[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(D2, axis=1)
        m_c = np.array([masses[labels == j].sum() for j in range(centers.shape[0])], dtype=float)
        to_split = np.where(m_c > 2 * bag_capacity_lb)[0]
        new_centers = []
        for j in range(centers.shape[0]):
            mask = labels == j
            if not np.any(mask):
                new_centers.append(centers[j])
                continue
            pts = cells[mask]
            w = masses[mask]
            if j in to_split:
                a = pts[np.argmax(((pts - pts.mean(0)) ** 2).sum(1))]
                b = pts[np.argmax(((pts - a) ** 2).sum(1))]
                ca, cb = a.copy(), b.copy()
                for _ in range(5):
                    da = ((pts - ca) ** 2).sum(1)
                    db = ((pts - cb) ** 2).sum(1)
                    lab = (da <= db)
                    wa = w[lab].sum() + 1e-9
                    wb = w[~lab].sum() + 1e-9
                    ca = np.array([np.sum(pts[lab, 0] * w[lab]) / wa, np.sum(pts[lab, 1] * w[lab]) / wa])
                    cb = np.array([np.sum(pts[~lab, 0] * w[~lab]) / wb, np.sum(pts[~lab, 1] * w[~lab]) / wb])
                new_centers.append(ca)
                new_centers.append(cb)
            else:
                cx = np.average(pts[:, 0], weights=w)
                cy = np.average(pts[:, 1], weights=w)
                new_centers.append([cx, cy])
        centers = np.array(new_centers)
    return centers


def centers_opt_discrete(
    cells: np.ndarray,
    masses: np.ndarray,
    yard: YardParams,
    candidate_spacing: float,
    alpha: float,
    beta: float,
    K_max: int,
    *,
    bag_b0: float = 0.0,
    bag_b1: float = 0.0,
    bag_capacity_lb: float = 35.0,
) -> Tuple[np.ndarray, float]:
    px = np.arange(candidate_spacing / 2, yard.L, candidate_spacing)
    py = np.arange(candidate_spacing / 2, yard.W, candidate_spacing)
    Psites = np.array(list(itertools.product(px, py)), dtype=float)
    if Psites.size == 0:
        center = np.array([[yard.L / 2.0, yard.W / 2.0]])
        return center, 0.0
    D = np.sqrt(((cells[:, None, :] - Psites[None, :, :]) ** 2).sum(axis=2))
    n = cells.shape[0]
    total_mass = float(masses.sum())
    best_total: float | None = None
    best_combo = None

    # Constant stuffing time across all combos (included to match MILP numerically)
    stuffing_s = float(bag_b1 * total_mass)
    for K in range(1, K_max + 1):
        for combo in itertools.combinations(range(len(Psites)), K):
            Csub = D[:, combo]                 # (n, K)
            idx = np.argmin(Csub, axis=1)      # nearest center index 0..K-1
            dmin = Csub[np.arange(n), idx]

            # Raking term (same functional form as MILP)
            rake_s = float(np.sum(alpha * masses * (dmin ** beta)))

            # Mass per chosen center and bag setup time
            mass_per_center = np.zeros(K, dtype=float)
            for i in range(n):
                mass_per_center[idx[i]] += masses[i]
            bags_per_center = np.ceil(mass_per_center / max(bag_capacity_lb, 1e-9))
            bag_setup_s = float(bag_b0 * bags_per_center.sum())

            # Full objective: raking + constant stuffing + bag setups
            total_s = rake_s + stuffing_s + bag_setup_s
            if (best_total is None) or (total_s < best_total):
                best_total = total_s
                best_combo = combo

    centers = Psites[list(best_combo)] if best_combo is not None else np.array([[yard.L / 2.0, yard.W / 2.0]])
    return centers, float(best_total if best_total is not None else 0.0)
