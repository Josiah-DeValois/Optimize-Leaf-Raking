from __future__ import annotations

from math import sqrt, pi
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import YardParams


def euclid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between rows of A and B."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def _ensure_centers(centers: np.ndarray, yard: Optional[YardParams]) -> np.ndarray:
    if centers.size == 0:
        if yard is None:
            raise ValueError("centers is empty and yard is None; provide a center or yard for fallback")
        return np.array([[yard.L / 2.0, yard.W / 2.0]], dtype=float)
    return centers


def baseline_rake_time_to_centers(
    cells: np.ndarray,
    masses: np.ndarray,
    centers: np.ndarray,
    alpha: float,
    beta: float,
    yard: Optional[YardParams] = None,
) -> float:
    centers = _ensure_centers(centers, yard)
    D = euclid(cells, centers)
    dmin = D.min(axis=1)
    return float(np.sum(alpha * masses * (dmin ** beta)))


def baseline_rake_time_to_front(
    masses: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    beta: float,
    alpha_eff: Optional[float] = None,
    beta_eff: Optional[float] = None,
) -> float:
    a = alpha if alpha_eff is None else alpha_eff
    b = beta if beta_eff is None else beta_eff
    dfront = Y.ravel()
    return float(np.sum(a * masses * (dfront ** b)))


def radial_arrival_times_calibrated(
    cells: np.ndarray,
    masses: np.ndarray,
    centers: np.ndarray,
    alpha: float,
    beta: float,
    angle_bins: int = 24,
    yard: Optional[YardParams] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    """Compute arrival time per cell for outside-in raking with ray discretization.

    Returns: (t_arrive, pile_id, per_pile)
      - t_arrive: (N,) array of times
      - pile_id: (N,) nearest pile index
      - per_pile: list of dicts with keys 't', 'm', 'M_total'
    """
    centers = _ensure_centers(centers, yard)
    D = euclid(cells, centers)
    pile_id = np.argmin(D, axis=1)
    r_to_pile = D[np.arange(len(cells)), pile_id]

    bin_w = 2 * np.pi / angle_bins
    t_arrive = np.full(len(cells), np.inf, dtype=float)

    for p in range(centers.shape[0]):
        cx, cy = centers[p]
        sel = (pile_id == p)
        if not np.any(sel):
            continue
        idxs = np.where(sel)[0]
        pts = cells[sel]
        vx = pts[:, 0] - cx
        vy = pts[:, 1] - cy
        theta = np.arctan2(vy, vx)
        bins = np.clip(np.floor((theta + np.pi) / bin_w).astype(int), 0, angle_bins - 1)
        for b in range(angle_bins):
            ray_mask = (bins == b)
            if not np.any(ray_mask):
                continue
            ray_idxs = idxs[ray_mask]
            r = r_to_pile[ray_idxs]
            order = np.argsort(-r)
            ray_idxs = ray_idxs[order]
            r = r[order]
            m = masses[ray_idxs]
            r_ext = np.concatenate([r, np.array([0.0])])
            M_cum = np.cumsum(m[::-1])[::-1]
            dt = alpha * M_cum * ((r_ext[:-1] - r_ext[1:]) ** beta)
            T = np.cumsum(dt)
            t_arrive[ray_idxs] = T

    raw_total = float(np.nanmax(t_arrive[np.isfinite(t_arrive)])) if np.isfinite(t_arrive).any() else 0.0
    target_total = baseline_rake_time_to_centers(cells, masses, centers, alpha, beta, yard)
    scale = (target_total / raw_total) if raw_total > 0 else 1.0
    t_arrive *= scale

    per_pile: List[Dict[str, np.ndarray]] = []
    for p in range(centers.shape[0]):
        mask = (pile_id == p)
        mp = masses[mask]
        per_pile.append({
            "t": t_arrive[mask],
            "m": mp,
            "M_total": float(mp.sum()),
        })
    return t_arrive, pile_id, per_pile


def deposit_pile_disks_from_masses(
    centers: np.ndarray,
    M_list: List[float],
    xs: np.ndarray,
    ys: np.ndarray,
    rho_cap: float,
    acell: float,
) -> np.ndarray:
    """Deposit circular 'disks' of density for pile masses onto a grid.

    Returns rho grid (ny, nx).
    """
    nx = len(xs)
    ny = len(ys)
    acc = np.zeros((ny, nx), dtype=float)
    if centers.size == 0:
        return acc
    Xc, Yc = np.meshgrid(xs, ys)
    for p, (cx, cy) in enumerate(centers):
        M = float(M_list[p])
        if M <= 0:
            continue
        r = sqrt(M / (pi * rho_cap))
        mask = (Xc - cx) ** 2 + (Yc - cy) ** 2 <= r ** 2
        area = float(mask.sum()) * acell
        if area <= 0:
            continue
        dens = min(rho_cap, M / area)
        acc[mask] += dens
    return acc


def deposit_piles_disk_from_arrivals(
    centers: np.ndarray,
    per_pile: List[Dict[str, np.ndarray]],
    t_sec: float,
    xs: np.ndarray,
    ys: np.ndarray,
    rho_cap: float,
    acell: float,
) -> np.ndarray:
    Ms: List[float] = []
    for p in range(len(per_pile)):
        tp = per_pile[p]["t"]
        mp = per_pile[p]["m"]
        Ms.append(float(mp[tp <= t_sec].sum()))
    return deposit_pile_disks_from_masses(centers, Ms, xs, ys, rho_cap, acell)

