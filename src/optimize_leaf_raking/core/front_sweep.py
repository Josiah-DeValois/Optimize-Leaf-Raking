from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import YardParams
from .raking import baseline_rake_time_to_front


def front_sweep_band_time(
    strip_ft: float,
    use_eff: bool,
    eta: float,
    delta_beta: float,
    skip_rescale: bool,
    yard: YardParams,
    mass_grid_init: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[int, np.ndarray, float, float]:
    """Return (rows_per, T_steps, alpha_eff, beta_eff) for front-sweep.

    - rows_per is number of grid rows swept per pass (strip thickness / s).
    - T_steps is the cumulative time at the end of each pass.
    Effective parameters (alpha_eff, beta_eff) adjust the physics for front sweeping.
    If skip_rescale=False, scale T_steps to match baseline front-distance timing.
    """
    s = yard.s
    ny = mass_grid_init.shape[0]
    rows_per = max(1, int(round(strip_ft / s)))

    # Effective physics for front-sweep
    alpha_eff = float(alpha / (eta if use_eff else 1.0))
    beta_eff = float(max(0.5, beta - (delta_beta if use_eff else 0.0)))

    # Raw pass timeline
    n_steps = int(np.ceil(ny / rows_per))
    T_raw = []
    raw_total = 0.0
    mass_per_row_init = mass_grid_init.sum(axis=1)  # (ny,)
    for k in range(n_steps):
        start = max(0, ny - (k + 1) * rows_per)
        M_above = float(mass_per_row_init[start:].sum())
        dt_k = alpha_eff * M_above * (rows_per * s) ** beta_eff
        raw_total += dt_k
        T_raw.append(raw_total)
    T_raw = np.array(T_raw, dtype=float)

    # Optional rescale to the original baseline (with original alpha, beta)
    if (not skip_rescale) and len(T_raw) > 0 and T_raw[-1] > 0:
        T_target = baseline_rake_time_to_front(
            masses=mass_grid_init.ravel(), Y=Y, alpha=alpha, beta=beta
        )
        scale = T_target / T_raw[-1]
        T_steps = T_raw * scale
    else:
        T_steps = T_raw

    return rows_per, T_steps, alpha_eff, beta_eff


def band_snapshot_with_spillage_columns(
    t_sec: float,
    rows_per: int,
    T_steps: np.ndarray,
    rho_cap: float,
    rho_init: np.ndarray,
    mass_grid_init: np.ndarray,
    Acell: float,
) -> np.ndarray:
    """Column-aware spillage for the active back-strip method.

    Returns a density grid (ny, nx) representing the current state of the yard.
    """
    ny, nx = rho_init.shape
    k = int(np.searchsorted(T_steps, t_sec, side="right"))
    k = min(k, len(T_steps))
    adv_rows = k * rows_per

    # Fraction inside current pass
    if k == len(T_steps):
        frac = 1.0
    else:
        prev_t = 0.0 if k == 0 else float(T_steps[k - 1])
        dt_k = float(T_steps[k] - prev_t)
        frac = 0.0 if dt_k <= 1e-12 else float(np.clip((t_sec - prev_t) / dt_k, 0.0, 1.0))

    rho = rho_init.copy()
    if adv_rows > 0:
        rho[ny - adv_rows :, :] = 0.0

    next_start = max(0, ny - (adv_rows + rows_per))
    next_end = max(-1, ny - adv_rows - 1)
    if next_start <= next_end and frac > 0:
        rho[next_start : next_end + 1, :] *= (1.0 - frac)

    # Mass per column entering band: fully-swept + fractional next strip
    M_full_cols = mass_grid_init[ny - adv_rows : ny, :].sum(axis=0) if adv_rows > 0 else np.zeros(nx)
    M_next_cols = (
        mass_grid_init[next_start : next_end + 1, :].sum(axis=0) if next_start <= next_end else np.zeros(nx)
    )
    M_band_cols = M_full_cols + frac * M_next_cols

    lead_row = min(max(0, ny - adv_rows), ny - 1)
    K_cell = rho_cap * Acell
    remaining = M_band_cols.copy()
    for r in range(lead_row, ny):
        if np.all(remaining <= 1e-12):
            break
        cap_row = np.maximum(0.0, K_cell - rho[r, :] * Acell)
        place = np.minimum(remaining, cap_row)
        rho[r, :] += place / Acell
        remaining -= place

    return rho


def final_band_and_bagging(
    rows_per: int,
    T_steps: np.ndarray,
    rho_cap: float,
    rho_init: np.ndarray,
    mass_grid_init: np.ndarray,
    Acell: float,
    bag_capacity_lb: float,
    b0: float,
    b1: float,
):
    """Compute final band after raking and bagging time based on total mass.

    Returns (rho_final_band, mass_final_band, M_band_total, bag_total_band, T_rake)
    """
    from .bagging import bag_mass_removed  # avoid cycle in imports at module import time

    T_rake = float(T_steps[-1]) if len(T_steps) else 0.0
    rho_final = band_snapshot_with_spillage_columns(
        T_rake, rows_per, T_steps, rho_cap, rho_init, mass_grid_init, Acell
    )
    mass_final_band = rho_final * Acell
    M_band_total = float(mass_final_band.sum())
    # For time accounting in UI we used n_bags*b0 + b1*M_band_total; use same
    n_bags_band = int(np.ceil(M_band_total / bag_capacity_lb))
    bag_total_band = float(n_bags_band * b0 + b1 * M_band_total)
    return rho_final, mass_final_band, M_band_total, bag_total_band, T_rake


def band_bagging_density(
    mass_final_band: np.ndarray,
    Y: np.ndarray,
    Acell: float,
    removed_mass: float,
) -> np.ndarray:
    """Return density grid after removing `removed_mass` front-first (ascending Y)."""
    ny, nx = mass_final_band.shape
    m0 = mass_final_band.ravel().copy()
    removed = max(0.0, min(removed_mass, float(m0.sum())))
    order = np.argsort(Y.ravel())
    m_sorted = m0[order]
    band_cum = np.cumsum(m_sorted)
    idx = int(np.searchsorted(band_cum, removed, side="right"))
    out = m0.copy()
    if idx > 0:
        out[order[:idx]] = 0.0
    if idx < len(order):
        prev_cum = 0.0 if idx == 0 else float(band_cum[idx - 1])
        need = removed - prev_cum
        cid = order[idx]
        out[cid] = max(0.0, out[cid] - need)
    return out.reshape(ny, nx) / Acell

