from __future__ import annotations

from typing import Dict, Tuple, Callable
import numpy as np


def mmss_to_seconds(t: str) -> int:
    t = t.strip()
    if ":" in t:
        mm, ss = t.split(":")
        return int(mm) * 60 + int(ss)
    return int(float(t))


def fit_power_time(distances, times) -> Tuple[float, float]:
    d = np.array(distances, dtype=float)
    T = np.array(times, dtype=float)
    mask = (d > 0) & (T > 0)
    d, T = d[mask], T[mask]
    x = np.log(d)
    y = np.log(T)
    A = np.vstack([np.ones_like(x), x]).T
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_k, beta = theta
    k = float(np.exp(ln_k))
    return k, float(beta)


def fit_bag_time(fullness_to_time: Dict[float, str]) -> Tuple[Tuple[float, float, float], Callable[[float], float]]:
    fracs = np.array(sorted(fullness_to_time.keys()), dtype=float)
    secs = np.array([mmss_to_seconds(fullness_to_time[f]) for f in fracs], dtype=float)
    X = np.vstack([np.ones_like(fracs), fracs, fracs ** 2]).T
    coef, *_ = np.linalg.lstsq(X, secs, rcond=None)
    s0, s1, s2 = coef
    if s2 < 0:  # prevent concave-down if data is nearly linear
        s2 = 0.0
        X2 = X[:, :2]
        s0, s1 = np.linalg.lstsq(X2, secs, rcond=None)[0]

    def t_of_fraction(f: float) -> float:
        return float(s0 + s1 * f + s2 * (f ** 2))

    return (float(s0), float(s1), float(s2)), t_of_fraction


def calibrate_rake_model(raking_splits: Dict[int, float]) -> Tuple[float, float]:
    distances = sorted(raking_splits.keys())
    times = [raking_splits[d] for d in distances]
    alpha, beta = fit_power_time(distances, times)
    return float(alpha), float(beta)


def calibrate_bag_model(
    bagging_times: Dict[float, str],
    bag_capacity_lb: float,
) -> Tuple[float, float]:
    (b0_hat, b1_hat, b2_hat), t_frac = fit_bag_time(bagging_times)
    b0 = float(t_frac(0.0))
    # sec/lb from slope between 40% and 80% full, normalized by mass step
    b1 = float((t_frac(0.8) - t_frac(0.4)) / (0.4 * bag_capacity_lb))
    return b0, b1

