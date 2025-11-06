from __future__ import annotations

from dataclasses import asdict
from math import radians
from typing import Tuple
import numpy as np

from .config import YardParams


def build_grid(params: YardParams):
    """Return (xs, ys, X, Y, nx, ny, Acell)."""
    L, W, s = params.L, params.W, params.s
    nx, ny = int(L / s), int(W / s)
    xs = np.linspace(s / 2, L - s / 2, nx)
    ys = np.linspace(s / 2, W - s / 2, ny)  # y: 0(front)→W(back)
    X, Y = np.meshgrid(xs, ys)
    Acell = s * s
    return xs, ys, X, Y, nx, ny, Acell


def build_mass_distribution(params: YardParams, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute rho_init over grid and return mass_grid_init (lb) per cell."""
    phi = radians(params.phi_deg)
    sigma_par = params.axis_ratio * params.sigma
    sigma_perp = params.sigma

    dx = X - params.tree[0]
    dy = Y - params.tree[1]
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    R_aniso = np.sqrt((u / sigma_par) ** 2 + (v / sigma_perp) ** 2)
    rho_init = params.rho0 + params.A_amp * np.exp(-(R_aniso ** params.p_pow))

    # Trunk mask to base density
    R_circ = np.sqrt(dx ** 2 + dy ** 2)
    rho_init = np.where(R_circ < params.trunk_radius, params.rho0, rho_init)
    return rho_init


def build_yard_and_masses(params: YardParams):
    """Construct grid and masses arrays.

    Returns: dict with keys
      - params (YardParams)
      - xs, ys, X, Y, nx, ny, Acell
      - rho_init (lb/ft^2), mass_grid_init (lb)
      - cells (N,2) centers in feet
      - masses (N,) vector
    """
    xs, ys, X, Y, nx, ny, Acell = build_grid(params)
    rho_init = build_mass_distribution(params, X, Y)
    mass_grid_init = rho_init * Acell
    cells = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)
    masses = mass_grid_init.ravel().astype(float)
    return {
        "params": params,
        "xs": xs,
        "ys": ys,
        "X": X,
        "Y": Y,
        "nx": nx,
        "ny": ny,
        "Acell": Acell,
        "rho_init": rho_init,
        "mass_grid_init": mass_grid_init,
        "cells": cells,
        "masses": masses,
    }

