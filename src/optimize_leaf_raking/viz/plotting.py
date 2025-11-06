from __future__ import annotations

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_cmap(colors: List[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("leaf_hot", colors, N=256)


def draw_density(
    ax: plt.Axes,
    rho: np.ndarray,
    extent: Tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: LinearSegmentedColormap,
):
    return ax.imshow(
        rho,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
    )

