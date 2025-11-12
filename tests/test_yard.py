import numpy as np
from optimize_leaf_raking.core.config import YardParams
from optimize_leaf_raking.core.yard import build_yard_and_masses


def test_yard_grid_and_masses_shapes():
    yard = YardParams(s=1.0)
    yd = build_yard_and_masses(yard)
    nx, ny = yd["nx"], yd["ny"]
    X, Y = yd["X"], yd["Y"]
    rho = yd["rho_init"]
    mass_grid = yd["mass_grid_init"]
    cells = yd["cells"]
    masses = yd["masses"]

    assert X.shape == (ny, nx)
    assert Y.shape == (ny, nx)
    assert rho.shape == (ny, nx)
    assert mass_grid.shape == (ny, nx)
    assert cells.shape == (nx * ny, 2)
    assert masses.shape == (nx * ny,)
    assert np.all(rho >= 0)
    assert float(masses.sum()) > 0

