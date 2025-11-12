import pytest
import numpy as np


def test_compute_fixed_centers_objective_smoke():
    pulp = pytest.importorskip("pulp")
    # Late import after ensuring pulp exists
    from optimize_leaf_raking.core.config import YardParams, CalibrationData
    from optimize_leaf_raking.core.yard import build_yard_and_masses
    from optimize_leaf_raking.core.calibration import calibrate_rake_model, calibrate_bag_model
    from optimize_leaf_raking.solvers.mip import compute_fixed_centers_objective, build_candidates

    yard = YardParams(s=1.0)
    yd = build_yard_and_masses(yard)
    alpha, beta = calibrate_rake_model(CalibrationData().raking_splits)
    b0, b1 = calibrate_bag_model(CalibrationData().bagging_times, bag_capacity_lb=35.0)

    # Simple centers: pick two arbitrary candidates
    P = build_candidates(yard.L, yard.W, 10.0)
    centers = P[[5, 10]]  # two sites

    obj = compute_fixed_centers_objective(
        yd["cells"], yd["masses"], centers, alpha, beta, b0, b1, 35.0, rel_gap=0.0
    )
    assert isinstance(obj, float) and obj > 0

