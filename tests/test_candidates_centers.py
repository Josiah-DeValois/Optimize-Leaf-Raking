import numpy as np

from optimize_leaf_raking.core.config import YardParams, CalibrationData
from optimize_leaf_raking.core.yard import build_yard_and_masses
from optimize_leaf_raking.core.calibration import calibrate_rake_model, calibrate_bag_model
from optimize_leaf_raking.solvers.mip import build_candidates, compute_fixed_centers_objective
from optimize_leaf_raking.core.centers import centers_opt_discrete


def test_build_candidates_count_and_bounds():
    yard = YardParams()
    P = build_candidates(yard.L, yard.W, spacing=10.0)
    # 6 x 4 grid for L=60, W=40 at spacing 10
    assert P.shape == (24, 2)
    assert np.isclose(P.min(0), [5.0, 5.0]).all()
    assert np.isclose(P.max(0), [55.0, 35.0]).all()


def test_centers_opt_discrete_and_fixed_assignment_objective():
    yard = YardParams(s=1.0)
    yd = build_yard_and_masses(yard)
    alpha, beta = calibrate_rake_model(CalibrationData().raking_splits)
    b0, b1 = calibrate_bag_model(CalibrationData().bagging_times, bag_capacity_lb=35.0)

    centers, enum_obj = centers_opt_discrete(
        yd["cells"], yd["masses"], yard, 10.0, alpha, beta, 5,
        bag_b0=b0, bag_b1=b1, bag_capacity_lb=35.0,
    )
    assert centers.ndim == 2 and centers.shape[1] == 2
    assert isinstance(enum_obj, float)
    assert enum_obj > 0

    # Fixed-centers assignment should not exceed the nearest-assignment objective
    obj_fixed = compute_fixed_centers_objective(
        yd["cells"], yd["masses"], centers, alpha, beta, b0, b1, 35.0, rel_gap=0.0
    )
    assert obj_fixed <= enum_obj + 1e-6

