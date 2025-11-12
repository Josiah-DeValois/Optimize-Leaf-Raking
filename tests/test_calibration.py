import math
from optimize_leaf_raking.core.config import CalibrationData
from optimize_leaf_raking.core.calibration import calibrate_rake_model, calibrate_bag_model


def test_calibration_values_reasonable():
    calib = CalibrationData()
    alpha, beta = calibrate_rake_model(calib.raking_splits)
    b0, b1 = calibrate_bag_model(calib.bagging_times, bag_capacity_lb=35.0)

    assert alpha > 0
    assert 0.5 < beta < 1.5
    assert b0 > 0
    assert b1 > 0

