from dataclasses import dataclass, field
from typing import Dict, Tuple, List


@dataclass
class YardParams:
    L: float = 60.0
    W: float = 40.0
    s: float = 1.0
    tree: Tuple[float, float] = (15.0, 20.0)
    trunk_radius: float = 1.5
    phi_deg: float = 90.0
    axis_ratio: float = 1.5
    sigma: float = 10.0
    rho0: float = 0.03
    A_amp: float = 0.28
    p_pow: float = 2.0


@dataclass
class RakeModel:
    alpha: float
    beta: float


@dataclass
class BagModel:
    b0: float
    b1: float
    bag_capacity_lb: float = 35.0


@dataclass
class FrontSweepParams:
    use_eff: bool = True
    skip_rescale: bool = True
    eta: float = 1.5
    delta_beta: float = 0.2
    strip_ft: float = 2.0


@dataclass
class VizParams:
    fps: int = 2
    seconds_per_frame: int = 60
    rho_cap: float = 3.0
    colors: List[str] = field(default_factory=lambda: [
        "#edf8e9", "#a1d99b", "#31a354", "#fed976", "#fd8d3c", "#e31a1c",
    ])


@dataclass
class CandidateParams:
    spacing: float = 10.0
    K_max: int = 5


@dataclass
class WalkParams:
    speed_ftps: float = 3.5
    order_method: str = "left_to_right"  # or "nn"


@dataclass
class CalibrationData:
    raking_splits: Dict[int, float] = field(default_factory=lambda: {
        10: 7, 20: 9, 30: 13, 40: 17, 50: 21, 60: 28, 70: 35, 80: 42, 90: 49, 100: 51,
    })
    bagging_times: Dict[float, str] = field(default_factory=lambda: {
        0.25: "1:39", 0.50: "1:45", 0.75: "1:45", 1.00: "2:00",
    })

