#!/usr/bin/env python3
"""
Run the 2x2 animated comparison with interactive front-sweep controls.

Usage:
  python3 scripts/run_viz.py [--show] [--save] [--format mp4|gif] [--fps N] [--spf seconds]

Outputs:
  - Optionally shows a window (default: --show)
  - Optionally saves an animation to results/figures/heatmaps_2x2.(mp4|gif)
"""

from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

# Ensure `src/` is on PYTHONPATH when running from repo
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimize_leaf_raking.core.config import (
    YardParams,
    VizParams,
    CandidateParams,
    WalkParams,
    FrontSweepParams,
    CalibrationData,
)
from optimize_leaf_raking.viz.animate import build_initial_state, make_figure_and_animation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="Show the window")
    p.add_argument("--save", action="store_true", help="Save the animation to results/figures")
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="Output format when saving")
    p.add_argument("--fps", type=int, default=2, help="Frames per second for animation display")
    p.add_argument("--spf", type=int, default=60, help="Seconds per animated frame")
    p.add_argument("--style", choices=["minimal", "interactive"], default="minimal", help="Figure style")
    args = p.parse_args()

    # Parameters (defaults match original.py)
    yard = YardParams()
    viz = VizParams(fps=args.fps, seconds_per_frame=args.spf)
    cand = CandidateParams(spacing=10.0, K_max=5)
    walk = WalkParams(speed_ftps=3.5, order_method="left_to_right")
    front = FrontSweepParams(use_eff=True, skip_rescale=True, eta=1.5, delta_beta=0.2, strip_ft=2.0)
    calib = CalibrationData()

    state = build_initial_state(yard, viz, cand, walk, front, calib)
    # Print totals to terminal (minutes)
    from optimize_leaf_raking.viz.animate import panel_totals_seconds, panel_totals_seconds_no_walk
    names = ["BF", "Front", "Micro", "Opt (disc)"]
    totals_walk = panel_totals_seconds(state)
    totals_nowalk = panel_totals_seconds_no_walk(state)
    print("Totals including walking (min):")
    for nm, m in zip(names, [t/60.0 for t in totals_walk]):
        print(f"  {nm:10}: {m:6.1f}")
    print("\nTotals without walking (min):")
    for nm, m in zip(names, [t/60.0 for t in totals_nowalk]):
        print(f"  {nm:10}: {m:6.1f}")
    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmaps_2x2.{args.format}"

    if args.style == "interactive":
        from optimize_leaf_raking.viz.animate import make_figure_and_animation
        fig, ani = make_figure_and_animation(state)
    else:
        from optimize_leaf_raking.viz.animate import make_minimal_figure_and_animation
        fig, ani = make_minimal_figure_and_animation(state)

    if args.save:
        # Save animation; prefer ffmpeg, fallback to pillow
        try:
            if args.format == "mp4":
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=args.fps)
                ani.save(out_path.as_posix(), writer=writer)
            else:
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=args.fps)
                ani.save(out_path.as_posix(), writer=writer)
            print(f"Saved animation to {out_path}")
        except Exception as e:
            print(f"Failed to save animation ({e}); try installing ffmpeg or pillow-support.")

    if args.show or not args.save:
        plt.show()


if __name__ == "__main__":
    main()
