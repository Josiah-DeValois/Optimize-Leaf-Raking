from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, Slider, Button

from ..core.config import (
    YardParams,
    VizParams,
    CandidateParams,
    WalkParams,
    FrontSweepParams,
    CalibrationData,
)
from ..core.calibration import calibrate_rake_model, calibrate_bag_model
from ..core.yard import build_yard_and_masses
from ..core.raking import (
    radial_arrival_times_calibrated,
    deposit_piles_disk_from_arrivals,
    deposit_pile_disks_from_masses,
    baseline_rake_time_to_centers,
)
from ..core.bagging import bag_mass_removed, compute_pile_order, walk_times_from_order
from ..core.centers import centers_bf, centers_micro, centers_opt_discrete
from ..core.front_sweep import (
    front_sweep_band_time,
    band_snapshot_with_spillage_columns,
    final_band_and_bagging,
    band_bagging_density,
)
from .plotting import get_cmap, draw_density
from optimize_leaf_raking.solvers.mip import compute_fixed_centers_objective
from pathlib import Path
import csv


@dataclass
class PanelInfo:
    centers: np.ndarray
    t_arrive: np.ndarray
    per_pile: List[Dict[str, np.ndarray]]
    pile_totals: np.ndarray
    order: np.ndarray
    bag_times: np.ndarray
    walk_times: np.ndarray


@dataclass
class VizState:
    yard: YardParams
    viz: VizParams
    cand: CandidateParams
    walk: WalkParams
    front: FrontSweepParams
    calib: CalibrationData
    alpha: float
    beta: float
    b0: float
    b1: float
    # Yard grids
    xs: np.ndarray
    ys: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    nx: int
    ny: int
    Acell: float
    rho_init: np.ndarray
    mass_grid_init: np.ndarray
    cells: np.ndarray
    masses: np.ndarray
    # Panels
    centers_list: List[np.ndarray]
    radial_data: Dict[int, PanelInfo]
    rake_total_secs: List[float]
    bag_total_secs: List[float]
    # Front sweep
    rows_per: int
    T_steps: np.ndarray
    rho_final_band: np.ndarray | None
    mass_final_band: np.ndarray | None
    M_band_total: float
    bag_total_band: float


def build_initial_state(
    yard: YardParams,
    viz: VizParams,
    cand: CandidateParams,
    walk: WalkParams,
    front: FrontSweepParams,
    calib_data: CalibrationData,
) -> VizState:
    # Yard grid and masses
    yd = build_yard_and_masses(yard)

    # Calibrate rake and bag models
    alpha, beta = calibrate_rake_model(calib_data.raking_splits)
    b0, b1 = calibrate_bag_model(calib_data.bagging_times, bag_capacity_lb=35.0)

    # Centers for panels: 0 (BF), 1 (front band placeholder), 2 (micro), 3 (Optimization)
    c0 = centers_bf(yard, yd["masses"], bag_capacity_lb=35.0)
    c1 = np.empty((0, 2))
    c2 = centers_micro(yd["cells"], yd["masses"], bag_capacity_lb=35.0, seed=42)
    # Prefer MILP centers from results/optimal_centers.csv if available; otherwise use discrete search
    c3 = None
    csv_path = Path("results/optimal_centers.csv")
    if csv_path.exists():
        try:
            rows = list(csv.DictReader(csv_path.open()))
            loaded = np.array([[float(r["x"]), float(r["y"]) ] for r in rows], dtype=float)
            if loaded.size:
                c3 = loaded
        except Exception:
            c3 = None
    if c3 is None:
        c3, _ = centers_opt_discrete(
            yd["cells"], yd["masses"], yard, cand.spacing, alpha, beta, cand.K_max,
            bag_b0=b0, bag_b1=b1, bag_capacity_lb=35.0,
        )
    centers_list = [c0, c1, c2, c3]

    # Precompute outside-in arrivals and bagging plan for 0,2,3
    radial_data: Dict[int, PanelInfo] = {}
    rake_total_secs = [0.0] * 4
    bag_total_secs = [0.0] * 4

    for idx in [0, 2, 3]:
        centers = centers_list[idx]
        t_arrive, pile_of_cell, per_pile = radial_arrival_times_calibrated(
            yd["cells"], yd["masses"], centers, alpha, beta, angle_bins=24, yard=yard
        )
        rake_total_secs[idx] = float(np.nanmax(t_arrive[np.isfinite(t_arrive)])) if np.isfinite(t_arrive).any() else 0.0
        pile_totals = np.array([pp["M_total"] for pp in per_pile]) if len(per_pile) else np.array([])
        order = compute_pile_order(centers, walk.order_method)
        walk_times = walk_times_from_order(centers, order, walk.speed_ftps)
        if pile_totals.size:
            n_bags = np.ceil(pile_totals / 35.0).astype(int)
            bag_times = n_bags * b0 + pile_totals * b1
            bag_total_secs[idx] = float(bag_times[order].sum() + walk_times.sum())
        else:
            bag_times = np.array([])
            bag_total_secs[idx] = 0.0
        radial_data[idx] = PanelInfo(
            centers=centers,
            t_arrive=t_arrive,
            per_pile=per_pile,
            pile_totals=pile_totals,
            order=order,
            bag_times=bag_times,
            walk_times=walk_times,
        )

    # Front-sweep: compute band timing and bag totals
    rows_per, T_steps, a_eff, b_eff = front_sweep_band_time(
        strip_ft=front.strip_ft,
        use_eff=front.use_eff,
        eta=front.eta,
        delta_beta=front.delta_beta,
        skip_rescale=front.skip_rescale,
        yard=yard,
        mass_grid_init=yd["mass_grid_init"],
        Y=yd["Y"],
        alpha=alpha,
        beta=beta,
    )
    rho_final_band, mass_final_band, M_band_total, bag_total_band, T_rake = final_band_and_bagging(
        rows_per, T_steps, viz.rho_cap, yd["rho_init"], yd["mass_grid_init"], yd["Acell"],
        bag_capacity_lb=35.0, b0=b0, b1=b1
    )
    rake_total_secs[1] = T_rake
    bag_total_secs[1] = bag_total_band

    # Override Optimization panel totals with exact MILP objective for the chosen centers
    if centers_list[3].size:
        obj_mip_fixed = compute_fixed_centers_objective(
            yd["cells"], yd["masses"], centers_list[3], alpha, beta, b0, b1, 35.0, rel_gap=0.0
        )
        rake_base = baseline_rake_time_to_centers(
            yd["cells"], yd["masses"], centers_list[3], alpha, beta, yard
        )
        bag_total_secs[3] = max(0.0, obj_mip_fixed - rake_base)
        rake_total_secs[3] = rake_base
        # Speed up Optimization panel dynamics to finish at MILP bag time
        info3 = radial_data.get(3)
        if info3 is not None:
            orig = float((info3.bag_times.sum() if info3.bag_times.size else 0.0) + (info3.walk_times.sum() if info3.walk_times.size else 0.0))
            desired = float(bag_total_secs[3])
            if orig > 1e-12 and desired > 0:
                scale = desired / orig
                info3.bag_times = info3.bag_times * scale
                info3.walk_times = info3.walk_times * scale

    return VizState(
        yard=yard,
        viz=viz,
        cand=cand,
        walk=walk,
        front=front,
        calib=calib_data,
        alpha=alpha,
        beta=beta,
        b0=b0,
        b1=b1,
        xs=yd["xs"],
        ys=yd["ys"],
        X=yd["X"],
        Y=yd["Y"],
        nx=yd["nx"],
        ny=yd["ny"],
        Acell=yd["Acell"],
        rho_init=yd["rho_init"],
        mass_grid_init=yd["mass_grid_init"],
        cells=yd["cells"],
        masses=yd["masses"],
        centers_list=centers_list,
        radial_data=radial_data,
        rake_total_secs=rake_total_secs,
        bag_total_secs=bag_total_secs,
        rows_per=rows_per,
        T_steps=T_steps,
        rho_final_band=rho_final_band,
        mass_final_band=mass_final_band,
        M_band_total=M_band_total,
        bag_total_band=bag_total_band,
    )


def panel_totals_seconds(state: VizState) -> List[float]:
    return [
        state.rake_total_secs[i] + (state.bag_total_secs[i] if i != 1 else state.bag_total_band)
        for i in range(4)
    ]


def panel_totals_seconds_no_walk(state: VizState) -> List[float]:
    """Totals excluding walking time between piles for outside-in panels.

    - Panels 0 (BF), 2 (Micro), 3 (Optimization): rake_total + sum(bag_times)
    - Panel 1 (Front): unchanged; no walking term is modeled there
    """
    totals = [0.0, 0.0, 0.0, 0.0]
    # Outside-in panels
    for i in (0, 2, 3):
        if i == 3:
            # Use overridden bag_total_secs for Optimization (matches MILP objective)
            bag_only = state.bag_total_secs[i]
        else:
            info = state.radial_data[i]
            bag_only = float(info.bag_times.sum()) if info.bag_times.size else 0.0
        totals[i] = state.rake_total_secs[i] + bag_only
    # Front-sweep panel (same as with-walking since no walking is modeled)
    totals[1] = state.rake_total_secs[1] + state.bag_total_band
    return totals


def density_snapshot_outside_in(state: VizState, panel_idx: int, t_sec: float) -> np.ndarray:
    info = state.radial_data[panel_idx]
    centers = info.centers
    T_rake = state.rake_total_secs[panel_idx]
    if t_sec <= T_rake or centers.size == 0:
        remain = state.rho_init.copy().ravel()
        arrived = (info.t_arrive <= t_sec)
        remain[arrived] = 0.0
        remain = remain.reshape(state.ny, state.nx)
        acc = deposit_piles_disk_from_arrivals(
            centers, info.per_pile, t_sec, state.xs, state.ys, state.viz.rho_cap, state.Acell
        )
        return remain + acc
    # Bagging (with walking)
    tau = t_sec - T_rake
    totals = info.pile_totals
    Ms_rem = totals.copy()
    order = info.order
    bag_times = info.bag_times
    walk_times = info.walk_times
    t_left = tau
    for j, p in enumerate(order):
        wj = walk_times[j] if j < len(walk_times) else 0.0
        if t_left <= wj + 1e-12:
            break  # still walking
        t_left -= wj
        Mj = Ms_rem[p]
        Tj = bag_times[p]
        if t_left >= Tj - 1e-12:
            Ms_rem[p] = 0.0
            t_left -= Tj
        else:
            removed = bag_mass_removed(Mj, t_left, 35.0, state.b0, state.b1)
            Ms_rem[p] = max(0.0, Mj - removed)
            t_left = 0.0
            break
    acc = deposit_pile_disks_from_masses(
        centers, Ms_rem, state.xs, state.ys, state.viz.rho_cap, state.Acell
    )
    return acc


def density_snapshot_front_sweep(state: VizState, t_sec: float) -> np.ndarray:
    T_rake = state.rake_total_secs[1]
    if t_sec <= T_rake:
        return band_snapshot_with_spillage_columns(
            t_sec, state.rows_per, state.T_steps, state.viz.rho_cap,
            state.rho_init, state.mass_grid_init, state.Acell
        )
    # Bagging: front-first
    tau = t_sec - T_rake
    if state.mass_final_band is None:
        return band_snapshot_with_spillage_columns(
            T_rake, state.rows_per, state.T_steps, state.viz.rho_cap,
            state.rho_init, state.mass_grid_init, state.Acell
        )
    removed = bag_mass_removed(state.M_band_total, tau, 35.0, state.b0, state.b1)
    removed = min(removed, state.M_band_total)
    return band_bagging_density(state.mass_final_band, state.Y, state.Acell, removed)


def make_figure_and_animation(state: VizState, show_centers: bool = True, save_path: str | None = None):
    cmap = get_cmap(state.viz.colors)
    extent = (0, state.yard.L, 0, state.yard.W)
    vmin, vmax = 0.0, state.viz.rho_cap

    # Build figure grid: 2x2 panels + control panel
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], wspace=0.25, hspace=0.18)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    axes = [ax00, ax01, ax10, ax11]
    ax_ctrl = fig.add_subplot(gs[:, 2])
    ax_ctrl.axis("off")

    titles = [
        "BF-centers (outside-in) — walking between piles",
        "Front-sweep (active strip) — column-aware spillage",
        "Micro-piles (outside-in) — walking between piles",
        "Optimization (discrete K≤5) — walking between piles",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, state.yard.L)
        ax.set_ylim(0, state.yard.W)
        ax.set_aspect("equal")
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")

    # Initial images
    imgs = []
    rho0 = state.rho_init
    imgs.append(draw_density(ax00, rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(ax01, rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(ax10, rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(ax11, rho0, extent, vmin, vmax, cmap))

    # Centers overlays
    for i in [0, 2, 3]:
        C = state.centers_list[i]
        if C.size:
            axes[i].scatter(C[:, 0], C[:, 1], s=28, c="#2b8cbe", marker="x", linewidths=1.4)

    # Build front-sweep controls
    # CheckButtons
    ctrl_y = 0.85
    cb_ax = fig.add_axes([0.73, 0.77, 0.24, 0.14])
    cb = CheckButtons(cb_ax, ["Front: use efficiency", "Front: skip rescale"],
                      [state.front.use_eff, state.front.skip_rescale])

    # Sliders
    s_eta_ax = fig.add_axes([0.73, 0.68, 0.24, 0.03])
    s_eta = Slider(s_eta_ax, "η (eff)", 1.0, 3.0, valinit=state.front.eta, valstep=0.05)
    s_db_ax = fig.add_axes([0.73, 0.62, 0.24, 0.03])
    s_db = Slider(s_db_ax, "Δβ", 0.0, 0.5, valinit=state.front.delta_beta, valstep=0.01)
    s_strip_ax = fig.add_axes([0.73, 0.56, 0.24, 0.03])
    s_strip = Slider(s_strip_ax, "Strip (ft)", 1.0, 6.0, valinit=state.front.strip_ft, valstep=0.5)

    # Apply button
    apply_ax = fig.add_axes([0.73, 0.50, 0.24, 0.04])
    apply_btn = Button(apply_ax, "Apply")

    # Derived timing
    def totals_minutes():
        totals = panel_totals_seconds(state)
        return [t / 60.0 for t in totals]

    text_box = fig.add_axes([0.73, 0.30, 0.24, 0.18])
    text_box.axis("off")
    text_obj = text_box.text(0.0, 1.0, "", va="top", family="monospace", fontsize=9)

    def refresh_stats():
        mins = totals_minutes()
        lines = ["Totals (min):"]
        names = ["BF", "Front", "Micro", "Opt (disc)"]
        for nm, t in zip(names, mins):
            lines.append(f"{nm:12}: {t:6.1f}")
        text_obj.set_text("\n".join(lines))

    refresh_stats()

    # Animation frames
    BASE_TOTAL = max(panel_totals_seconds(state))
    total_minutes = int(np.ceil(BASE_TOTAL / 60.0))
    n_frames = max(1, total_minutes + 1)

    def frame_to_time(i):
        return float(i * state.viz.seconds_per_frame)

    def update(i):
        t_sec = frame_to_time(i)
        imgs[0].set_data(density_snapshot_outside_in(state, 0, t_sec))
        imgs[1].set_data(density_snapshot_front_sweep(state, t_sec))
        imgs[2].set_data(density_snapshot_outside_in(state, 2, t_sec))
        imgs[3].set_data(density_snapshot_outside_in(state, 3, t_sec))
        return imgs

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 * (1.0 / max(1, state.viz.fps)),
        blit=False,
        repeat=False,
    )

    # Apply control updates
    def on_apply(event=None):
        # Update front sweep params
        vals = cb.get_status()
        state.front.use_eff = bool(vals[0])
        state.front.skip_rescale = bool(vals[1])
        state.front.eta = float(s_eta.val)
        state.front.delta_beta = float(s_db.val)
        state.front.strip_ft = float(s_strip.val)
        # Recompute band timing and bag totals
        state.rows_per, state.T_steps, a_eff, b_eff = front_sweep_band_time(
            strip_ft=state.front.strip_ft,
            use_eff=state.front.use_eff,
            eta=state.front.eta,
            delta_beta=state.front.delta_beta,
            skip_rescale=state.front.skip_rescale,
            yard=state.yard,
            mass_grid_init=state.mass_grid_init,
            Y=state.Y,
            alpha=state.alpha,
            beta=state.beta,
        )
        (
            state.rho_final_band,
            state.mass_final_band,
            state.M_band_total,
            state.bag_total_band,
            T_rake,
        ) = final_band_and_bagging(
            state.rows_per,
            state.T_steps,
            state.viz.rho_cap,
            state.rho_init,
            state.mass_grid_init,
            state.Acell,
            bag_capacity_lb=35.0,
            b0=state.b0,
            b1=state.b1,
        )
        state.rake_total_secs[1] = T_rake
        # Refresh stats text
        refresh_stats()

    apply_btn.on_clicked(on_apply)

    return fig, ani


def make_minimal_figure_and_animation(state: VizState):
    """Minimal 2x2 animation matching the provided example GIF.

    - 2x2 panels with titles only
    - Global colorbar on the right (0..rho_cap, lb/ft^2)
    - Suptitle and top-right minute counter
    """
    cmap = get_cmap(state.viz.colors)
    extent = (0, state.yard.L, 0, state.yard.W)
    vmin, vmax = 0.0, state.viz.rho_cap

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5))
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    titles = ["BF-centers", "Front sweep", "Micro-piles", "Optimization"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, state.yard.L)
        ax.set_ylim(0, state.yard.W)
        ax.set_aspect("equal")
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")

    # Initial images
    imgs = []
    rho0 = state.rho_init
    imgs.append(draw_density(axes[0], rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(axes[1], rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(axes[2], rho0, extent, vmin, vmax, cmap))
    imgs.append(draw_density(axes[3], rho0, extent, vmin, vmax, cmap))

    # Centers overlays
    for i in [0, 2, 3]:
        C = state.centers_list[i]
        if C.size:
            axes[i].scatter(C[:, 0], C[:, 1], s=28, c="#2b8cbe", marker="x", linewidths=1.4)

    # Tree trunk marker (white square) at yard.tree
    tx, ty = state.yard.tree
    for ax in axes:
        ax.scatter([tx], [ty], s=70, c="white", marker="s", edgecolors="none")

    # Global colorbar on the right
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045, pad=0.04)
    cbar.set_label("lb/ft^2")

    # Titles
    fig.suptitle("Comparing leaf raking strategies", fontsize=18)

    # Minute counter text (top-right)
    BASE_TOTAL = max(panel_totals_seconds(state))
    total_minutes = int(np.ceil(BASE_TOTAL / 60.0))
    n_frames = max(1, total_minutes + 1)
    minute_text = fig.text(0.89, 0.96, f"Minute 0 / {total_minutes}")

    def frame_to_time(i):
        return float(i * state.viz.seconds_per_frame)

    def update(i):
        t_sec = frame_to_time(i)
        imgs[0].set_data(density_snapshot_outside_in(state, 0, t_sec))
        imgs[1].set_data(density_snapshot_front_sweep(state, t_sec))
        imgs[2].set_data(density_snapshot_outside_in(state, 2, t_sec))
        imgs[3].set_data(density_snapshot_outside_in(state, 3, t_sec))
        minute_text.set_text(f"Minute {i} / {total_minutes}")
        return imgs

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 * (1.0 / max(1, state.viz.fps)),
        blit=False,
        repeat=False,
    )

    return fig, ani
