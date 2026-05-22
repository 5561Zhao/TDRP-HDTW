# -*- coding: utf-8 -*-
"""
Section 5.4 experiment driver:
Impact of platform time-window designs on robust truck-drone routing.

This module reuses the solver in `sol_NSGA.py` and only changes how STWs are
constructed before solving, so results isolate platform time-window design effects.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sol_NSGA


# 5.4: platform time-window designs (hour scale in paper)
TW_DESIGNS_HOUR: Dict[str, List[Tuple[float, float]]] = {
    "type1_taobao": [(9, 11), (11, 13), (13, 15), (15, 17), (17, 19)],
    "type2_sf": [(9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
    "type3_pdd": [(9, 11), (11, 13), (13, 15), (15, 17), (17, 19), (9, 13), (13, 19)],
    "type4_paper": [(9, 11), (10, 13), (11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19)],
}

TW_DESIGN_LABELS = {
    "type1_taobao": "Type 1 (Taobao)",
    "type2_sf": "Type 2 (SF)",
    "type3_pdd": "Type 3 (PDD)",
    "type4_paper": "Type 4 (This paper)",
}


def hour_to_unit(hour_value: float, day_start_hour: float = 9.0, units_per_hour: float = 120.0) -> float:
    """Map paper hour-scale intervals [9, 19] to model scale [0, 1200]."""
    return (hour_value - day_start_hour) * units_per_hour


def convert_design_to_units(hour_slots: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(hour_to_unit(s), hour_to_unit(e)) for s, e in hour_slots]


TW_DESIGNS_UNIT: Dict[str, List[Tuple[float, float]]] = {
    key: convert_design_to_units(value) for key, value in TW_DESIGNS_HOUR.items()
}

DAY_START = 0.0
DAY_END = 1200.0


def _parse_solomon_rows(file_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start_line = 0
    for i, line in enumerate(lines):
        if "CUST NO." in line:
            data_start_line = i + 1
            break

    if data_start_line == 0:
        raise ValueError("Could not find Solomon header line 'CUST NO.'")

    for line in lines[data_start_line:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        rows.append(
            {
                "id": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "demand": float(parts[3]),
                "ready": float(parts[4]),
                "due": float(parts[5]),
                "service": float(parts[6]),
            }
        )

    if not rows:
        raise ValueError("No customer rows parsed from Solomon file")

    return rows


def _overlap_len(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _build_fixed_width_dtw(
    center: float,
    width: float = 120.0,
    horizon_start: float = DAY_START,
    horizon_end: float = DAY_END,
) -> Tuple[float, float]:
    """
    Build a fixed-width hidden DTW around a center, clipped to planning horizon.
    Width is preserved by shifting when close to boundaries.
    """
    if width <= 0:
        raise ValueError("DTW width must be positive")
    if width > (horizon_end - horizon_start):
        raise ValueError("DTW width cannot exceed horizon length")

    half = width * 0.5
    left = center - half
    right = center + half

    if left < horizon_start:
        right += (horizon_start - left)
        left = horizon_start
    if right > horizon_end:
        left -= (right - horizon_end)
        right = horizon_end

    left = max(horizon_start, left)
    right = min(horizon_end, right)
    return float(left), float(right)


def _select_stw_for_customer(
    desired_start: float,
    desired_end: float,
    stw_candidates: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Customer chooses STW with maximum overlap with hidden desired TW.
    Tie-break: nearest center, then earlier slot.
    """
    if desired_end < desired_start:
        desired_start, desired_end = desired_end, desired_start

    desired_center = (desired_start + desired_end) * 0.5
    best_slot = stw_candidates[0]
    best_overlap = -1.0
    best_center_gap = float("inf")

    for s, e in stw_candidates:
        ov = _overlap_len(desired_start, desired_end, s, e)
        center_gap = abs(((s + e) * 0.5) - desired_center)

        better = False
        if ov > best_overlap + 1e-9:
            better = True
        elif abs(ov - best_overlap) <= 1e-9 and center_gap < best_center_gap - 1e-9:
            better = True
        elif abs(ov - best_overlap) <= 1e-9 and abs(center_gap - best_center_gap) <= 1e-9 and s < best_slot[0]:
            better = True

        if better:
            best_slot = (s, e)
            best_overlap = ov
            best_center_gap = center_gap

    return best_slot


def load_solomon_data_with_tw_design(
    file_path: str,
    tw_design_key: str,
    n_customers: int | None = 10,
    offset: int = 10,
    random_seed: int = 42,
    dtw_fixed_width: float = 120.0,
    customer_type_probs: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Dict:
    """
    Build common_data for sol_NSGA under a specific platform STW design.

    Each customer first gets a random hidden DTW:
    start in [1, 1079], end = start + 120 (or +dtw_fixed_width if overridden).
    Offered STW is selected from the design menu by max-overlap rule.
    For satisfaction evaluation, DTW is clipped by selected STW boundaries.
    """
    if tw_design_key not in TW_DESIGNS_UNIT:
        raise ValueError(f"Unknown tw_design_key: {tw_design_key}")

    rows = _parse_solomon_rows(file_path)
    stw_menu = TW_DESIGNS_UNIT[tw_design_key]

    nodes: List[int] = [0]
    coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, float] = {}
    time_windows: Dict[int, List[float]] = {}
    service_times: Dict[int, float] = {}
    dtw_widths: Dict[int, float] = {}

    depot = rows[0]
    coords[0] = (depot["x"], depot["y"])
    demands[0] = 0.0
    time_windows[0] = [0.0, 1200000.0]
    service_times[0] = 0.0
    dtw_widths[0] = 0.0

    rng_dtw = np.random.RandomState(random_seed)

    added = 0
    for raw_idx in range(1, len(rows)):
        if n_customers is not None and added >= n_customers:
            break
        if raw_idx <= offset:
            continue

        r = rows[raw_idx]
        new_id = added + 1

        nodes.append(new_id)
        coords[new_id] = (r["x"], r["y"])
        demands[new_id] = r["demand"]
        service_times[new_id] = r["service"]

        desired_start = float(rng_dtw.randint(1, 1080))  # [1, 1079]
        desired_end = desired_start + float(dtw_fixed_width)

        stw_start, stw_end = _select_stw_for_customer(desired_start, desired_end, stw_menu)
        time_windows[new_id] = [float(stw_start), float(stw_end)]

        # Satisfaction DTW is STW-clipped DTW, e.g. [130,250] with [0,240] -> [130,240].
        adj_start = max(desired_start, stw_start)
        adj_end = min(desired_end, stw_end)
        if adj_end < adj_start:
            if desired_end <= stw_start:
                adj_start = stw_start
                adj_end = stw_start
            else:
                adj_start = stw_end
                adj_end = stw_end

        dtw_widths[new_id] = float(max(0.0, adj_end - adj_start))

        added += 1

    probs = np.array(customer_type_probs, dtype=float)
    probs = probs / probs.sum()

    rng = np.random.RandomState(random_seed)
    choices = ["D", "P", "SPD"]
    types_str: Dict[int, str] = {}
    types_int: Dict[int, int] = {}

    for nid in nodes:
        if nid == 0:
            continue
        t_str = str(rng.choice(choices, p=probs))
        types_str[nid] = t_str
        if t_str == "P":
            types_int[nid] = 1
        elif t_str == "D":
            types_int[nid] = 2
        else:
            types_int[nid] = 3

    return {
        "nodes": nodes,
        "coords": coords,
        "demands": demands,
        "time_windows": time_windows,
        "service_times": service_times,
        "dtw_widths": dtw_widths,
        "types_str": types_str,
        "types_int": types_int,
        "depot_tw": time_windows[0],
    }

def _non_dominated(points: Sequence[Sequence[float]]) -> List[List[float]]:
    front: List[List[float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            front.append([float(p[0]), float(p[1])])
    return front


def _resolve_instance_path(path_hint: str) -> str:
    if os.path.exists(path_hint):
        return path_hint

    hint_name = os.path.basename(path_hint)
    candidates = list(Path(".").rglob(hint_name))
    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(f"Instance file not found: {path_hint}")


def _monkey_patch_nsga_dtw_from_common() -> Callable:
    """
    Inject solver-side DTW widths from common_data['dtw_widths'] without editing
    sol_NSGA.py. Returns original __init__ for later restoration.
    """
    original_init = sol_NSGA.ProblemInstance.__init__

    def _patched_init(self, common_data, num_customers):
        original_init(self, common_data, num_customers)

        dtw_widths = common_data.get("dtw_widths")
        if dtw_widths is None:
            return

        if isinstance(self.dtw_width, dict):
            for nid in self.nodes:
                self.dtw_width[nid] = float(dtw_widths.get(nid, self.dtw_width.get(nid, 0.0)))
            return

        arr = np.asarray(self.dtw_width, dtype=np.float64)
        for nid in self.nodes:
            if isinstance(dtw_widths, dict):
                arr[nid] = float(dtw_widths.get(nid, arr[nid]))
            else:
                arr[nid] = float(dtw_widths[nid])
        self.dtw_width = arr

    sol_NSGA.ProblemInstance.__init__ = _patched_init
    return original_init

def run_section54_twtype_study(
    instance_file: str,
    n_customers: int = 10,
    offset: int = 10,
    random_seed: int = 42,
    dtw_fixed_width: float = 120.0,
    quiet_solver: bool = False,
    hide_solver_plots: bool = True,
) -> Dict[str, Dict]:
    instance_file = _resolve_instance_path(instance_file)

    results: Dict[str, Dict] = {}

    original_show = sol_NSGA.plt.show
    original_init = _monkey_patch_nsga_dtw_from_common()
    if hide_solver_plots:
        sol_NSGA.plt.show = lambda *args, **kwargs: None

    try:
        for design_key in TW_DESIGNS_HOUR:
            label = TW_DESIGN_LABELS[design_key]
            print("\n" + "#" * 70)
            print(f"# Running TW Design: {label}")
            print("#" * 70)

            common_data = load_solomon_data_with_tw_design(
                file_path=instance_file,
                tw_design_key=design_key,
                n_customers=n_customers,
                offset=offset,
                random_seed=random_seed,
                dtw_fixed_width=dtw_fixed_width,
            )

            t0 = time.time()
            if quiet_solver:
                with redirect_stdout(io.StringIO()):
                    best_dist, min_cost_obj2, front_objs = sol_NSGA.run_nsga_solver(common_data)
            else:
                best_dist, min_cost_obj2, front_objs = sol_NSGA.run_nsga_solver(common_data)
            runtime = time.time() - t0

            front_objs = [list(map(float, p)) for p in front_objs]
            if front_objs:
                # objective[1] = -satisfaction (minimization); max sat => min objective[1]
                max_sat = float(-min(p[1] for p in front_objs))
            else:
                max_sat = float(-min_cost_obj2)

            results[design_key] = {
                "label": label,
                "best_dist": float(best_dist),
                "best_sat": max_sat,
                "front_objs": front_objs,
                "runtime": runtime,
            }

        # shared metrics reference for fair cross-design comparison
        all_points = [p for r in results.values() for p in r["front_objs"]]
        ref_front = _non_dominated(all_points)
        ref_point = [max(p[0] for p in all_points), max(p[1] for p in all_points)]

        for key, r in results.items():
            hv, igd, sp = sol_NSGA.calculate_metrics(r["front_objs"], ref_point, ref_front)
            r["hv"] = hv
            r["igd"] = igd
            r["sp"] = sp

        return results

    finally:
        sol_NSGA.plt.show = original_show
        sol_NSGA.ProblemInstance.__init__ = original_init


def print_summary(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 110)
    print("Section 5.4 Summary: Impact of Platform Time-Window Designs")
    print("=" * 110)
    print(f"{'Design':<24} {'Best Dist':>12} {'Best Sat':>10} {'HV':>12} {'IGD':>12} {'SP':>12} {'Runtime(s)':>12}")
    print("-" * 110)

    for key in TW_DESIGNS_HOUR:
        r = results[key]
        igd_str = f"{r['igd']:.6f}" if r['igd'] is not None else "N/A"
        print(
            f"{r['label']:<24} {r['best_dist']:>12.2f} {r['best_sat']:>10.2f} "
            f"{r['hv']:>12.4f} {igd_str:>12} {r['sp']:>12.6f} {r['runtime']:>12.2f}"
        )

    print("=" * 110)


def plot_results(results: Dict[str, Dict], output_prefix: str = "twtype_54") -> None:
    colors = {
        "type1_taobao": "#1f77b4",
        "type2_sf": "#2ca02c",
        "type3_pdd": "#d62728",
        "type4_paper": "#ff7f0e",
    }
    markers = {
        "type1_taobao": "o",
        "type2_sf": "^",
        "type3_pdd": "s",
        "type4_paper": "D",
    }

    plt.figure(figsize=(12, 8))
    for key in TW_DESIGNS_HOUR:
        front = results[key]["front_objs"]
        dists = [p[0] for p in front]
        sats = [-p[1] for p in front]
        plt.scatter(
            dists,
            sats,
            c=colors[key],
            marker=markers[key],
            s=60,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
            label=results[key]["label"],
        )

    plt.xlabel("Total Distance (Minimize)")
    plt.ylabel("Worst-case Satisfaction (Maximize)")
    plt.title("Section 5.4: Pareto Front Comparison under Platform TW Designs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pareto.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [results[k]["label"] for k in TW_DESIGNS_HOUR]
    keys = list(TW_DESIGNS_HOUR.keys())

    hv_vals = [results[k]["hv"] for k in keys]
    sp_vals = [results[k]["sp"] for k in keys]
    rt_vals = [results[k]["runtime"] for k in keys]
    bar_colors = [colors[k] for k in keys]

    axes[0].bar(labels, hv_vals, color=bar_colors, edgecolor="k")
    axes[0].set_title("Hypervolume (HV)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, sp_vals, color=bar_colors, edgecolor="k")
    axes[1].set_title("Spacing (SP)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, rt_vals, color=bar_colors, edgecolor="k")
    axes[2].set_title("Runtime (s)")
    axes[2].tick_params(axis="x", rotation=20)

    plt.suptitle("Section 5.4 Metric Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_metrics.png", dpi=150)
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Section 5.4 TW-design study based on sol_NSGA")
    parser.add_argument(
        "--instance",
        type=str,
        default=os.path.join("solomon标准算例-时间窗", "c1", "c101.txt"),
        help="Solomon instance path",
    )
    parser.add_argument("--n-customers", type=int, default=40, help="Number of customers to load")
    parser.add_argument("--offset", type=int, default=10, help="Skip first offset customers in Solomon file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for customer-type generation")
    parser.add_argument("--dtw-fixed-width", type=float, default=120.0, help="Raw DTW width before STW clipping")
    parser.add_argument("--quiet-solver", action="store_true", help="Suppress verbose prints from inner solver")
    parser.add_argument("--show-solver-plots", action="store_true", help="Keep plots produced by each inner solver run")
    parser.add_argument("--no-summary-plots", action="store_true", help="Do not draw 5.4 summary plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    results = run_section54_twtype_study(
        instance_file=args.instance,
        n_customers=args.n_customers,
        offset=args.offset,
        random_seed=args.seed,
        dtw_fixed_width=args.dtw_fixed_width,
        quiet_solver=args.quiet_solver,
        hide_solver_plots=not args.show_solver_plots,
    )

    print_summary(results)

    if not args.no_summary_plots:
        plot_results(results, output_prefix="twtype_54")


if __name__ == "__main__":
    main()
























