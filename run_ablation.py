import argparse
import copy
import csv
import os
import random
import time

import numpy as np

import data_loader
import sol_NSGA


DEFAULT_INSTANCE = os.path.join("solomon标准算例-时间窗", "r1", "r101.txt")

VARIANTS = [
    ("Full PS-MOEA", {}),
    # ("w/o MOLS decoder", {"use_mols_decoder": False}),
    # ("w/o hybrid initialization", {"use_hybrid_init": False}),
    ("w/o ALNS mutation", {"use_alns_mutation": False}),
    ("w/o normalized crowding distance", {"use_normalized_crowding_distance": False}),
]


def extract_non_dominated(front):
    non_dominated = []
    for i, sol_i in enumerate(front):
        dominated = False
        for j, sol_j in enumerate(front):
            if i == j:
                continue
            if (sol_j[0] <= sol_i[0] and sol_j[1] <= sol_i[1]) and (
                sol_j[0] < sol_i[0] or sol_j[1] < sol_i[1]
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(list(sol_i))
    return non_dominated


def deduplicate(front, ndigits=6):
    unique = []
    seen = set()
    for obj in front:
        key = (round(float(obj[0]), ndigits), round(float(obj[1]), ndigits))
        if key not in seen:
            seen.add(key)
            unique.append(list(obj))
    return unique


def compute_unified_metrics(results):
    combined = []
    for data in results.values():
        data["front_objs"] = deduplicate(data["front_objs"])
        combined.extend(data["front_objs"])

    reference_front = extract_non_dominated(combined)
    all_obj0 = [obj[0] for obj in combined]
    all_obj1 = [obj[1] for obj in combined]
    ideal = [min(all_obj0), min(all_obj1)]
    range_0 = max(all_obj0) - min(all_obj0)
    range_1 = max(all_obj1) - min(all_obj1)
    if range_0 < 1e-10:
        range_0 = 1.0
    if range_1 < 1e-10:
        range_1 = 1.0

    ref_point = [
        max(all_obj0) + 0.1 * range_0,
        max(all_obj1) + 0.1 * range_1,
    ]

    def normalize(front):
        return [
            [(obj[0] - ideal[0]) / range_0, (obj[1] - ideal[1]) / range_1]
            for obj in front
        ]

    norm_ref_point = [
        (ref_point[0] - ideal[0]) / range_0,
        (ref_point[1] - ideal[1]) / range_1,
    ]
    norm_ref_front = normalize(reference_front)

    metrics = {}
    for name, data in results.items():
        front = data["front_objs"]
        norm_front = normalize(front)
        hv, _, sp = sol_NSGA.calculate_metrics(
            norm_front, norm_ref_point, norm_ref_front
        )
        mid = sum(
            (((obj[0] - ideal[0]) / range_0) ** 2 +
             ((obj[1] - ideal[1]) / range_1) ** 2) ** 0.5
            for obj in front
        ) / len(front)
        metrics[name] = {
            "HV": hv,
            "MID": mid,
            "SP": sp,
            "N": len(front),
            "MinCost": min(obj[0] for obj in front),
            "MaxSat": -min(obj[1] for obj in front),
            "Runtime": data["runtime"],
        }

    return metrics, reference_front, ref_point


def run_one_variant(name, config, common_data, seed):
    print("\n" + "#" * 72)
    print(f"# ABLATION: {name} | seed={seed}")
    print("#" * 72)

    random.seed(seed)
    np.random.seed(seed)
    sol_NSGA.SOLUTION_CACHE.clear()

    config = dict(config)
    config["name"] = name

    start = time.time()
    _, _, front = sol_NSGA.run_nsga_solver(
        copy.deepcopy(common_data),
        ablation=config,
    )
    runtime = time.time() - start
    return {"front_objs": [list(obj) for obj in front], "runtime": runtime}


def print_table(metrics):
    print("\n" + "=" * 96)
    print("  ABLATION STUDY RESULTS")
    print("=" * 96)
    header = (
        f"{'Variant':<28} {'HV':>10} {'MID':>10} {'SP':>10} {'N':>5} "
        f"{'MinCost':>12} {'MaxSat':>10} {'Runtime(s)':>12}"
    )
    print(header)
    print("-" * 96)
    for name, m in metrics.items():
        sp = f"{m['SP']:.6f}" if m["SP"] is not None else "N/A"
        print(
            f"{name:<28} {m['HV']:>10.4f} {m['MID']:>10.6f} {sp:>10} "
            f"{m['N']:>5d} {m['MinCost']:>12.2f} {m['MaxSat']:>10.4f} "
            f"{m['Runtime']:>12.2f}"
        )
    print("=" * 96)


def save_csv(metrics, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "HV", "MID", "SP", "N", "MinCost", "MaxSat", "Runtime"])
        for name, m in metrics.items():
            writer.writerow([
                name,
                m["HV"],
                m["MID"],
                "" if m["SP"] is None else m["SP"],
                m["N"],
                m["MinCost"],
                m["MaxSat"],
                m["Runtime"],
            ])


def resolve_instance_path(base_dir, instance):
    if os.path.isabs(instance):
        return instance
    return os.path.join(base_dir, instance)


def run_ablation_experiment(data_file, customers=60, offset=0, seed=42):
    print(f"\nInstance: {data_file}")
    print(f"Customers: {customers} | offset={offset} | seed={seed}")
    common_data = data_loader.load_solomon_data(
        data_file,
        n_customers=customers,
        offset=offset,
        random_seed=seed,
    )

    results = {}
    for name, config in VARIANTS:
        results[name] = run_one_variant(name, config, common_data, seed)

    metrics, reference_front, ref_point = compute_unified_metrics(results)
    print(f"\nReference front size: {len(reference_front)}")
    print(f"Reference point: {ref_point}")
    print_table(metrics)
    return metrics, reference_front, ref_point


def main():
    parser = argparse.ArgumentParser(description="Run PS-MOMF ablation variants.")
    parser.add_argument("--customers", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = resolve_instance_path(base_dir, args.instance)
    metrics, _, _ = run_ablation_experiment(
        data_file,
        customers=args.customers,
        offset=args.offset,
        seed=args.seed,
    )

    out_path = resolve_instance_path(base_dir, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_csv(metrics, out_path)
    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
