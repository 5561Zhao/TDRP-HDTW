import argparse
import csv
import os

from run_ablation import run_ablation_experiment, save_csv


DEFAULT_INSTANCES = [
    os.path.join("solomon标准算例-时间窗", "r1", "r101.txt"),
    os.path.join("solomon标准算例-时间窗", "rc1", "rc101.txt"),
    os.path.join("solomon标准算例-时间窗", "c1", "c101.txt"),
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def instance_label(instance):
    return os.path.splitext(os.path.basename(instance))[0]


def resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def save_summary(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Instance",
            "Seed",
            "Variant",
            "HV",
            "MID",
            "SP",
            "N",
            "MinCost",
            "MaxSat",
            "Runtime",
        ])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments on r101, rc101 and c101 with multiple seeds."
    )
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", default="ablation_batch_results")
    parser.add_argument("--summary", default="ablation_batch_summary.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = resolve_path(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for instance in args.instances:
        data_file = resolve_path(base_dir, instance)
        label = instance_label(data_file)
        for seed in args.seeds:
            print("\n" + "=" * 88)
            print(f"Batch run: instance={label} | seed={seed}")
            print("=" * 88)

            metrics, _, _ = run_ablation_experiment(
                data_file,
                customers=args.customers,
                offset=args.offset,
                seed=seed,
            )

            detail_path = os.path.join(output_dir, f"{label}_seed{seed}.csv")
            save_csv(metrics, detail_path)
            print(f"Saved detail CSV: {detail_path}")

            for variant, m in metrics.items():
                summary_rows.append([
                    label,
                    seed,
                    variant,
                    m["HV"],
                    m["MID"],
                    "" if m["SP"] is None else m["SP"],
                    m["N"],
                    m["MinCost"],
                    m["MaxSat"],
                    m["Runtime"],
                ])

    summary_path = resolve_path(output_dir, args.summary)
    save_summary(summary_rows, summary_path)
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
