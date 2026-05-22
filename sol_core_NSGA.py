import copy
import time

import data_loader
import sol_NSGA


def run_core_nsga_solver(common_data, shared_ref_point=None, shared_ref_front=None):
    """
    Core NSGA-II baseline:
    keep the NSGA-II framework and hybrid initialization in sol_NSGA.py,
    but disable the stronger Hybrid enhancement modules and use a moderate
    standalone budget so it remains a meaningful, but weaker, internal baseline.
    """
    ablation = {
        "name": "Core NSGA-II",
        "use_mols_decoder": False,
        "use_hybrid_init": True,
        "use_low_cost_probes": False,
        "use_low_cost_weight_bias": False,
        "gen_max": 150,
    }
    return sol_NSGA.run_nsga_solver(
        common_data,
        shared_ref_point=shared_ref_point,
        shared_ref_front=shared_ref_front,
        ablation=ablation,
    )


if __name__ == "__main__":
    common_data = data_loader.load_solomon_data(
        "solomon标准算例-时间窗/r1/r101.txt",
        n_customers=60,
        offset=0,
        random_seed=42,
    )
    print(common_data)

    t0 = time.time()
    obj0, obj1, front = run_core_nsga_solver(copy.deepcopy(common_data))
    runtime = time.time() - t0

    print("\nCore NSGA-II completed.")
    print(f"Representative objective 1: {obj0}")
    print(f"Representative objective 2: {obj1}")
    print(f"Front size: {len(front)}")
    print(f"Runtime: {runtime:.2f}s")
