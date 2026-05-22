import time
import pandas as pd
import data_loader
import gurobi_jq  # 对应之前的 gurobi.py
import gurobi_dis  # 对应之前的 gurobi.py
import gurobi_sat  # 对应之前的 gurobi.py
import sol_NSGA# 对应之前的 Improved NSGA-II.py


def main():
    # ==========================
    # 1. 实验配置
    # ==========================
    DATA_FILE = 'solomon标准算例-时间窗/c1/c101.txt'  # 确保文件在同级目录
    N_CUSTOMERS = 15 # 客户数量 (建议先用 8-10 测试 Gurobi，太大跑不动)
    RANDOM_SEED = 42  # 固定种子，确保类型生成一致

    print(f"{'=' * 60}")
    print(f"FSTSP-MT 对比实验 | N={N_CUSTOMERS} | Seed={RANDOM_SEED}")
    print(f"{'=' * 60}\n")

    # ==========================
    # 2. 统一数据加载
    # ==========================
    print(f"[Main] Loading Data from {DATA_FILE}...")
    try:
        common_data = data_loader.load_solomon_data(
            file_path=DATA_FILE,
            n_customers=N_CUSTOMERS,
            random_seed=RANDOM_SEED
        )
        print(f"[Main] Data Loaded. Nodes: {common_data['nodes']}")
        print(common_data)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {DATA_FILE}，请确保已下载 Solomon 数据集。")
        return

    print(f"\n{'-' * 20} Running Gurobi_dis (Epsilon-Constraint) {'-' * 20}")
    t_start_g = time.time()
    # [修改] 初始化全局参考点变量
    global_ref_point = None

    try:
        # [修改] 接收两个返回值：解集 和 参考点
        gurobi_solutions, global_ref_point = gurobi_dis.run_gurobi_solver(common_data)
        status_g = "Solved" if gurobi_solutions else "Infeasible"
    except Exception as e:
        print(f"Gurobi_dis Error: {e}")
        gurobi_solutions = None
        status_g = "Error"

    t_end_g = time.time()
    real_time_g = t_end_g - t_start_g

    if gurobi_solutions:
        # 打印找到的帕累托解的数量
        print(f"Gurobi_dis Result: Found {len(gurobi_solutions)} Pareto solutions.")
        print(f"Runtime: {real_time_g:.2f}s")
    else:
        print("Gurobi_dis Failed to find solutions.")



    # ==========================
    # 4. 运行 NSGA-II (Heuristic)
    # ==========================
    print(f"\n{'-' * 20} Running NSGA-II (Meta-heuristic) {'-' * 20}")
    t_start_n = time.time()

    try:
        # nsga 返回 (distance, -satisfaction)
        # [修改] 将 Gurobi 计算出的 global_ref_point 传入 NSGA
        print(common_data)
        nsga_dist, nsga_neg_sat = sol_NSGA.run_nsga_solver(common_data, shared_ref_point=global_ref_point)
        nsga_sat = -nsga_neg_sat
    except Exception as e:
        print(f"NSGA Error: {e}")
        nsga_dist, nsga_sat = None, None

    # t_end_n = time.time()
    # real_time_n = t_end_n - t_start_n
    #
    # print(f"NSGA Result: Dist={nsga_dist:.2f}, Sat={nsga_sat:.2f}, Time={real_time_n:.2f}s")

    # ==========================
    # # 5. 结果对比分析
    # # ==========================
    # print(f"\n\n{'=' * 60}")
    # print("COMPARISON RESULT: Exact Pareto Front vs NSGA-II Solution")
    # print(f"{'=' * 60}")
    #
    # # 1. 时间对比
    # print(f"{'Metric':<20} | {'Gurobi (Exact)':<20} | {'NSGA-II (Approx)':<20}")
    # print(f"{'-' * 66}")
    # print(f"{'Time (s)':<20} | {real_time_g:<20.4f} | {real_time_n:<20.4f}")
    #
    # # 2. 解的质量对比
    # print(f"{'-' * 66}")
    # print(f"Gurobi Pareto Solutions (Sat, Cost):")
    # if gurobi_solutions:
    #     # 遍历打印 Gurobi 的所有解
    #     for i, (sat, cost) in enumerate(gurobi_solutions):
    #         print(f"  Sol {i + 1}: Sat={sat:.4f}, Cost={cost:.2f}")
    # else:
    #     print("  No solutions found.")
    #
    # print(f"{'-' * 66}")
    # print(f"NSGA-II Solution:")
    # if nsga_dist is not None:
    #     print(f"  Result: Sat={nsga_sat:.4f}, Cost={nsga_dist:.2f}")
    #
    #     # 简单的支配性检查 (Domination Check)
    #     # 检查 NSGA 的解是否被 Gurobi 的某个解严格支配（即 Gurobi 解成本更低且满意度更高）
    #     is_dominated = False
    #     if gurobi_solutions:
    #         for g_sat, g_cost in gurobi_solutions:
    #             # 注意：我们希望 Cost 越小越好，Sat 越大越好
    #             if g_cost <= nsga_dist and g_sat >= nsga_sat:
    #                 if g_cost < nsga_dist or g_sat > nsga_sat:
    #                     is_dominated = True
    #                     break
    #
    #     if is_dominated:
    #         print(f"  >> Analysis: This NSGA solution is DOMINATED by the Exact Pareto Front.")
    #     else:
    #         print(f"  >> Analysis: This NSGA solution is Non-Dominated (Good quality).")
    #
    # else:
    #     print("  No solution found.")
    #
    # print(f"{'=' * 60}")
# 3.2 运行 Gurobi_sat (Exact Solver)
    # ==========================
    print(f"\n{'-' * 20} Running Gurobi_sat (MIP) {'-' * 20}")
    t_start_g = time.time()
    global_ref_point = None
    try:
        gurobi_solutions_sat, global_ref_point = gurobi_sat.run_gurobi_solver(common_data)
        status_g_sat  = "Solved" if gurobi_solutions_sat  else "Infeasible"
    except Exception as e:
        print(f"Gurobi_sat Error: {e}")
        gurobi_solutions_sat = None
        status_g_sat  = "Error"

    t_end_g = time.time()
    real_time_g = t_end_g - t_start_g

    if gurobi_solutions_sat:
        # 打印找到的帕累托解的数量
        print(f"Gurobi_sat Result: Found {len(gurobi_solutions_sat)} Pareto solutions.")
        print(f"Runtime: {real_time_g:.2f}s")
    else:
        print("Gurobi_sat Failed to find solutions.")

if __name__ == "__main__":
    main()