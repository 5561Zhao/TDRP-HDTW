import data_loader
import time
import sol_NSGA
# import sol_core_NSGA
import sol_pure_NSGA
import sol_MOPSO

"""
多算法统一对比调度器
===================
功能:
  1. 加载相同的测试数据
  2. 依次运行多个求解器, 收集各自的 Pareto 前沿目标值
  3. 合并所有前沿, 提取非支配集作为 Combined Reference Front
  4. 确定共享参考点 (Shared Reference Point)
  5. 用统一基准重新计算 HV / IGD / Spacing / Runtime
  6. 打印对比表格

说明:
  - 每个求解器内部会先打印自身的局部指标 (此时 IGD 为 N/A, 因无参考前沿)
  - 本文件在所有求解器运行完毕后, 基于统一参考前沿重新计算全部指标, 实现公平对比
"""


def extract_non_dominated(combined_front):
    """
    从合并的目标值集合中提取非支配解集 (Combined Reference Front)
    combined_front: list of [obj1, obj2], 均为最小化方向
    返回: 非支配子集 (list of [obj1, obj2])

    算法: 逐一检查每个解是否被集合中任何其他解支配
    时间复杂度: O(n^2), 对本场景的前沿规模 (数百量级) 完全可接受
    """
    non_dominated = []
    n = len(combined_front)

    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # 检查 j 是否支配 i (两目标均为最小化)
            if (combined_front[j][0] <= combined_front[i][0] and
                combined_front[j][1] <= combined_front[i][1]) and \
               (combined_front[j][0] < combined_front[i][0] or
                combined_front[j][1] < combined_front[i][1]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(list(combined_front[i]))

    return non_dominated


def run_comparison(common_data):
    """
    统一调度多个求解器并进行公平对比
    """

    # ==========================================
    # 阶段1: 依次运行多个求解器, 收集 Pareto 前沿
    # ==========================================
    # 注意: 此阶段各求解器内部会打印自身局部指标 (IGD = N/A)
    # 统一对比在阶段3中完成

    results = {}

    # --- 1.1 混合模因 NSGA-II ---
    print("\n" + "#" * 60)
    print("# SOLVER 1: Hybrid Memetic NSGA-II (sol_NSGA.py)")
    print("#" * 60)
    t_start = time.time()
    obj0_1, obj1_1, front_objs_1 = sol_NSGA.run_nsga_solver(common_data)
    t_end = time.time()
    results['Hybrid NSGA-II'] = {
        'front_objs': [list(obj) for obj in front_objs_1],
        'runtime': t_end - t_start
    }

    # --- 1.2 纯 NSGA-II ---
    print("\n" + "#" * 60)
    print("# SOLVER 2: Pure NSGA-II (sol_pure_NSGA.py)")
    print("#" * 60)
    t_start = time.time()
    obj0_2, obj1_2, front_objs_2 = sol_pure_NSGA.run_pure_nsga_solver(common_data)
    t_end = time.time()
    results['Pure NSGA-II'] = {
        'front_objs': [list(obj) for obj in front_objs_2],
        'runtime': t_end - t_start
    }

    # --- 1.3 MOPSO ---
    print("\n" + "#" * 60)
    print("# SOLVER 3: MOPSO (sol_MOPSO.py)")
    print("#" * 60)
    t_start = time.time()
    obj0_3, obj1_3, front_objs_3 = sol_MOPSO.run_mopso_solver(common_data)
    t_end = time.time()
    results['MOPSO'] = {
        'front_objs': [list(obj) for obj in front_objs_3],
        'runtime': t_end - t_start
    }

    # ==========================================
    # 阶段2: 构建统一基准
    # ==========================================
    print("\n" + "=" * 60)
    print("  PHASE 2: Building Unified Benchmark")
    print("=" * 60)

    # 2.1 合并所有前沿
    # 2.1 各算法前沿去重 (消除缓存/种群趋同导致的重复目标值)
    for name, data in results.items():
        original_size = len(data['front_objs'])
        seen = set()
        unique_objs = []
        for obj in data['front_objs']:
            key = (obj[0], obj[1])
            if key not in seen:
                seen.add(key)
                unique_objs.append(obj)
        data['front_objs'] = unique_objs
        print(f"  {name}: {original_size} -> {len(unique_objs)} (deduplicated)")

    # 2.2 合并所有前沿
    combined_all = []
    for name, data in results.items():
        combined_all.extend(data['front_objs'])

    print(f"Total solutions collected: {len(combined_all)}")

    # 2.2 提取非支配集作为 Combined Reference Front
    shared_ref_front = extract_non_dominated(combined_all)
    print(f"Combined Reference Front size: {len(shared_ref_front)}")

    # 2.3 确定共享参考点 (所有前沿中各目标的最大值)
    all_obj0 = [s[0] for s in combined_all]
    all_obj1 = [s[1] for s in combined_all]
    range_0 = max(all_obj0) - min(all_obj0)
    range_1 = max(all_obj1) - min(all_obj1)
    if range_0 < 1e-10: range_0 = 1.0
    if range_1 < 1e-10: range_1 = 1.0
    shared_ref_point = [max(all_obj0) + 0.1* range_0, max(all_obj1) +0.1* range_1]
    print(f"Shared Reference Point: {shared_ref_point}")

    # 理想点: 各目标在合并前沿中的最优值 (两目标均为最小化)
    # 用于 MID (Mean Ideal Distance) 计算
    ideal_point = [min(all_obj0), min(all_obj1)]
    print(f"Ideal Point: {ideal_point}")

    # ==========================================
    # 阶段3: 统一指标计算 (使用任一求解器中的 calculate_metrics)
    # ==========================================
    print("\n" + "=" * 60)
    print("  PHASE 3: Unified Metrics Comparison")
    print("=" * 60)

    # 使用 sol_NSGA 中的 calculate_metrics (三个文件中该函数实现完全一致)
    calc_fn = sol_NSGA.calculate_metrics

    # 归一化函数: 将目标值映射到 [0, 1] 区间, 消除量纲差异
    # 使用 Phase 2 已有的 all_obj0/all_obj1 的极差 (range_0, range_1)
    # 和最小值 (ideal_point) 作为归一化基准, 与 MID 的归一化方式一致
    def normalize_front(front):
        return [[(obj[0] - ideal_point[0]) / range_0,
                 (obj[1] - ideal_point[1]) / range_1] for obj in front]

    norm_ref_point = [(shared_ref_point[0] - ideal_point[0]) / range_0,
                      (shared_ref_point[1] - ideal_point[1]) / range_1]
    norm_ref_front = normalize_front(shared_ref_front)

    metrics_table = {}
    for name, data in results.items():
        norm_front = normalize_front(data['front_objs'])

        # HV 和 SP 基于归一化后的数据计算, 消除目标间量纲差异
        hv_val, _, sp_val = calc_fn(
            norm_front,
            norm_ref_point,
            norm_ref_front
        )
        front = data['front_objs']
        min_cost = min(obj[0] for obj in front)
        max_sat = -min(obj[1] for obj in front)

        # MID (Mean Ideal Distance): 前沿中每个解到理想点的归一化欧氏距离的均值
        # 定义: MID = (1/|A|) * Σ sqrt(((f1-f1*)/range_0)^2 + ((f2-f2*)/range_1)^2)
        # 越小表示前沿整体越接近理想点, 衡量收敛性
        # 归一化方式与 normalize_front 一致
        mid_val = sum(
            (((obj[0] - ideal_point[0]) / range_0) ** 2 +
             ((obj[1] - ideal_point[1]) / range_1) ** 2) ** 0.5
            for obj in front
        ) / len(front)

        metrics_table[name] = {
            'HV': hv_val,
            'MID': mid_val,
            'SP': sp_val,
            'Runtime': data['runtime'],
            'MinCost': min_cost,
            'MaxSat': max_sat,
            'FrontSize': len(front)
        }
    # ==========================================
    # 阶段4: 打印对比表格
    # ==========================================
    print("\n" + "=" * 70)
    print("  UNIFIED PERFORMANCE COMPARISON (Shared Reference)")
    print("=" * 70)
    print(f"  Reference Point: {shared_ref_point}")
    print(f"  Reference Front Size: {len(shared_ref_front)}")
    print("-" * 70)

    # 表头
    header = (f"{'Algorithm':<22} {'HV':>12} {'MID':>12} {'SP':>12} "
              f"{'N':>6} {'MinCost':>12} {'MaxSat':>12} {'Runtime(s)':>12}")
    print(header)
    print("-" * 94)

    # 数据行
    display_order = ['Hybrid NSGA-II', 'Pure NSGA-II', 'MOPSO']
    for name in display_order:
        m = metrics_table[name]
        sp_str = f"{m['SP']:.6f}" if m['SP'] is not None else "N/A"
        print(f"{name:<22} {m['HV']:>12.4f} {m['MID']:>12.6f} {sp_str:>12} "
              f"{m['FrontSize']:>6d} {m['MinCost']:>12.2f} {m['MaxSat']:>12.4f} "
              f"{m['Runtime']:>12.2f}")

    print("-" * 94)

    # 标注最优
    # HV: 越大越好
    best_hv_name = max(metrics_table, key=lambda k: metrics_table[k]['HV'])
    # MID: 越小越好
    best_mid_name = min(metrics_table, key=lambda k: metrics_table[k]['MID'])
    # SP: 越小越好
    valid_sp_items = {k: v for k, v in metrics_table.items() if v['SP'] is not None}
    best_sp_name = min(valid_sp_items, key=lambda k: valid_sp_items[k]['SP']) if valid_sp_items else "N/A"
    # MinCost: 越小越好
    best_cost_name = min(metrics_table, key=lambda k: metrics_table[k]['MinCost'])
    # MaxSat: 越大越好
    best_sat_name = max(metrics_table, key=lambda k: metrics_table[k]['MaxSat'])
    # Runtime: 越小越好
    best_rt_name = min(metrics_table, key=lambda k: metrics_table[k]['Runtime'])

    print(f"\n  Best HV  (↑ higher is better) : {best_hv_name}")
    print(f"  Best MID (↓ lower is better)  : {best_mid_name}")
    print(f"  Best SP  (↓ lower is better)  : {best_sp_name}")
    print(f"  Best Cost(↓ lower is better)  : {best_cost_name}")
    print(f"  Best Sat (↑ higher is better) : {best_sat_name}")
    print(f"  Best Time(↓ lower is better)  : {best_rt_name}")
    print("=" * 94)

    return metrics_table


if __name__ == "__main__":
    # 数据加载参数与三个求解器的 __main__ 块完全一致
    common_data = data_loader.load_solomon_data(
        'solomon标准算例-时间窗/r1/r101.txt',
        n_customers=100,
        random_seed=42
    )
    print(common_data)

    metrics = run_comparison(common_data)
