import numpy as np
import random
import matplotlib.pyplot as plt
import data_loader
import time

"""
纯 NSGA-II 算法求解 Risk-Aware TDRP (卡车-无人机协同路径规划问题)
============================================================
与 sol_NSGA.py (混合模因算法) 的对比基线。

问题定义 (与 sol_NSGA.py 完全一致):
  - 目标1: 最小化总行驶距离 (卡车曼哈顿距离 + 无人机欧几里得距离)
  - 目标2: 最大化鲁棒满意度 (基于分布鲁棒的最坏情况时间窗满意度, 存储为负值以统一为最小化)
  - 约束: 时间窗约束(软约束/惩罚)、无人机数量限制(2架)、客户类型约束(D→P双任务)

纯 NSGA-II 设计要点:
  - 解码器: 贪心 Split (无 MOLS 多目标标签设定)
  - 变异: 仅经典算子 (swap / insertion / inversion), 无 ALNS
  - 无局部搜索 (无 VND 变邻域下降)
  - 初始化: 纯随机
  - 无解缓存
  - 无个体偏好权重基因
"""


# ==========================================
# 1. 问题实例与数据生成 (与 sol_NSGA.py 完全一致)
# ==========================================
class ProblemInstance:
    def __init__(self, common_data, num_customers):
        self.num_nodes = num_customers + 1
        self.nodes = list(range(self.num_nodes))

        # --- 核心参数 ---
        self.NUM_DRONES = 2  # 核心限制：无人机数量
        self.v_truck = 1.0
        self.v_drone = 2.0
        self.drone_capacity = 2.0
        self.service_duration = 0  # 服务时间

        self.dtw_width_ratio = 0.5
        self.M = 10000.0
        self.stw = common_data['time_windows']
        self.dtw_width = {i: (self.stw[i][1] - self.stw[i][0]) * self.dtw_width_ratio for i in self.nodes}
        self.coords = common_data['coords']
        self.customer_types = common_data['types_int']

        # [性能优化] 预计算距离矩阵
        self._precompute_distances()

    def _precompute_distances(self):
        """初始化时一次性计算所有节点对的距离"""
        n = self.num_nodes
        self.dist_matrix_truck = np.zeros((n, n))
        self.dist_matrix_drone = np.zeros((n, n))

        ordered_coords = []
        for i in range(n):
            ordered_coords.append(np.array(self.coords[i]))

        coords_np = np.array(ordered_coords)

        for i in range(n):
            for j in range(n):
                c1 = coords_np[i]
                c2 = coords_np[j]
                # 卡车：曼哈顿距离
                self.dist_matrix_truck[i][j] = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                # 无人机：欧几里得距离
                self.dist_matrix_drone[i][j] = np.linalg.norm(c1 - c2)

    def get_truck_dist(self, i, j):
        return self.dist_matrix_truck[i][j]

    def get_drone_dist(self, i, j):
        return self.dist_matrix_drone[i][j]

    def check_dual_task_constraints(self, j, l):
        """双任务约束: 必须是 D(配送) -> P(取件)"""
        if self.customer_types[j] != 2: return False  # J must be Delivery
        if self.customer_types[l] != 1: return False  # L must be Pickup
        return True


# ==========================================
# 2. 个体编码 (纯NSGA-II: 无偏好权重基因)
# ==========================================
class Individual:
    def __init__(self, num_customers):
        self.permutation = list(np.random.permutation(num_customers) + 1)
        self.decode_weight = random.uniform(0.0, 3.0)  # 解码权重: 平衡距离与时间的偏好
        self.objectives = [0.0, 0.0]
        self.rank = 0
        self.crowding_distance = 0.0
        self.decoded_schedule = None
        self.constraint_violation = 0.0
        self.is_feasible = True


# ==========================================
# 3. 贪心 Split 解码器 (纯NSGA-II的核心区别)
# ==========================================
def greedy_split_decode(ind, instance):
    """
    贪心 Split 解码器
    ================
    与 sol_NSGA.py 中 MOLS 多目标标签设定算法的区别:
      - MOLS: 维护每个节点的 Pareto 标签集, 通过支配剪枝搜索多条非支配路径
      - 贪心 Split: 顺序扫描, 基于距离节省量贪心决定是否派无人机

    策略:
      按 permutation 顺序扫描, 对每个位置尝试分配 0~NUM_DRONES 架无人机:
      - 计算纯卡车段的距离
      - 计算卡车+无人机并行的距离和时间
      - 选择距离节省量最大的方案 (贪心准则)
    """
    extended_perm = [0] + ind.permutation + [0]
    num_ext = len(extended_perm)

    v_truck = instance.v_truck
    v_drone = instance.v_drone
    max_drones = instance.NUM_DRONES

    # 动态规划表: dp[i] = (累积距离, 累积时间, 前驱索引, 该步的sortie方案)
    # 动态规划表: dp_score 为加权决策依据, dp_dist/dp_time 跟踪实际物理量
    INF = float('inf')
    dp_score = [INF] * num_ext  # 加权得分 (决策依据)
    dp_dist = [INF] * num_ext
    dp_time = [INF] * num_ext
    dp_prev = [-1] * num_ext
    dp_sorties = [None] * num_ext  # 存储每一步的无人机方案

    # 加权系数: weight=0 → 纯距离优化, weight越大 → 越重视时间(与满意度相关)
    # v_truck 用于统一量纲 (将时间转为等效距离)
    weight = ind.decode_weight

    dp_score[0] = 0.0
    dp_dist[0] = 0.0
    dp_time[0] = 0.0

    for i in range(num_ext - 1):
        if dp_score[i] == INF:
            continue

        curr_node = extended_perm[i]
        search_limit = min(i + 1 + max_drones + 1, num_ext)

        for j in range(i + 1, search_limit):
            next_truck_node = extended_perm[j]
            drone_candidates = extended_perm[i + 1: j]
            num_candidates = len(drone_candidates)

            truck_leg_dist = instance.get_truck_dist(curr_node, next_truck_node)

            segment_dist = 0.0
            segment_time = 0.0
            current_sortie_plan = []
            is_valid = True

            # Case 0: 纯卡车 (无无人机)
            if num_candidates == 0:
                segment_dist = truck_leg_dist
                segment_time = truck_leg_dist / v_truck

            # Case 1: 双任务无人机 (2个候选, 必须满足 D→P 约束)
            elif num_candidates == 2 and \
                    instance.check_dual_task_constraints(drone_candidates[0], drone_candidates[1]):
                u, v_node = drone_candidates[0], drone_candidates[1]

                d_flight = instance.get_drone_dist(curr_node, u) + \
                           instance.get_drone_dist(u, v_node) + \
                           instance.get_drone_dist(v_node, next_truck_node)

                segment_dist = truck_leg_dist + d_flight
                truck_t = truck_leg_dist / v_truck
                drone_t = d_flight / v_drone
                segment_time = max(truck_t, drone_t)

                current_sortie_plan.append((curr_node, [u, v_node], next_truck_node))

            # Case 2: 单任务无人机 (1个或多个独立sortie)
            else:
                if num_candidates > max_drones:
                    is_valid = False
                else:
                    total_flight_dist = 0.0
                    max_sortie_time = 0.0

                    for k_node in drone_candidates:
                        f_dist = instance.get_drone_dist(curr_node, k_node) + \
                                 instance.get_drone_dist(k_node, next_truck_node)
                        total_flight_dist += f_dist
                        flight_time = f_dist / v_drone
                        if flight_time > max_sortie_time:
                            max_sortie_time = flight_time

                        current_sortie_plan.append((curr_node, [k_node], next_truck_node))

                    segment_dist = truck_leg_dist + total_flight_dist
                    truck_t = truck_leg_dist / v_truck
                    segment_time = max(truck_t, max_sortie_time)

            if not is_valid:
                continue

            # 贪心准则: 最小化累积距离 (纯距离导向)
            new_dist = dp_dist[i] + segment_dist
            new_time = dp_time[i] + segment_time
            new_score = dp_score[i] + segment_dist + weight * v_truck * segment_time

            if new_score < dp_score[j]:
                dp_score[j] = new_score
                dp_dist[j] = new_dist
                dp_time[j] = new_time
                dp_prev[j] = i
                dp_sorties[j] = current_sortie_plan

    # 回溯构建路径
    if dp_score[-1] == INF:
        # 无可行分割, 回退纯卡车
        final_truck_route = [0] + ind.permutation + [0]
        accepted_sorties = []
    else:
        # 回溯
        path_indices = []
        idx = num_ext - 1
        while idx != -1:
            path_indices.append(idx)
            idx = dp_prev[idx]
        path_indices.reverse()

        final_truck_route = [extended_perm[k] for k in path_indices]
        accepted_sorties = []
        for k in path_indices:
            if dp_sorties[k]:
                accepted_sorties.extend(dp_sorties[k])

    # 精确评估
    total_sat, service_times, arrival_times, total_violation_time = fast_timing_and_satisfaction(
        instance, final_truck_route, accepted_sorties
    )

    ind.constraint_violation = total_violation_time
    ind.is_feasible = (total_violation_time < 1e-6)

    # 计算精确距离
    dist_truck = 0.0
    for i in range(len(final_truck_route) - 1):
        dist_truck += instance.get_truck_dist(final_truck_route[i], final_truck_route[i + 1])

    dist_drone = 0.0
    for (l, p_nodes, r) in accepted_sorties:
        dist_drone += instance.get_drone_dist(l, p_nodes[0])
        for k in range(len(p_nodes) - 1):
            dist_drone += instance.get_drone_dist(p_nodes[k], p_nodes[k + 1])
        dist_drone += instance.get_drone_dist(p_nodes[-1], r)

    ind.objectives[0] = dist_truck + dist_drone
    ind.objectives[1] = -total_sat
    ind.decoded_schedule = (final_truck_route, accepted_sorties)


# ==========================================
# 4. 时间推演与满意度计算 (与 sol_NSGA.py 完全一致)
# ==========================================
def fast_timing_and_satisfaction(instance, truck_route, drone_sorties):
    """
    快速计算给定路径的最优时间安排和满意度（带软约束惩罚）
    与 sol_NSGA.py 中的实现完全一致, 确保评估公平性
    """
    # 收集所有涉及节点
    all_nodes = set(truck_route)
    for (_, path_nodes, _) in drone_sorties:
        for n in path_nodes:
            all_nodes.add(n)

    # 到达时间和服务开始时间
    arrival_time = {}
    service_time = {}

    # 建立依赖关系图
    dependencies = {n: [] for n in all_nodes}

    # (A) 卡车路径依赖
    return_to_depot_edge = None
    for k in range(len(truck_route) - 1):
        u = truck_route[k]
        v = truck_route[k + 1]
        if v == 0:
            return_to_depot_edge = (u, instance.get_truck_dist(u, v) / instance.v_truck)
            continue
        travel_t = instance.get_truck_dist(u, v) / instance.v_truck
        dependencies[v].append((u, travel_t, 'truck'))

    # (B) 无人机路径依赖
    for (launch_node, path_nodes, recover_node) in drone_sorties:
        d_first = path_nodes[0]
        fly_out = instance.get_drone_dist(launch_node, d_first) / instance.v_drone
        dependencies[d_first].append((launch_node, fly_out, 'drone'))

        for k in range(len(path_nodes) - 1):
            dn_curr = path_nodes[k]
            dn_next = path_nodes[k + 1]
            fly = instance.get_drone_dist(dn_curr, dn_next) / instance.v_drone
            dependencies[dn_next].append((dn_curr, fly, 'drone'))

        if recover_node != 0:
            d_last = path_nodes[-1]
            fly_in = instance.get_drone_dist(d_last, recover_node) / instance.v_drone
            dependencies[recover_node].append((d_last, fly_in, 'drone'))

    # 拓扑排序
    in_degree = {n: len(dependencies[n]) for n in all_nodes}
    queue = [n for n in all_nodes if in_degree[n] == 0]
    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)

        for other_node in all_nodes:
            if node == other_node:
                continue
            for (pred, _, _) in dependencies[other_node]:
                if pred == node:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

    if len(topo_order) != len(all_nodes):
        return 0.0, {}, {}, float('inf')

    # 前向传播
    arrival_time[0] = 0.0
    service_time[0] = 0.0
    total_violation_time = 0.0

    for node in topo_order:
        if node == 0:
            continue

        tw_start = instance.stw[node][0]
        tw_end = instance.stw[node][1]
        max_arrival = tw_start

        for (pred_node, travel_t, mode) in dependencies[node]:
            if pred_node not in service_time:
                continue
            if mode == 'drone':
                pred_ready_time = arrival_time.get(pred_node, service_time[pred_node])
            else:
                pred_ready_time = service_time[pred_node]

            candidate_arrival = pred_ready_time + travel_t
            max_arrival = max(max_arrival, candidate_arrival)

        arrival_time[node] = max_arrival

        violation_time = max(0.0, max_arrival - tw_end)
        if violation_time > 0:
            total_violation_time += violation_time

        dtw_width = instance.dtw_width[node]
        slope_len = (tw_end - tw_start) - dtw_width

        if slope_len < 1e-5:
            optimal_service = max_arrival
        else:
            optimal_center = (tw_start + tw_end) / 2.0
            if max_arrival <= tw_end:
                optimal_service = max(max_arrival, min(optimal_center, tw_end))
            else:
                optimal_service = max_arrival

        service_time[node] = optimal_service

    # 检查返回Depot的归巢约束
    if return_to_depot_edge is not None:
        last_node, return_travel_time = return_to_depot_edge
        if last_node in service_time:
            depot_arrival = service_time[last_node] + return_travel_time
            depot_tw_end = instance.stw[0][1]
            depot_violation = max(0.0, depot_arrival - depot_tw_end)
            if depot_violation > 0:
                total_violation_time += depot_violation

    # 计算总满意度 (Paper 4 分布鲁棒公式)
    total_satisfaction = 0.0
    for node in all_nodes:
        if node == 0:
            continue

        t_service = service_time[node]
        tw_start = instance.stw[node][0]
        tw_end = instance.stw[node][1]

        delta_i = (tw_end - tw_start) - instance.dtw_width[node]

        if delta_i < 1e-5:
            alpha = 1.0
        else:
            sat_leftmost = (tw_end - t_service) / delta_i
            sat_rightmost = (t_service - tw_start) / delta_i
            alpha = max(0.0, min(1.0, sat_leftmost, sat_rightmost))

        total_satisfaction += alpha

    return total_satisfaction, service_time, arrival_time, total_violation_time


# ==========================================
# 5. NSGA-II 标准框架
# ==========================================

# --- 5.1 快速非支配排序 (与 sol_NSGA.py 完全一致) ---
def fast_non_dominated_sort(population):
    """
    带约束的快速非支配排序
    规则：可行解总是支配不可行解；不可行解之间比较违规量
    """
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            p_dominates_q = False
            q_dominates_p = False

            # 规则1：可行解总是支配不可行解
            if p.is_feasible and not q.is_feasible:
                p_dominates_q = True
            elif not p.is_feasible and q.is_feasible:
                q_dominates_p = True
            # 规则2：两者都可行，使用标准Pareto支配
            elif p.is_feasible and q.is_feasible:
                if (p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and \
                        (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]):
                    p_dominates_q = True
                elif (q.objectives[0] <= p.objectives[0] and q.objectives[1] <= p.objectives[1]) and \
                        (q.objectives[0] < p.objectives[0] or q.objectives[1] < p.objectives[1]):
                    q_dominates_p = True
            # 规则3：两者都不可行，比较违规量和目标
            else:
                if p.constraint_violation < q.constraint_violation:
                    p_dominates_q = True
                elif p.constraint_violation > q.constraint_violation:
                    q_dominates_p = True
                else:
                    if (p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and \
                            (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]):
                        p_dominates_q = True
                    elif (q.objectives[0] <= p.objectives[0] and q.objectives[1] <= p.objectives[1]) and \
                            (q.objectives[0] < p.objectives[0] or q.objectives[1] < p.objectives[1]):
                        q_dominates_p = True

            if p_dominates_q:
                p.dominated_solutions.append(q)
            elif q_dominates_p:
                p.domination_count += 1

        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


# --- 5.2 拥挤度计算 (与 sol_NSGA.py 完全一致, 含归一化) ---
def crowding_distance_assignment(front):
    l = len(front)
    if l == 0: return
    for p in front: p.crowding_distance = 0

    min_objs = [min(ind.objectives[m] for ind in front) for m in range(2)]
    max_objs = [max(ind.objectives[m] for ind in front) for m in range(2)]

    for m in range(2):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = float('inf')
        front[l - 1].crowding_distance = float('inf')

        obj_diff = max_objs[m] - min_objs[m]
        if obj_diff == 0:
            continue

        for i in range(1, l - 1):
            dist_contribution = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_diff
            front[i].crowding_distance += dist_contribution


# --- 5.3 锦标赛选择 (与 sol_NSGA.py 完全一致) ---
def tournament_selection(population):
    i, j = random.sample(range(len(population)), 2)
    p1, p2 = population[i], population[j]
    if p1.rank < p2.rank:
        return p1
    elif p2.rank < p1.rank:
        return p2
    elif p1.crowding_distance > p2.crowding_distance:
        return p1
    else:
        return p2


# --- 5.4 OX交叉 (纯NSGA-II: 无偏好权重继承) ---
def crossover(p1, p2, instance):
    size = len(p1.permutation)
    s, e = sorted(random.sample(range(size), 2))

    c1_perm = [-1] * size
    c2_perm = [-1] * size

    c1_perm[s:e] = p1.permutation[s:e]
    c2_perm[s:e] = p2.permutation[s:e]

    def fill_remaining(child_perm, fill_parent):
        curr = e
        for i in range(size):
            cand = fill_parent.permutation[(e + i) % size]
            if cand not in child_perm[s:e]:
                child_perm[curr] = cand
                curr = (curr + 1) % size

    fill_remaining(c1_perm, p2)
    fill_remaining(c2_perm, p1)

    c1 = Individual(size)
    c1.permutation = c1_perm
    c1.decode_weight = p1.decode_weight  # 继承父代1的解码权重

    c2 = Individual(size)
    c2.permutation = c2_perm
    c2.decode_weight = p2.decode_weight  # 继承父代2的解码权重

    return c1, c2


# --- 5.5 经典变异 (纯NSGA-II: 仅 swap / insertion / inversion, 无 ALNS) ---
def mutation(ind, instance):
    """
    纯 NSGA-II 经典变异算子
    与 sol_NSGA.py 的区别: 移除了 ALNS (自适应大邻域搜索) 算子
    包含四种变异算子:
      1. Insertion Mutation (插入变异)
      2. Swap Mutation (交换变异)
      3. Inversion Mutation (逆序变异 / 2-opt)
      4. Weight Mutation (解码权重重采样, 改变距离-时间权衡偏好)
    """
    size = len(ind.permutation)
    r = random.choice([1, 2, 3, 4])

    if r == 1:
        # Insertion Mutation (插入变异)
        i, j = random.sample(range(size), 2)
        val_node = ind.permutation.pop(i)
        ind.permutation.insert(j, val_node)

    elif r == 2:
        # Swap Mutation (交换变异)
        i, j = random.sample(range(size), 2)
        ind.permutation[i], ind.permutation[j] = ind.permutation[j], ind.permutation[i]

    elif r == 3:
        # Inversion Mutation (逆序变异)
        i, j = sorted(random.sample(range(size), 2))
        ind.permutation[i:j + 1] = ind.permutation[i:j + 1][::-1]

    elif r == 4:
        # Weight Mutation (解码权重重采样)
        # 重采样范围与初始化一致, 使解码器探索不同的距离-时间权衡区域
        ind.decode_weight = random.uniform(0.0, 3.0)


# ==========================================
# 6. 性能指标计算 (与 sol_NSGA.py 完全一致)
# ==========================================
def calculate_metrics(pareto_front, reference_point, reference_front=None):
    """
    计算性能指标: HV (超体积), IGD (反向世代距离), Spacing (间距)
    pareto_front: list of [obj1, obj2] (假设均为最小化方向)
    reference_point: [ref_obj1, ref_obj2] (HV 所需, 必须被所有解支配)
    reference_front: list of [obj1, obj2] (IGD 所需的参考前沿, 可选)
    """
    # 1. 计算 HV (Hypervolume) - 2D Minimization
    sorted_front = sorted(pareto_front, key=lambda x: x[0])

    hv = 0.0
    ref_x, ref_y = reference_point

    for i in range(len(sorted_front)):
        curr_x, curr_y = sorted_front[i]

        if curr_x > ref_x or curr_y > ref_y:
            continue

        if i < len(sorted_front) - 1:
            next_x = sorted_front[i + 1][0]
            width = next_x - curr_x
        else:
            width = ref_x - curr_x

        height = ref_y - curr_y

        if width > 0 and height > 0:
            hv += width * height

    # 2. 计算 IGD (Inverted Generational Distance)
    # 定义: IGD(A, R) = (1/|R|) * Σ_{r∈R} min_{a∈A} dist(r, a)
    # 需要外部参考前沿; 若未提供则返回 None
    # 归一化: 使用参考前沿自身的目标极差, 消除量纲差异
    igd = None
    if reference_front is not None and len(reference_front) > 0:
        ref_obj0 = [p[0] for p in reference_front]
        ref_obj1 = [p[1] for p in reference_front]
        range_0 = max(ref_obj0) - min(ref_obj0)
        range_1 = max(ref_obj1) - min(ref_obj1)
        if range_0 < 1e-10: range_0 = 1.0  # 防止除零 (所有解在该目标上相同)
        if range_1 < 1e-10: range_1 = 1.0

        total_min_dist = 0.0
        for ref_pt in reference_front:
            min_dist = float('inf')
            for sol in pareto_front:
                d = (((sol[0] - ref_pt[0]) / range_0) ** 2 +
                     ((sol[1] - ref_pt[1]) / range_1) ** 2) ** 0.5
                if d < min_dist:
                    min_dist = d
            total_min_dist += min_dist
        igd = total_min_dist / len(reference_front)

    # 3. 计算 Spacing (Schott, 1995)
    # 定义: SP = sqrt( (1/(n-1)) * Σ (d_bar - d_i)^2 )
    # 其中 d_i = min_{j≠i} L1_normalized(i, j)
    # 值越小, 解分布越均匀
    # 归一化: 使用自身前沿的目标极差
    sp = 0.0
    n = len(pareto_front)
    if n > 1:
        front_obj0 = [p[0] for p in pareto_front]
        front_obj1 = [p[1] for p in pareto_front]
        range_f0 = max(front_obj0) - min(front_obj0)
        range_f1 = max(front_obj1) - min(front_obj1)
        if range_f0 < 1e-10: range_f0 = 1.0
        if range_f1 < 1e-10: range_f1 = 1.0

        d_list = []
        for i in range(n):
            min_d = float('inf')
            for j in range(n):
                if i == j:
                    continue
                d = abs(pareto_front[i][0] - pareto_front[j][0]) / range_f0 + \
                    abs(pareto_front[i][1] - pareto_front[j][1]) / range_f1
                if d < min_d:
                    min_d = d
            d_list.append(min_d)
        d_bar = sum(d_list) / n
        sp = (sum((d - d_bar) ** 2 for d in d_list) / (n - 1)) ** 0.5

    return hv, igd, sp


# ==========================================
# 7. 详细时间表打印 (与 sol_NSGA.py 完全一致)
# ==========================================
def print_detailed_schedule(instance, individual):
    """对给定的个体进行详细的时间推演和满意度计算，并打印表格"""
    truck_route, drone_sorties = individual.decoded_schedule

    print("\n" + "=" * 60)
    print("      DETAILED SCHEDULE ANALYSIS")
    print("=" * 60)

    total_sat, service_times, arrival_times, total_violation_time = fast_timing_and_satisfaction(
        instance, truck_route, drone_sorties
    )

    is_feasible = (total_violation_time < 1e-6)
    feasibility_status = "可行" if is_feasible else f"不可行(违规{total_violation_time:.1f}秒)"
    print(f"\n整体评估:")
    print(f"  总满意度: {total_sat:.4f}")
    print(f"  可行性: {feasibility_status}")
    print(f"  约束违规: {individual.constraint_violation:.2f} 秒")
    print()

    final_data = []
    involved_nodes = set(truck_route)
    for (_, p_nodes, _) in drone_sorties:
        for n in p_nodes:
            involved_nodes.add(n)

    node_satisfaction = {}
    for node in involved_nodes:
        if node == 0:
            node_satisfaction[0] = None
            continue

        t_service = service_times.get(node, instance.stw[node][0])
        tw_start = instance.stw[node][0]
        tw_end = instance.stw[node][1]

        delta_i = (tw_end - tw_start) - instance.dtw_width[node]

        if delta_i < 1e-5:
            alpha = 1.0
        else:
            sat_leftmost = (tw_end - t_service) / delta_i
            sat_rightmost = (t_service - tw_start) / delta_i
            alpha = max(0.0, min(1.0, sat_leftmost, sat_rightmost))

        node_satisfaction[node] = alpha

    depot_start = service_times.get(0, 0.0)
    final_data.append({
        'id': 0, 'type': 'Depot', 'arr': depot_start, 'srv': depot_start,
        'sat': '-', 'tw_start': instance.stw[0][0], 'tw_end': instance.stw[0][1]
    })

    for k in range(len(truck_route) - 1):
        u = truck_route[k]
        v = truck_route[k + 1]
        if v == 0: continue

        srv_v = service_times.get(v, instance.stw[v][0])
        arr_v = arrival_times.get(v, service_times.get(v, instance.stw[v][0]))
        sat_v = node_satisfaction.get(v, 0.0)

        final_data.append({
            'id': v, 'type': 'Truck', 'arr': arr_v, 'srv': srv_v,
            'sat': sat_v, 'tw_start': instance.stw[v][0], 'tw_end': instance.stw[v][1]
        })

    for (launch_node, path_nodes, recover_node) in drone_sorties:
        t_launch = arrival_times.get(launch_node, instance.stw[launch_node][0])
        prev_node = launch_node
        curr_time = t_launch

        for d_node in path_nodes:
            fly_t = instance.get_drone_dist(prev_node, d_node) / instance.v_drone
            arr_d = curr_time + fly_t
            srv_d = service_times.get(d_node, instance.stw[d_node][0])
            sat_d = node_satisfaction.get(d_node, 0.0)

            final_data.append({
                'id': d_node, 'type': 'Drone', 'arr': arr_d, 'srv': srv_d,
                'sat': sat_d, 'tw_start': instance.stw[d_node][0], 'tw_end': instance.stw[d_node][1]
            })
            curr_time = srv_d
            prev_node = d_node

    if truck_route[-1] == 0:
        last_customer = truck_route[-2]
        travel_t = instance.get_truck_dist(last_customer, 0) / instance.v_truck
        srv_last = service_times.get(last_customer, instance.stw[last_customer][0])
        depot_arrival = srv_last + travel_t

        final_data.append({
            'id': 0, 'type': 'Depot(End)', 'arr': depot_arrival, 'srv': depot_arrival,
            'sat': '-', 'tw_start': instance.stw[0][0], 'tw_end': instance.stw[0][1]
        })

    final_data.sort(key=lambda x: (x['id'], x['type']))

    print(f"{'Node':<6} {'Type':<12} {'TW':<18} {'Arrival':<10} {'Service':<10} {'Sat':<8} {'Status':<8}")
    print("-" * 82)

    for item in final_data:
        tw_str = f"[{item['tw_start']:.0f},{item['tw_end']:.0f}]"

        if isinstance(item['sat'], float):
            sat_str = f"{item['sat']:.4f}"
        else:
            sat_str = item['sat']

        if item['arr'] > item['tw_end'] + 1e-3:
            status = "晚到"
        elif item['srv'] > item['tw_end'] + 1e-3:
            status = "超时"
        else:
            status = "正常"

        print(f"{item['id']:<6} {item['type']:<12} {tw_str:<18} "
              f"{item['arr']:<10.2f} {item['srv']:<10.2f} {sat_str:<8} {status:<8}")

    print("-" * 82)
    print(f"\n说明: ")
    print(f"  - TW: 时间窗 [开始, 结束]")
    print(f"  - Arrival: 到达时间")
    print(f"  - Service: 服务开始时间")
    print(f"  - Sat: 鲁棒满意度 (0-1)")
    print(f"  - Status: 正常/晚到/超时")
    print("=" * 60)


# ==========================================
# 8. 可视化 (与 sol_NSGA.py 结构一致)
# ==========================================
def plot_solution(instance, truck_route, drone_tasks, obj_vals=None):
    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 10))

    # 无人机路径
    drone_styles = {
        0: {'color': 'tab:red', 'ls': '--', 'label': 'Drone #1'},
        1: {'color': 'tab:green', 'ls': '-.', 'label': 'Drone #2'},
    }

    # 简单按顺序分配 Drone ID
    for idx, (l, p_nodes, r) in enumerate(drone_tasks):
        d_id = idx % instance.NUM_DRONES
        style = drone_styles.get(d_id, {'color': 'black', 'ls': '-', 'label': f'Drone #{d_id + 1}'})

        full_path = [l] + p_nodes + [r]
        coords = np.array([instance.coords[n] for n in full_path])

        lbl = style['label'] if idx < instance.NUM_DRONES else ""
        plt.plot(coords[:, 0], coords[:, 1], color=style['color'], linestyle=style['ls'],
                 linewidth=2, alpha=0.8, zorder=2, label=lbl)

        for k in range(len(coords) - 1):
            p1, p2 = coords[k], coords[k + 1]
            plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.5, (p2[1] - p1[1]) * 0.5,
                      head_width=0.3, color=style['color'], length_includes_head=True, zorder=3)

    # 卡车路径
    truck_coords = np.array([instance.coords[n] for n in truck_route])
    plt.plot(truck_coords[:, 0], truck_coords[:, 1], color='black', linewidth=3,
             alpha=0.6, label='Truck', zorder=1)

    for k in range(len(truck_coords) - 1):
        p1, p2 = truck_coords[k], truck_coords[k + 1]
        plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.55, (p2[1] - p1[1]) * 0.55,
                  head_width=0.3, color='black', alpha=0.3, zorder=1)

    # 节点
    for n in instance.nodes:
        c = instance.coords[n]
        if n == 0:
            plt.scatter(c[0], c[1], c='black', marker='s', s=200, zorder=10, label='Depot')
            plt.text(c[0], c[1], "0", color='white', ha='center', va='center', fontweight='bold')
        else:
            ctype = instance.customer_types[n]
            col = 'orange' if ctype == 1 else ('dodgerblue' if ctype == 2 else 'purple')
            mark = '^' if ctype == 1 else ('o' if ctype == 2 else 'D')
            plt.scatter(c[0], c[1], c=col, marker=mark, s=120, edgecolors='k', zorder=5)
            plt.text(c[0], c[1], str(n), color='white', ha='center', va='center', fontweight='bold', fontsize=9)

    title = "Pure NSGA-II Solution Visualization"
    if obj_vals: title += f"\nDist: {obj_vals[0]:.2f} | Sat: {-obj_vals[1]:.2f}"
    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label['Type D'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10)
    by_label['Type P'] = Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10)
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# 9. 主程序入口
# ==========================================
def run_pure_nsga_solver(common_data, shared_ref_point=None, shared_ref_front=None):
    """
    纯 NSGA-II 求解器
    与 sol_NSGA.py 的 run_nsga_solver 对比:
      - 相同: 种群规模、迭代次数、选择/交叉机制、非支配排序、拥挤度、指标计算
      - 不同: 解码器(贪心Split)、变异(无ALNS)、无局部搜索、纯随机初始化、无缓存
    """
    # 参数配置 (与 sol_NSGA.py 完全一致)
    POP_SIZE = 500
    GEN_MAX = 600
    NUM_CUSTOMERS = len(common_data['nodes']) - 1

    # 1. 初始化并注入数据
    instance = ProblemInstance(common_data, num_customers=NUM_CUSTOMERS)
    data_loader.inject_data_to_nsga(instance, common_data)
    DTW_RATIO = 0.5

    for node_id in instance.nodes:
        if node_id == 0: continue
        stw_len = instance.stw[node_id][1] - instance.stw[node_id][0]
        instance.dtw_width[node_id] = stw_len * DTW_RATIO

    print("Pure NSGA-II Data Overwritten with Solomon File.")

    # 2. 初始化种群 (混合启发式, 与 sol_NSGA.py 保持公平)
    t_solver_start = time.time()
    print("Initialize Population (Hybrid Heuristic for Fair Comparison)...")
    population = []

    # --- 辅佐函数 (直接借鉴 sol_NSGA.py) ---
    def get_nn_tour(inst):
        unvisited = set(inst.nodes)
        unvisited.remove(0)
        curr = 0
        tour = []
        while unvisited:
            nxt = min(unvisited, key=lambda n: inst.get_truck_dist(curr, n))
            tour.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return tour

    def get_drone_aware_tour(inst, customers):
        d_nodes = [n for n in customers if inst.customer_types[n] == 2]
        p_nodes = [n for n in customers if inst.customer_types[n] == 1]
        spd_nodes = [n for n in customers if inst.customer_types[n] == 3]

        d_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        p_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        spd_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))

        drone_pairs = []
        remaining_d = list(d_nodes)
        remaining_p = list(p_nodes)
        while remaining_d and remaining_p:
            drone_pairs.append((remaining_d.pop(0), remaining_p.pop(0)))
        unpaired = remaining_d + remaining_p

        tour = []
        pi, si = 0, 0
        while pi < len(drone_pairs) or si < len(spd_nodes):
            if si < len(spd_nodes):
                tour.append(spd_nodes[si])
                si += 1
            if pi < len(drone_pairs):
                tour.append(drone_pairs[pi][0])
                tour.append(drone_pairs[pi][1])
                pi += 1
        tour.extend(unpaired)
        return tour

    def perturb_tour(tour, strength=1):
        if strength == 0:
            return list(tour)
        new_tour = list(tour)
        for _ in range(strength):
            idx1, idx2 = sorted(random.sample(range(len(new_tour)), 2))
            seg = new_tour[idx1:idx2]
            seg.reverse()
            new_tour[idx1:idx2] = seg
        return new_tour

    # 预计算基准路径
    nn_tour = get_nn_tour(instance)
    all_customers = [n for n in instance.nodes if n != 0]
    time_sorted_tour = sorted(all_customers, key=lambda n: instance.stw[n][0])
    drone_aware_tour = get_drone_aware_tour(instance, all_customers)

    limit_spatial = POP_SIZE // 4
    limit_temporal = POP_SIZE // 2
    limit_drone = (POP_SIZE * 3) // 4

    for i in range(POP_SIZE):
        ind = Individual(NUM_CUSTOMERS)

        # if i < limit_spatial:
        #     ind.permutation = perturb_tour(nn_tour, strength=i)
        # elif i < limit_temporal:
        #     ind.permutation = perturb_tour(time_sorted_tour, strength=i - limit_spatial)
        # elif i < limit_drone:
        #     ind.permutation = perturb_tour(drone_aware_tour, strength=i - limit_temporal)
        # else:
        ind.permutation = list(np.random.permutation(NUM_CUSTOMERS) + 1)

        greedy_split_decode(ind, instance)
        population.append(ind)

    # 3. 进化主循环
    print("Start Evolution (Pure NSGA-II)...")
    for gen in range(GEN_MAX):
        offspring = []

        # 常规 NSGA-II 进化流程
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            c1, c2 = crossover(p1, p2, instance)
            mutation(c1, instance)
            mutation(c2, instance)
            greedy_split_decode(c1, instance)
            greedy_split_decode(c2, instance)
            offspring.extend([c1, c2])

        # 合并 + 非支配排序 + 精英保留
        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)

        new_pop = []
        for front in fronts:
            crowding_distance_assignment(front)
            if len(new_pop) + len(front) <= POP_SIZE:
                new_pop.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_pop.extend(front[:POP_SIZE - len(new_pop)])
                break
        population = new_pop

        # 无局部搜索 (纯 NSGA-II)

        if gen % 10 == 0:
            feasible_pop = [p for p in population if p.is_feasible]
            if feasible_pop:
                best_dist = min(p.objectives[0] for p in feasible_pop)
                max_sat = -min(p.objectives[1] for p in feasible_pop)
                print(f"Gen {gen}: Best Feasible Dist={best_dist:.2f}, Max Robust Sat={max_sat:.2f}")
            else:
                raw_dist = min(p.objectives[0] for p in population)
                print(f"Gen {gen}: No feasible sol yet. Raw Dist={raw_dist:.2f}")

    # 4. 结果输出
    t_solver_end = time.time()
    solver_runtime = t_solver_end - t_solver_start
    print("\nEvolution Finished (Pure NSGA-II).")
    pareto_front = fast_non_dominated_sort(population)[0]

    min_cost_sol = min(pareto_front, key=lambda p: p.objectives[0])
    max_sat_sol = min(pareto_front, key=lambda p: p.objectives[1])

    print("\n[Phase 1] Determining Bounds (Extracted from Pure NSGA-II Pareto Front)...")
    print("  -> Max Satisfaction Solution:")
    print(f"     Max Sat: {-max_sat_sol.objectives[1]:.2f} (Cost: {max_sat_sol.objectives[0]:.2f})")
    ms_truck_route, ms_drone_sorties = max_sat_sol.decoded_schedule
    print(f"     Truck Route: {ms_truck_route}")
    print(f"     Drone Sorties: {ms_drone_sorties}")

    print("  -> Min Cost Solution:")
    print(f"     Min Cost: {min_cost_sol.objectives[0]:.2f} (Sat: {-min_cost_sol.objectives[1]:.2f})")
    mc_truck_route, mc_drone_sorties = min_cost_sol.decoded_schedule
    print(f"     Truck Route: {mc_truck_route}")
    print(f"     Drone Sorties: {mc_drone_sorties}")

    print("\nPrinting detailed schedule for Min Cost Solution:")
    print_detailed_schedule(instance, min_cost_sol)

    print("\nPrinting detailed schedule for Max Sat Solution:")
    print_detailed_schedule(instance, max_sat_sol)

    # # 5. Pareto前沿可视化
    # dists = [p.objectives[0] for p in pareto_front]
    # sats = [-p.objectives[1] for p in pareto_front]
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(dists, sats, c='blue', s=50, label='Pure NSGA-II Pareto Solutions')
    # plt.xlabel('Total Distance (Minimize)')
    # plt.ylabel('Worst-Case Satisfaction (Maximize)')
    # plt.title(f'Pure NSGA-II for Risk-Aware TDRP (N={NUM_CUSTOMERS})')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # 6. 路径可视化
    # best_sol = pareto_front[0]
    # print("\n--- Sample Pareto Solution ---")
    # print(f"Distance: {best_sol.objectives[0]:.2f}")
    # print(f"Satisfaction: {-best_sol.objectives[1]:.2f}")
    # t_route, d_tasks = best_sol.decoded_schedule
    # print(f"Truck Route: {t_route}")
    # print(f"Drone Sorties (Launch, Serve, Recover): {d_tasks}")
    #
    # print("Plotting best solution...")
    # plot_solution(instance, t_route, d_tasks, obj_vals=best_sol.objectives)

    # 7. 指标计算
    pareto_front_pop = fast_non_dominated_sort(population)[0]
    nsga_front_objs = [ind.objectives for ind in pareto_front_pop]

    if shared_ref_point is not None:
        print(f"\n[Metric] Using Shared Reference Point: {shared_ref_point}")
        ref_point = shared_ref_point
    else:
        max_cost_obs = max(ind.objectives[0] for ind in pareto_front_pop)
        max_neg_sat_obs = max(ind.objectives[1] for ind in pareto_front_pop)
        ref_point = [max_cost_obs, max_neg_sat_obs]
        print(f"\n[Metric] Using Local Reference Point (Self-Adaptive): {ref_point}")

    hv_val, igd_val, sp_val = calculate_metrics(nsga_front_objs, ref_point, shared_ref_front)

    igd_str = f"{igd_val:.6f}" if igd_val is not None else "N/A (需提供shared_ref_front)"

    print("\n" + "=" * 40)
    print("Performance Metrics (Pure NSGA-II)")
    print("=" * 40)
    print(f"Reference Point        : {ref_point}")
    print(f"HV  (Hypervolume)      : {hv_val:.4f}")
    print(f"IGD (Inv. Gen. Dist.)  : {igd_str}")
    print(f"SP  (Spacing)          : {sp_val:.6f}")
    print(f"Runtime (s)            : {solver_runtime:.2f}")
    print("=" * 40 + "\n")

    best_dist_sol = min(pareto_front_pop, key=lambda p: p.objectives[0])
    return best_dist_sol.objectives[0], best_dist_sol.objectives[1], nsga_front_objs


if __name__ == "__main__":
    common_data = data_loader.load_solomon_data('solomon标准算例-时间窗/r1/r101.txt', n_customers=50,
                                                random_seed=42)
    print(common_data)
    t_start_g = time.time()
    run_pure_nsga_solver(common_data)
    t_end_g = time.time()
    real_time_g = t_end_g - t_start_g
    print("run time:", real_time_g)
