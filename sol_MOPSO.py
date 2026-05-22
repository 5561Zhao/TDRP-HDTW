import numpy as np
import random
import matplotlib.pyplot as plt
import data_loader
import time

"""
多目标粒子群优化算法 (MOPSO) 求解 Risk-Aware TDRP
==================================================
与 sol_NSGA.py (混合模因算法) 及 sol_pure_NSGA.py (纯NSGA-II) 的对比基线。

问题定义 (与 sol_NSGA.py 完全一致):
  - 目标1: 最小化总行驶距离 (卡车曼哈顿距离 + 无人机欧几里得距离)
  - 目标2: 最大化鲁棒满意度 (存储为负值以统一为最小化)
  - 约束: 时间窗约束(软约束/惩罚)、无人机数量限制(2架)、客户类型约束(D→P双任务)

MOPSO 离散化设计:
  - 粒子编码: permutation (客户访问顺序排列)
  - 速度表示: swap sequence [(i,j), (k,l), ...], 表示一系列交换操作
  - 位置更新: x(t+1) = apply_swaps(x(t), v(t+1))
  - 速度更新: v = w⊗v + c1⊗(pbest⊖x) + c2⊗(gbest⊖x)
      ⊖: 求两排列的交换差序列
      ⊗: 概率截取 (以给定概率保留交换序列中的每个交换)
  - 个人最优(pbest): Pareto支配判断更新
  - 全局最优(gbest): 从外部存档中按拥挤度轮盘选择
  - 外部存档(Archive): 固定容量, 非支配解集 + 拥挤度裁剪
  - 解码器: 贪心Split (与纯NSGA-II一致, 保证公平比较)
  - 初始化: 纯随机
  - 约束处理: 与sol_NSGA.py一致 (可行性优先)
"""


# ==========================================
# 1. 问题实例与数据生成 (与 sol_NSGA.py 完全一致)
# ==========================================
class ProblemInstance:
    def __init__(self, common_data, num_customers):
        self.num_nodes = num_customers + 1
        self.nodes = list(range(self.num_nodes))

        self.NUM_DRONES = 2
        self.v_truck = 1.0
        self.v_drone = 2.0
        self.drone_capacity = 2.0
        self.service_duration = 0

        self.dtw_width_ratio = 0.5
        self.M = 10000.0
        self.stw = common_data['time_windows']
        self.dtw_width = {i: (self.stw[i][1] - self.stw[i][0]) * self.dtw_width_ratio for i in self.nodes}
        self.coords = common_data['coords']
        self.customer_types = common_data['types_int']

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
                self.dist_matrix_truck[i][j] = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                self.dist_matrix_drone[i][j] = np.linalg.norm(c1 - c2)

    def get_truck_dist(self, i, j):
        return self.dist_matrix_truck[i][j]

    def get_drone_dist(self, i, j):
        return self.dist_matrix_drone[i][j]

    def check_dual_task_constraints(self, j, l):
        """双任务约束: 必须是 D(配送) -> P(取件)"""
        if self.customer_types[j] != 2: return False
        if self.customer_types[l] != 1: return False
        return True


# ==========================================
# 2. 粒子编码
# ==========================================
class Particle:
    def __init__(self, num_customers):
        self.permutation = list(np.random.permutation(num_customers) + 1)
        self.decode_weight = random.uniform(0.0, 3.0)  # 解码权重: 平衡距离与时间的偏好
        self.velocity = []  # swap sequence: list of (i, j) index pairs
        self.weight_velocity = 0.0  # decode_weight 维度的连续速度 (标准PSO)

        self.objectives = [0.0, 0.0]
        self.decoded_schedule = None
        self.constraint_violation = 0.0
        self.is_feasible = True

        # 个人最优 (pbest)
        self.pbest_permutation = list(self.permutation)
        self.pbest_decode_weight = self.decode_weight  # 与 pbest_permutation 对应
        self.pbest_objectives = [float('inf'), float('inf')]
        self.pbest_constraint_violation = float('inf')
        self.pbest_is_feasible = False
        self.pbest_decoded_schedule = None

        # 用于NSGA兼容的属性 (指标计算/排序)
        self.rank = 0
        self.crowding_distance = 0.0


# ==========================================
# 3. 离散PSO核心算子
# ==========================================
def permutation_difference(perm_a, perm_b):
    """
    计算两个排列的交换差序列: perm_a ⊖ perm_b
    返回一个最小交换序列 swap_seq, 使得 apply_swaps(perm_b, swap_seq) == perm_a

    算法: 逐位对齐, 找到 perm_b 中与 perm_a[i] 不同的位置, 记录交换
    """
    target = list(perm_a)
    current = list(perm_b)
    size = len(current)
    swap_seq = []

    # 建立 current 中值到索引的映射
    pos_map = {}
    for idx, val in enumerate(current):
        pos_map[val] = idx

    for i in range(size):
        if current[i] != target[i]:
            # 找到 target[i] 在 current 中的位置
            j = pos_map[target[i]]
            # 记录交换 (i, j) — 注意这是排列索引
            swap_seq.append((i, j))
            # 执行交换并更新映射
            pos_map[current[i]] = j
            pos_map[current[j]] = i
            current[i], current[j] = current[j], current[i]

    return swap_seq


def probabilistic_select(swap_seq, probability):
    """
    概率截取算子 (⊗): 以给定概率保留交换序列中的每个交换
    对应离散PSO公式中的系数乘法
    """
    if probability <= 0:
        return []
    if probability >= 1.0:
        return list(swap_seq)
    return [swap for swap in swap_seq if random.random() < probability]


def apply_swaps(permutation, swap_seq):
    """
    将交换序列应用到排列上: x_new = apply_swaps(x, v)
    """
    result = list(permutation)
    size = len(result)
    for (i, j) in swap_seq:
        # 安全检查: 索引不越界
        if 0 <= i < size and 0 <= j < size:
            result[i], result[j] = result[j], result[i]
    return result


def merge_swap_sequences(*sequences):
    """
    合并多个交换序列 (顺序拼接)
    对应离散PSO速度更新中的加法 (+)
    """
    merged = []
    for seq in sequences:
        merged.extend(seq)
    return merged


def truncate_velocity(swap_seq, max_len):
    """
    速度截断: 限制交换序列长度, 防止过长导致搜索随机化
    """
    if len(swap_seq) <= max_len:
        return swap_seq
    return swap_seq[:max_len]


# ==========================================
# 4. 贪心 Split 解码器 (与 sol_pure_NSGA.py 完全一致)
# ==========================================
def greedy_split_decode(particle, instance):
    """
    贪心 Split 解码器 (与纯NSGA-II版本完全一致)
    """
    extended_perm = [0] + particle.permutation + [0]
    num_ext = len(extended_perm)

    v_truck = instance.v_truck
    v_drone = instance.v_drone
    max_drones = instance.NUM_DRONES

    INF = float('inf')
    dp_score = [INF] * num_ext  # 加权得分 (决策依据)
    dp_dist = [INF] * num_ext
    dp_time = [INF] * num_ext
    dp_prev = [-1] * num_ext
    dp_sorties = [None] * num_ext

    weight = particle.decode_weight

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

            if num_candidates == 0:
                segment_dist = truck_leg_dist
                segment_time = truck_leg_dist / v_truck

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

            new_dist = dp_dist[i] + segment_dist
            new_time = dp_time[i] + segment_time
            new_score = dp_score[i] + segment_dist + weight * v_truck * segment_time

            if new_score < dp_score[j]:
                dp_score[j] = new_score
                dp_dist[j] = new_dist
                dp_time[j] = new_time
                dp_prev[j] = i
                dp_sorties[j] = current_sortie_plan

    if dp_score[-1] == INF:
        final_truck_route = [0] + particle.permutation + [0]
        accepted_sorties = []
    else:
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

    total_sat, service_times, arrival_times, total_violation_time = fast_timing_and_satisfaction(
        instance, final_truck_route, accepted_sorties
    )

    particle.constraint_violation = total_violation_time
    particle.is_feasible = (total_violation_time < 1e-6)

    dist_truck = 0.0
    for i in range(len(final_truck_route) - 1):
        dist_truck += instance.get_truck_dist(final_truck_route[i], final_truck_route[i + 1])

    dist_drone = 0.0
    for (l, p_nodes, r) in accepted_sorties:
        dist_drone += instance.get_drone_dist(l, p_nodes[0])
        for k in range(len(p_nodes) - 1):
            dist_drone += instance.get_drone_dist(p_nodes[k], p_nodes[k + 1])
        dist_drone += instance.get_drone_dist(p_nodes[-1], r)

    particle.objectives[0] = dist_truck + dist_drone
    particle.objectives[1] = -total_sat
    particle.decoded_schedule = (final_truck_route, accepted_sorties)


# ==========================================
# 5. 时间推演与满意度计算 (与 sol_NSGA.py 完全一致)
# ==========================================
def fast_timing_and_satisfaction(instance, truck_route, drone_sorties):
    """
    快速计算给定路径的最优时间安排和满意度（带软约束惩罚）
    与 sol_NSGA.py 中的实现完全一致, 确保评估公平性
    """
    all_nodes = set(truck_route)
    for (_, path_nodes, _) in drone_sorties:
        for n in path_nodes:
            all_nodes.add(n)

    arrival_time = {}
    service_time = {}
    dependencies = {n: [] for n in all_nodes}

    return_to_depot_edge = None
    for k in range(len(truck_route) - 1):
        u = truck_route[k]
        v = truck_route[k + 1]
        if v == 0:
            return_to_depot_edge = (u, instance.get_truck_dist(u, v) / instance.v_truck)
            continue
        travel_t = instance.get_truck_dist(u, v) / instance.v_truck
        dependencies[v].append((u, travel_t, 'truck'))

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

    if return_to_depot_edge is not None:
        last_node, return_travel_time = return_to_depot_edge
        if last_node in service_time:
            depot_arrival = service_time[last_node] + return_travel_time
            depot_tw_end = instance.stw[0][1]
            depot_violation = max(0.0, depot_arrival - depot_tw_end)
            if depot_violation > 0:
                total_violation_time += depot_violation

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
# 6. 外部存档管理 (MOPSO核心组件)
# ==========================================
def dominates_constrained(p, q):
    """
    带约束的支配判断 (与 sol_NSGA.py 的规则完全一致)
    返回 True 如果 p 支配 q
    """
    # 规则1：可行解总是支配不可行解
    if p.is_feasible and not q.is_feasible:
        return True
    if not p.is_feasible and q.is_feasible:
        return False

    # 规则2：两者都可行，使用标准 Pareto 支配
    if p.is_feasible and q.is_feasible:
        return (p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and \
               (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1])

    # 规则3：两者都不可行，先比违规量再比目标
    if p.constraint_violation < q.constraint_violation:
        return True
    if p.constraint_violation > q.constraint_violation:
        return False
    return (p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and \
           (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1])


def objective_key(obj, ndigits=6):
    """Objective-space key used to prevent duplicate archive points."""
    return (round(float(obj[0]), ndigits), round(float(obj[1]), ndigits))


def unique_objective_particles(particles):
    unique = []
    seen = set()
    for particle in particles:
        key = objective_key(particle.objectives)
        if key not in seen:
            seen.add(key)
            unique.append(particle)
    return unique


def update_archive(archive, new_particle, max_size):
    """
    将新粒子加入外部存档, 维护非支配性质
    若存档满, 按拥挤度裁剪最拥挤的解
    """
    # 1. 检查新粒子是否被存档中任何解支配
    for arc in archive:
        if objective_key(arc.objectives) == objective_key(new_particle.objectives):
            return  # Same objective point; keep only one copy.
        if dominates_constrained(arc, new_particle):
            return  # 被支配, 不加入

    # 2. 移除被新粒子支配的旧解
    archive[:] = [arc for arc in archive if not dominates_constrained(new_particle, arc)]

    # 3. 加入新粒子 (深拷贝关键属性)
    new_entry = Particle(len(new_particle.permutation))
    new_entry.permutation = list(new_particle.permutation)
    new_entry.decode_weight = new_particle.decode_weight
    new_entry.objectives = list(new_particle.objectives)
    new_entry.constraint_violation = new_particle.constraint_violation
    new_entry.is_feasible = new_particle.is_feasible
    new_entry.decoded_schedule = new_particle.decoded_schedule
    archive.append(new_entry)

    # 4. 若超容量, 按拥挤度裁剪
    if len(archive) > max_size:
        _crowding_distance_archive(archive)
        # 移除拥挤度最小的 (非边界) 解
        archive.sort(key=lambda x: x.crowding_distance)
        archive.pop(0)  # 移除最拥挤的


def _crowding_distance_archive(archive):
    """计算存档中解的拥挤度 (与 sol_NSGA.py 的归一化逻辑一致)"""
    l = len(archive)
    if l == 0:
        return
    for p in archive:
        p.crowding_distance = 0.0

    min_objs = [min(ind.objectives[m] for ind in archive) for m in range(2)]
    max_objs = [max(ind.objectives[m] for ind in archive) for m in range(2)]

    for m in range(2):
        archive.sort(key=lambda x: x.objectives[m])
        archive[0].crowding_distance = float('inf')
        archive[l - 1].crowding_distance = float('inf')

        obj_diff = max_objs[m] - min_objs[m]
        if obj_diff == 0:
            continue

        for i in range(1, l - 1):
            dist_contribution = (archive[i + 1].objectives[m] - archive[i - 1].objectives[m]) / obj_diff
            archive[i].crowding_distance += dist_contribution


def select_leader(archive):
    """
    从外部存档中选择全局引导者 (gbest)
    策略: 按拥挤度的轮盘赌选择, 稀疏区域的解更易被选中, 促进Pareto前沿均匀扩展
    """
    if len(archive) == 0:
        return None
    if len(archive) == 1:
        return archive[0]

    _crowding_distance_archive(archive)

    # 拥挤度可能包含 inf, 替换为有限最大值
    max_finite_cd = 0.0
    for p in archive:
        if p.crowding_distance != float('inf') and p.crowding_distance > max_finite_cd:
            max_finite_cd = p.crowding_distance

    # 替换 inf 为 2 * max_finite (使边界解有较高但非无穷的权重)
    replace_val = max(max_finite_cd * 2.0, 1.0)
    weights = []
    for p in archive:
        w = replace_val if p.crowding_distance == float('inf') else p.crowding_distance
        weights.append(max(w, 1e-6))  # 避免零权重

    total = sum(weights)
    probs = [w / total for w in weights]

    # 轮盘赌选择
    r = random.random()
    cumsum = 0.0
    for idx, prob in enumerate(probs):
        cumsum += prob
        if r <= cumsum:
            return archive[idx]
    return archive[-1]


# ==========================================
# 7. pbest 更新逻辑
# ==========================================
def update_pbest(particle):
    """
    更新粒子的个人最优 (pbest)
    规则 (与 sol_NSGA.py 的约束支配逻辑一致):
      1. 新位置支配 pbest -> 替换
      2. pbest 支配新位置 -> 保留
      3. 互不支配 -> 随机选择一个保留
    """
    curr_dominates_pbest = _dominates_obj(particle, particle, is_pbest=True)
    pbest_dominates_curr = _pbest_dominates_obj(particle)

    if curr_dominates_pbest:
        _copy_to_pbest(particle)
    elif pbest_dominates_curr:
        pass  # 保留 pbest
    else:
        # 互不支配, 50% 概率替换
        if random.random() < 0.5:
            _copy_to_pbest(particle)


def _dominates_obj(p, q, is_pbest=False):
    """当前位置是否支配 pbest"""
    p_objs = p.objectives
    p_feas = p.is_feasible
    p_viol = p.constraint_violation

    q_objs = q.pbest_objectives if is_pbest else q.objectives
    q_feas = q.pbest_is_feasible if is_pbest else q.is_feasible
    q_viol = q.pbest_constraint_violation if is_pbest else q.constraint_violation

    if p_feas and not q_feas:
        return True
    if not p_feas and q_feas:
        return False

    if p_feas and q_feas:
        return (p_objs[0] <= q_objs[0] and p_objs[1] <= q_objs[1]) and \
               (p_objs[0] < q_objs[0] or p_objs[1] < q_objs[1])

    if p_viol < q_viol:
        return True
    if p_viol > q_viol:
        return False
    return (p_objs[0] <= q_objs[0] and p_objs[1] <= q_objs[1]) and \
           (p_objs[0] < q_objs[0] or p_objs[1] < q_objs[1])


def _pbest_dominates_obj(p):
    """pbest 是否支配当前位置"""
    pb_objs = p.pbest_objectives
    pb_feas = p.pbest_is_feasible
    pb_viol = p.pbest_constraint_violation

    c_objs = p.objectives
    c_feas = p.is_feasible
    c_viol = p.constraint_violation

    if pb_feas and not c_feas:
        return True
    if not pb_feas and c_feas:
        return False

    if pb_feas and c_feas:
        return (pb_objs[0] <= c_objs[0] and pb_objs[1] <= c_objs[1]) and \
               (pb_objs[0] < c_objs[0] or pb_objs[1] < c_objs[1])

    if pb_viol < c_viol:
        return True
    if pb_viol > c_viol:
        return False
    return (pb_objs[0] <= c_objs[0] and pb_objs[1] <= c_objs[1]) and \
           (pb_objs[0] < c_objs[0] or pb_objs[1] < c_objs[1])


def _copy_to_pbest(particle):
    """将当前位置拷贝到 pbest"""
    particle.pbest_permutation = list(particle.permutation)
    particle.pbest_decode_weight = particle.decode_weight  # 与 pbest_permutation 同步
    particle.pbest_objectives = list(particle.objectives)
    particle.pbest_constraint_violation = particle.constraint_violation
    particle.pbest_is_feasible = particle.is_feasible
    particle.pbest_decoded_schedule = particle.decoded_schedule


# ==========================================
# 8. 变异算子 (防止早熟收敛)
# ==========================================
def turbulence_mutation(particle, mutation_rate=0.1):
    """
    扰动变异: 以一定概率对粒子执行多次随机交换和权重重采样
    防止种群过早收敛到局部最优

    依据 Xu et al. (EJOR 2020) Lemma 5.2:
      变异算子需满足任意邻域可达性 P_delta(x*, x) > 0,
      即排列空间中任意目标排列在一步变异中以正概率可达。
      单次交换仅覆盖 swap-distance=1 的邻居, 不满足此条件。
      改为 randint(1, size) 次随机交换后, 任意排列均可达。
    依据 Proposition 4.3(b):
      变异后重置速度, 防止惯性分量将粒子拉回变异前位置导致停滞。
    """
    # 排列扰动
    if random.random() < mutation_rate:
        size = len(particle.permutation)
        num_swaps = random.randint(1, size)
        for _ in range(num_swaps):
            i, j = random.sample(range(size), 2)
            particle.permutation[i], particle.permutation[j] = particle.permutation[j], particle.permutation[i]
        particle.velocity = []

    # 权重扰动 (独立于排列扰动, 使用相同的变异概率)
    if random.random() < mutation_rate:
        particle.decode_weight = random.uniform(0.0, 3.0)
        particle.weight_velocity = 0.0


# ==========================================
# 9. 性能指标计算 (与 sol_NSGA.py 完全一致)
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
    sp = None
    n = len(pareto_front)
    if n >= 3:
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
# 10. 非支配排序 (用于最终输出, 与 sol_NSGA.py 完全一致)
# ==========================================
def fast_non_dominated_sort(population):
    """带约束的快速非支配排序"""
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            p_dominates_q = False
            q_dominates_p = False

            if p.is_feasible and not q.is_feasible:
                p_dominates_q = True
            elif not p.is_feasible and q.is_feasible:
                q_dominates_p = True
            elif p.is_feasible and q.is_feasible:
                if (p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and \
                        (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]):
                    p_dominates_q = True
                elif (q.objectives[0] <= p.objectives[0] and q.objectives[1] <= p.objectives[1]) and \
                        (q.objectives[0] < p.objectives[0] or q.objectives[1] < p.objectives[1]):
                    q_dominates_p = True
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


# ==========================================
# 11. 详细时间表打印 (与 sol_NSGA.py 完全一致)
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
# 12. 可视化 (与 sol_NSGA.py 结构一致)
# ==========================================
def plot_solution(instance, truck_route, drone_tasks, obj_vals=None):
    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 10))

    drone_styles = {
        0: {'color': 'tab:red', 'ls': '--', 'label': 'Drone #1'},
        1: {'color': 'tab:green', 'ls': '-.', 'label': 'Drone #2'},
    }

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

    truck_coords = np.array([instance.coords[n] for n in truck_route])
    plt.plot(truck_coords[:, 0], truck_coords[:, 1], color='black', linewidth=3,
             alpha=0.6, label='Truck', zorder=1)
    for k in range(len(truck_coords) - 1):
        p1, p2 = truck_coords[k], truck_coords[k + 1]
        plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.55, (p2[1] - p1[1]) * 0.55,
                  head_width=0.3, color='black', alpha=0.3, zorder=1)

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

    title = "MOPSO Solution Visualization"
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
# 13. 主程序入口
# ==========================================
def run_mopso_solver(common_data, shared_ref_point=None, shared_ref_front=None):
    """
    MOPSO 求解器
    与 sol_NSGA.py / sol_pure_NSGA.py 的对比:
      - 相同: 问题定义、约束处理、解码器(贪心Split)、评估函数、指标计算、参数规模
      - 不同: 搜索范式为粒子群优化 (速度-位置更新), 外部存档管理, 无非支配排序选择
    """
    # 参数配置 (与前两个算法保持一致的评估预算)
    SWARM_SIZE = 800       # 粒子数 (= POP_SIZE)
    MAX_ITER = 900          # 最大迭代次数 (= GEN_MAX)
    ARCHIVE_SIZE = 300     # 外部存档容量

    # PSO 超参数
    W_MAX = 0.9            # 惯性权重上界
    W_MIN = 0.4            # 惯性权重下界
    C1 = 1.5               # 认知系数 (个人学习因子)
    C2 = 1.5               # 社会系数 (全局学习因子)
    MUTATION_RATE = 0.15   # 扰动变异概率

    RESTART_RATE = 0.08    # light random restart to avoid one-point archive collapse

    NUM_CUSTOMERS = len(common_data['nodes']) - 1

    # 1. 初始化问题实例
    instance = ProblemInstance(common_data, num_customers=NUM_CUSTOMERS)
    data_loader.inject_data_to_nsga(instance, common_data)
    DTW_RATIO = 0.5

    for node_id in instance.nodes:
        if node_id == 0: continue
        stw_len = instance.stw[node_id][1] - instance.stw[node_id][0]
        instance.dtw_width[node_id] = stw_len * DTW_RATIO

    print("MOPSO Data Overwritten with Solomon File.")

    # 速度长度限制 (防止交换序列过长导致随机化)
    MAX_VELOCITY_LEN = NUM_CUSTOMERS

    # 2. 初始化粒子群 (混合启发式, 与 sol_NSGA.py 保持公平)
    t_solver_start = time.time()
    print("Initialize Swarm (Hybrid Heuristic for Fair Comparison)...")
    swarm = []
    archive = []

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

    limit_spatial = SWARM_SIZE // 4
    limit_temporal = SWARM_SIZE // 2
    limit_drone = (SWARM_SIZE * 3) // 4

    for i in range(SWARM_SIZE):
        p = Particle(NUM_CUSTOMERS)
        
        if i < limit_spatial:
            p.permutation = perturb_tour(nn_tour, strength=i)
        elif i < limit_temporal:
            p.permutation = perturb_tour(time_sorted_tour, strength=i - limit_spatial)
        elif i < limit_drone:
            p.permutation = perturb_tour(drone_aware_tour, strength=i - limit_temporal)
        else:
            p.permutation = list(np.random.permutation(NUM_CUSTOMERS) + 1)
            
        p.velocity = []  # 初始速度为空 (静止)
        greedy_split_decode(p, instance)
        _copy_to_pbest(p)  # 初始 pbest = 当前位置
        swarm.append(p)

        # 尝试加入存档
        update_archive(archive, p, ARCHIVE_SIZE)

    print(f"Initial Archive Size: {len(archive)}")

    # 3. 迭代主循环
    print("Start MOPSO Iteration...")
    for iteration in range(MAX_ITER):
        # 线性递减惯性权重
        w = W_MAX - (W_MAX - W_MIN) * (iteration / MAX_ITER)

        for p in swarm:
            # --- 3.1 选择全局引导者 ---
            leader = select_leader(archive)
            if leader is None:
                leader = p  # 存档为空时自引导 (极少见)

            # --- 3.2 速度更新 (离散PSO公式) ---
            # v(t+1) = w ⊗ v(t) + c1 ⊗ (pbest ⊖ x) + c2 ⊗ (gbest ⊖ x)

            # 惯性分量: 以概率 w 保留旧速度中的每个交换
            inertia_part = probabilistic_select(p.velocity, w)

            # 认知分量: pbest - x 的交换差序列, 以概率 c1*rand 截取
            cognitive_diff = permutation_difference(p.pbest_permutation, p.permutation)
            r1 = random.random()
            cognitive_part = probabilistic_select(cognitive_diff, C1 * r1)

            # 社会分量: gbest - x 的交换差序列, 以概率 c2*rand 截取
            social_diff = permutation_difference(leader.permutation, p.permutation)
            r2 = random.random()
            social_part = probabilistic_select(social_diff, C2 * r2)

            # 合并为新速度
            new_velocity = merge_swap_sequences(inertia_part, cognitive_part, social_part)

            # 速度截断
            new_velocity = truncate_velocity(new_velocity, MAX_VELOCITY_LEN)
            p.velocity = new_velocity

            # --- 3.2b decode_weight 维度 (连续PSO: 标准速度-位置公式) ---
            # 复用同一轮的 w, C1, C2, r1, r2, 无新增参数
            p.weight_velocity = (w * p.weight_velocity
                                 + C1 * r1 * (p.pbest_decode_weight - p.decode_weight)
                                 + C2 * r2 * (leader.decode_weight - p.decode_weight))

            # --- 3.3 位置更新 ---
            p.permutation = apply_swaps(p.permutation, new_velocity)
            p.decode_weight = p.decode_weight + p.weight_velocity
            # 边界约束: 与初始化范围一致 (Particle.__init__ 第93行: uniform(0.0, 3.0))
            p.decode_weight = max(0.0, min(3.0, p.decode_weight))

            # --- 3.4 扰动变异 (防止早熟收敛) ---
            turbulence_mutation(p, MUTATION_RATE)

            if len(unique_objective_particles(archive)) < 3 and random.random() < RESTART_RATE:
                seed = random.choice([nn_tour, time_sorted_tour, drone_aware_tour])
                p.permutation = perturb_tour(seed, strength=random.randint(1, 8))
                p.decode_weight = random.choice([0.0, 0.5, 1.5, 2.5, 3.0])
                p.velocity = []
                p.weight_velocity = 0.0

            # --- 3.5 解码评估 ---
            greedy_split_decode(p, instance)

            # --- 3.6 更新 pbest ---
            update_pbest(p)

            # --- 3.7 更新存档 ---
            update_archive(archive, p, ARCHIVE_SIZE)

        # 打印进度
        if iteration % 10 == 0:
            feasible_arc = [a for a in archive if a.is_feasible]
            if feasible_arc:
                best_dist = min(a.objectives[0] for a in feasible_arc)
                max_sat = -min(a.objectives[1] for a in feasible_arc)
                print(f"Iter {iteration}: Archive={len(archive)}, "
                      f"Best Feasible Dist={best_dist:.2f}, Max Robust Sat={max_sat:.2f}")
            else:
                print(f"Iter {iteration}: Archive={len(archive)}, No feasible sol yet.")

    # 4. 结果输出
    t_solver_end = time.time()
    solver_runtime = t_solver_end - t_solver_start
    print("\nMOPSO Iteration Finished.")
    print(f"Final Archive Size: {len(archive)}")

    # 使用存档作为 Pareto 前沿
    pareto_front = archive

    if not pareto_front:
        print("WARNING: Archive is empty. Using swarm for output.")
        pareto_front = swarm

    min_cost_sol = min(pareto_front, key=lambda p: p.objectives[0])
    max_sat_sol = min(pareto_front, key=lambda p: p.objectives[1])

    print("\n[Phase 1] Determining Bounds (Extracted from MOPSO Archive)...")
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

    # 5. Pareto前沿可视化
    dists = [p.objectives[0] for p in pareto_front]
    sats = [-p.objectives[1] for p in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.scatter(dists, sats, c='green', s=50, label='MOPSO Archive Solutions')
    plt.xlabel('Total Distance (Minimize)')
    plt.ylabel('Worst-Case Satisfaction (Maximize)')
    plt.title(f'MOPSO for Risk-Aware TDRP (N={NUM_CUSTOMERS})')
    plt.grid(True)
    plt.legend()
    plt.show()

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
    pareto_front = unique_objective_particles(pareto_front)
    mopso_front_objs = [list(ind.objectives) for ind in pareto_front]
    print(f"Unique metric front size (MOPSO): {len(mopso_front_objs)}")

    if shared_ref_point is not None:
        print(f"\n[Metric] Using Shared Reference Point: {shared_ref_point}")
        ref_point = shared_ref_point
    else:
        max_cost_obs = max(ind.objectives[0] for ind in pareto_front)
        max_neg_sat_obs = max(ind.objectives[1] for ind in pareto_front)
        ref_point = [max_cost_obs, max_neg_sat_obs]
        print(f"\n[Metric] Using Local Reference Point (Self-Adaptive): {ref_point}")

    hv_val, igd_val, sp_val = calculate_metrics(mopso_front_objs, ref_point, shared_ref_front)

    igd_str = f"{igd_val:.6f}" if igd_val is not None else "N/A (需提供shared_ref_front)"

    print("\n" + "=" * 40)
    print("Performance Metrics (MOPSO)")
    print("=" * 40)
    print(f"Reference Point        : {ref_point}")
    print(f"HV  (Hypervolume)      : {hv_val:.4f}")
    print(f"IGD (Inv. Gen. Dist.)  : {igd_str}")
    sp_str = f"{sp_val:.6f}" if sp_val is not None else "N/A (degenerate front)"
    print(f"SP  (Spacing)          : {sp_str}")
    print(f"Runtime (s)            : {solver_runtime:.2f}")
    print("=" * 40 + "\n")

    best_dist_sol = min(pareto_front, key=lambda p: p.objectives[0])
    return best_dist_sol.objectives[0], best_dist_sol.objectives[1], mopso_front_objs


if __name__ == "__main__":
    common_data = data_loader.load_solomon_data('solomon标准算例-时间窗/r1/r101.txt', n_customers=20,
                                                random_seed=42)
    print(common_data)
    t_start_g = time.time()
    run_mopso_solver(common_data)
    t_end_g = time.time()
    real_time_g = t_end_g - t_start_g
    print("run time:", real_time_g)
