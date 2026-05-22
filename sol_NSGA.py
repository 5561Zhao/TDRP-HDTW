import numpy as np
import random
import matplotlib.pyplot as plt
import data_loader
import time
from numba import jit


@jit(nopython=True)
def calc_robust_satisfaction_jit(t_service, tw_start, tw_end, dtw_width):
    # 逻辑完全对应原代码，但预编译为机器码
    delta_i = (tw_end - tw_start) - dtw_width

    if delta_i < 1e-5:
        return 1.0

    # 对应论文 Paper 4 的鲁棒满意度公式
    sat_leftmost = (tw_end - t_service) / delta_i
    sat_rightmost = (t_service - tw_start) / delta_i

    # 手动实现 max/min 以确保类型稳定
    alpha = min(1.0, sat_leftmost)
    alpha = min(alpha, sat_rightmost)
    alpha = max(0.0, alpha)

    return alpha


@jit(nopython=True)
def calc_dist_matrix_jit(coords):
    n = len(coords)
    # 初始化矩阵
    mat_truck = np.zeros((n, n), dtype=np.float64)  # 显式指定 float64
    mat_drone = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            c1 = coords[i]
            c2 = coords[j]
            # 曼哈顿距离
            mat_truck[i, j] = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            # 欧几里得距离
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            mat_drone[i, j] = (dx * dx + dy * dy) ** 0.5

    return mat_truck, mat_drone


# --- [新增] Numba 加速的核心计算函数 ---

@jit(nopython=True)
def check_dominance_jit(existing_objs, new_dist, new_time):
    """
    检查 (new_dist, new_time) 是否被 existing_objs 中的任意解支配。
    existing_objs: (N, 2) 的 float64 数组
    返回: True (被支配, 丢弃), False (不被支配)
    """
    n = existing_objs.shape[0]
    eps = 1e-5
    for i in range(n):
        e_dist = existing_objs[i, 0]
        e_time = existing_objs[i, 1]

        # 现有解 e 支配 新解 new ?
        # 条件: e_dist <= new_dist AND e_time <= new_time 且至少有一个严格小于
        if e_dist <= new_dist + eps and e_time <= new_time + eps:
            if e_dist < new_dist - eps or e_time < new_time - eps:
                return True
    return False


@jit(nopython=True)
def fast_timing_core_jit(num_nodes, truck_route, sorties_encoded,
                         stw_start, stw_end, dtw_width,
                         dist_mat_truck, dist_mat_drone,
                         v_truck, v_drone):
    """
    [修正版] 包含严格的鲁棒服务时间计算逻辑，修复满意度计算错误。
    """
    # 1. 预分配内存
    max_edges = len(truck_route) + sorties_encoded.shape[0] * 3

    head = np.full(num_nodes, -1, dtype=np.int32)
    next_edge = np.full(max_edges, -1, dtype=np.int32)
    to_node = np.zeros(max_edges, dtype=np.int32)
    weight = np.zeros(max_edges, dtype=np.float64)
    mode = np.zeros(max_edges, dtype=np.int32)

    edge_cnt = 0
    in_degree = np.zeros(num_nodes, dtype=np.int32)
    active_nodes = np.zeros(num_nodes, dtype=np.bool_)

    for i in range(len(truck_route)):
        active_nodes[truck_route[i]] = True

    # (A) 构建卡车边
    ret_depot_time = 0.0
    has_ret = False
    last_truck_node = -1

    for k in range(len(truck_route) - 1):
        u = truck_route[k]
        v = truck_route[k + 1]

        if v == 0:
            has_ret = True
            last_truck_node = u
            ret_depot_time = dist_mat_truck[u, v] / v_truck
            continue

        w = dist_mat_truck[u, v] / v_truck

        idx = edge_cnt;
        edge_cnt += 1
        to_node[idx] = v;
        weight[idx] = w;
        mode[idx] = 0
        next_edge[idx] = head[u];
        head[u] = idx
        in_degree[v] += 1

    # (B) 构建无人机边
    n_sorties = sorties_encoded.shape[0]
    for i in range(n_sorties):
        l_node = sorties_encoded[i, 0]
        n1 = sorties_encoded[i, 1]
        n2 = sorties_encoded[i, 2]
        r_node = sorties_encoded[i, 3]

        active_nodes[n1] = True
        w1 = dist_mat_drone[l_node, n1] / v_drone

        idx = edge_cnt;
        edge_cnt += 1
        to_node[idx] = n1;
        weight[idx] = w1;
        mode[idx] = 1
        next_edge[idx] = head[l_node];
        head[l_node] = idx
        in_degree[n1] += 1

        curr = n1
        if n2 != -1:
            active_nodes[n2] = True
            w2 = dist_mat_drone[n1, n2] / v_drone
            idx = edge_cnt;
            edge_cnt += 1
            to_node[idx] = n2;
            weight[idx] = w2;
            mode[idx] = 1
            next_edge[idx] = head[n1];
            head[n1] = idx
            in_degree[n2] += 1
            curr = n2

        if r_node != 0:
            w3 = dist_mat_drone[curr, r_node] / v_drone
            idx = edge_cnt;
            edge_cnt += 1
            to_node[idx] = r_node;
            weight[idx] = w3;
            mode[idx] = 1
            next_edge[idx] = head[curr];
            head[curr] = idx
            in_degree[r_node] += 1

    # (C) 拓扑排序与时间传播
    queue = np.zeros(num_nodes, dtype=np.int32)
    q_head = 0
    q_tail = 0

    for i in range(num_nodes):
        if active_nodes[i] and in_degree[i] == 0:
            queue[q_tail] = i;
            q_tail += 1

    arrival_time = np.full(num_nodes, -1.0, dtype=np.float64)
    service_time = np.full(num_nodes, -1.0, dtype=np.float64)

    arrival_time[0] = 0.0
    service_time[0] = 0.0

    processed_count = 0
    active_count = 0
    for i in range(num_nodes):
        if active_nodes[i]: active_count += 1

    total_violation = 0.0
    total_sat = 0.0

    while q_head < q_tail:
        u = queue[q_head];
        q_head += 1
        processed_count += 1

        e_idx = head[u]
        while e_idx != -1:
            v = to_node[e_idx]
            w = weight[e_idx]
            m = mode[e_idx]

            start_t = 0.0
            if m == 1:
                if arrival_time[u] > -0.1:
                    start_t = arrival_time[u]
                else:
                    start_t = service_time[u]
            else:
                start_t = service_time[u]

            prop_time = start_t + w
            if prop_time > arrival_time[v]:
                arrival_time[v] = prop_time

            in_degree[v] -= 1
            if in_degree[v] == 0:
                # --- [关键修正开始] ---
                arr_v = arrival_time[v]
                tw_s = stw_start[v]
                tw_e = stw_end[v]
                w_i = dtw_width[v]

                # 1. 违规计算
                if arr_v > tw_e:
                    total_violation += (arr_v - tw_e)

                # 2. 服务时间计算 (完全还原原代码逻辑)
                # 计算斜坡长度，判断时间窗是否过窄
                slope_len = (tw_e - tw_s) - w_i

                opt_srv = arr_v  # 默认初始化

                if slope_len < 1e-5:
                    # 时间窗非常窄，只能尽早服务
                    opt_srv = max(arr_v, tw_s)
                else:
                    # 鲁棒优化：尝试在时间窗中间服务
                    optimal_center = (tw_s + tw_e) * 0.5

                    if arr_v <= tw_e:
                        val = optimal_center
                        # 限制逻辑：
                        # 1. 不能晚于时间窗结束 (虽然center通常在中间，但防止w_i异常)
                        if tw_e < val: val = tw_e
                        # 2. 核心：如果到达时间比中间点晚，被迫推迟服务时间
                        if arr_v > val: val = arr_v
                        # 3. 补充：不能早于时间窗开始 (虽然center > tw_s，但防止浮点误差)
                        if val < tw_s: val = tw_s

                        opt_srv = val
                    else:
                        # 已经迟到，只能立即服务
                        opt_srv = arr_v

                service_time[v] = opt_srv

                # 3. 满意度计算 (逻辑未变，但输入变了)
                delta_i = (tw_e - tw_s) - w_i
                if delta_i < 1e-5:
                    sat = 1.0
                else:
                    sat_left = (tw_e - opt_srv) / delta_i
                    sat_right = (opt_srv - tw_s) / delta_i

                    # 手写 min/max
                    sat = sat_left
                    if sat_right < sat: sat = sat_right
                    if sat > 1.0: sat = 1.0
                    if sat < 0.0: sat = 0.0

                total_sat += sat
                # --- [关键修正结束] ---

                queue[q_tail] = v;
                q_tail += 1

            e_idx = next_edge[e_idx]

    if has_ret and last_truck_node != -1:
        last_srv = service_time[last_truck_node]
        if last_srv > -0.1:
            depot_arr = last_srv + ret_depot_time
            depot_end = stw_end[0]
            if depot_arr > depot_end:
                total_violation += (depot_arr - depot_end)

    valid = (processed_count == active_count)
    return total_sat, service_time, arrival_time, total_violation, valid
# ==========================================
# 1. 问题实例与数据生成
# ==========================================
class ProblemInstance:
    def __init__(self, common_data, num_customers):
        self.num_nodes = num_customers + 1
        self.nodes = list(range(self.num_nodes))

        # --- 新增/修改参数 ---
        self.NUM_DRONES = 2
        self.v_truck = 1.0
        self.v_drone = 2.0
        self.drone_capacity = 2.0
        self.service_duration = 0

        self.dtw_width_ratio = 0.5
        self.stw = common_data['time_windows']

        raw_coords = common_data['coords']
        ordered_coords = []
        for i in range(self.num_nodes):
            # 兼容 list 或 dict：无论是 list[i] 还是 dict[i] 都能取到
            ordered_coords.append(raw_coords[i])

        # 现在 ordered_coords 是纯列表 [[x,y], [x,y]...]，可以安全转换
        self.coords = np.array(ordered_coords, dtype=np.float64)

        self.customer_types = common_data['types_int']

        # [优化] 将 dtw_width 改为数组
        self.dtw_width = np.zeros(self.num_nodes, dtype=np.float64)
        for i in self.nodes:
            # 确保 stw 取值也是 float 运算
            width = float(self.stw[i][1] - self.stw[i][0]) * self.dtw_width_ratio
            self.dtw_width[i] = width

        # [性能优化] 预计算距离矩阵
        self._precompute_distances()

    def _precompute_distances(self):
        # [优化] 调用 JIT 编译后的函数
        self.dist_matrix_truck, self.dist_matrix_drone = calc_dist_matrix_jit(self.coords)

    def get_truck_dist(self, i, j):
        # [性能优化] O(1) 查表
        return self.dist_matrix_truck[i][j]

    def get_drone_dist(self, i, j):
        # [性能优化] O(1) 查表
        return self.dist_matrix_drone[i][j]

    def check_dual_task_constraints(self, j, l):
        # 必须是 D -> P
        if self.customer_types[j] != 2: return False  # J must be Delivery
        if self.customer_types[l] != 1: return False  # L must be Pickup
        return True

# ==========================================
# 2. 染色体 (保持不变)
# ==========================================
class Individual:
    def __init__(self, num_customers):
        self.permutation = list(np.random.permutation(num_customers) + 1)

        self.objectives = [0.0, 0.0]
        self.rank = 0
        self.crowding_distance = 0.0
        self.decoded_schedule = None
        self.constraint_violation = 0.0
        self.is_feasible = True
        self.preferred_weight = random.uniform(0.0, 3.0)  # 新增：个体的偏好权重，在初始化时确定

# ==========================================
# 3. 核心解码器 (深度重构)
# ==========================================
SOLUTION_CACHE = {}
CACHE_LIMIT = 0


def decode_and_evaluate(ind, instance, priority_mode=None, use_exact_solver=False, progress=0.0, time_weight=None):
    selection_weight = time_weight if time_weight is not None else ind.preferred_weight
    discrete_weight = round(selection_weight, 1)
    gene_key = (tuple(ind.permutation), discrete_weight)

    segment_plan_cache = {}

    if gene_key in SOLUTION_CACHE:
        cached_res = SOLUTION_CACHE[gene_key]
        ind.objectives = list(cached_res['objs'])
        ind.constraint_violation = cached_res['viol']
        ind.is_feasible = cached_res['feas']
        ind.decoded_schedule = cached_res['sched']
        return

    extended_perm = [0] + ind.permutation + [0]
    num_nodes = len(extended_perm)

    # [结构重构] 使用分离的数据结构代替 Object
    labels_objs = [[] for _ in range(num_nodes)]
    labels_info = [[] for _ in range(num_nodes)]

    labels_objs_np = [np.empty((0, 2), dtype=np.float64) for _ in range(num_nodes)]
    # 起点初始化
    labels_objs[0].append([0.0, 0.0])
    labels_info[0].append((-1, -1, None))

    labels_objs_np[0] = np.array(labels_objs[0], dtype=np.float64)

    v_truck = instance.v_truck
    v_drone = instance.v_drone
    MAX_LABELS = 5

    for i in range(num_nodes - 1):
        if not labels_objs[i]:
            continue

        curr_front_np = labels_objs_np[i]
        curr_info = labels_info[i]

        curr_node = extended_perm[i]
        search_limit = min(i + 1 + instance.NUM_DRONES + 1, num_nodes)
        # search_limit = min(i + 1 + instance.NUM_DRONES * 2 + 1, num_nodes)
        for j in range(i + 1, search_limit):
            next_truck_node = extended_perm[j]
            drone_candidates = extended_perm[i + 1: j]
            num_candidates = len(drone_candidates)
            truck_leg_dist = instance.get_truck_dist(curr_node, next_truck_node)

            if num_candidates == 0:
                all_plans = [([], truck_leg_dist, truck_leg_dist / v_truck)]
            else:

                segment_key = (tuple(drone_candidates), curr_node, next_truck_node)

                if segment_key in segment_plan_cache:
                    all_plans = segment_plan_cache[segment_key]
                else:
                    all_plans = _enumerate_segment_plans(
                        drone_candidates, curr_node, next_truck_node,
                        instance, truck_leg_dist, v_truck, v_drone
                    )
                    segment_plan_cache[segment_key] = all_plans

            tw_end_j = instance.stw[next_truck_node][1] if next_truck_node != 0 else float('inf')

            target_front = labels_objs[j]
            target_front_np = labels_objs_np[j]

            # 临时缓冲区，用于本轮扩展
            new_entries_obj = []
            new_entries_info = []

            for sortie_plan, segment_dist, segment_time in all_plans:
                # 遍历当前节点的所有标签 (k是索引)
                for k in range(len(curr_front_np)):
                    prev_dist = curr_front_np[k, 0]
                    prev_time = curr_front_np[k, 1]

                    new_cum_dist = prev_dist + segment_dist
                    new_cum_time = prev_time + segment_time

                    if next_truck_node != 0 and new_cum_time > tw_end_j + instance.mean_tw_width:
                        continue

                    # [核心优化] 调用 JIT 函数判断支配
                    # 1. 检查是否被目标节点已有的标签支配
                    if check_dominance_jit(target_front_np, new_cum_dist, new_cum_time):
                        continue

                    # 2. 检查是否被本轮新产生的标签支配 (简单的贪心去重，避免 Python 循环过慢)
                    # 为性能考虑，这里略去对 new_entries 的全量两两比较，依靠 step 1 和后续的剪枝

                    new_entries_obj.append([new_cum_dist, new_cum_time])
                    new_entries_info.append((i, k, sortie_plan))

            # 批量更新到目标节点
            if new_entries_obj:
                # 标签目标与最终目标对齐：距离 + 满意度代理；时间保留为可扩展性维度。

                combined_objs = target_front + new_entries_obj
                combined_info = labels_info[j] + new_entries_info
                combined = list(zip(combined_objs, combined_info))
                combined.sort(key=lambda x: (x[0][0], x[0][1]))

                final_objs = []
                final_info = []
                min_time_so_far = float('inf')
                for obj, inf in combined:
                    current_time = obj[1]
                    if current_time < min_time_so_far - 1e-5:
                        final_objs.append(obj)
                        final_info.append(inf)
                        min_time_so_far = current_time
                labels_objs[j] = final_objs
                labels_info[j] = final_info
                if final_objs:
                    labels_objs_np[j] = np.array(final_objs, dtype=np.float64)
                else:
                    labels_objs_np[j] = np.empty((0, 2), dtype=np.float64)

    # 4. 路径回溯 (Selection & Backtracking)
    final_node_idx = num_nodes - 1
    final_objs = labels_objs[final_node_idx]
    final_infos = labels_info[final_node_idx]

    if not final_objs:
        final_truck_route = [0] + ind.permutation + [0]
        accepted_sorties = []
    else:
        best_idx = -1
        best_score = float('inf')

        for k, obj in enumerate(final_objs):
            score = obj[0] + obj[1] * instance.v_truck * selection_weight
            if score < best_score:
                best_score = score
                best_idx = k

        final_truck_route = []
        accepted_sorties = []

        curr_node_i = final_node_idx
        curr_label_k = best_idx

        trace_nodes = []
        trace_sorties = []

        while curr_node_i > 0:
            trace_nodes.append(extended_perm[curr_node_i])

            prev_i, prev_k, sortie_plan = labels_info[curr_node_i][curr_label_k]

            if sortie_plan:
                trace_sorties = sortie_plan + trace_sorties

            curr_node_i = prev_i
            curr_label_k = prev_k

        final_truck_route = [0] + trace_nodes[::-1]
        accepted_sorties = trace_sorties

    # 6. 最终精确评估 [修改为调用 JIT Wrapper]
    total_sat, _, _, total_violation_time = fast_timing_and_satisfaction(
        instance, final_truck_route, accepted_sorties
    )

    ind.constraint_violation = total_violation_time
    ind.is_feasible = (total_violation_time < 1e-6)

    # 距离计算保持 Python 即可，开销不大
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

    SOLUTION_CACHE[gene_key] = {
        'objs': tuple(ind.objectives),
        'viol': ind.constraint_violation,
        'feas': ind.is_feasible,
        'sched': ind.decoded_schedule
    }
    if len(SOLUTION_CACHE) > CACHE_LIMIT:
        SOLUTION_CACHE.clear()


def _enumerate_segment_plans(candidates, launch_node, recover_node,
                             instance, truck_leg_dist, v_truck, v_drone):
    """
    [优化版] 基于物理约束（同步性与几何性）的生成器，替代纯暴力枚举。

    第一性原理约束：
    1. 资源约束：候选节点数不能超过无人机的最大携带能力。
    2. 时间同步：无人机飞行时间不应显著长于卡车行驶时间 (避免卡车长时间等待)。
    3. 几何三角：若 Drone 距离 >> Truck 距离，则该分配物理上低效。
    """
    num_drones = instance.NUM_DRONES
    get_dd = instance.get_drone_dist
    n_cand = len(candidates)

    # [规则 1] 基数剪枝 (Cardinality Pruning)
    # 鸽巢原理：如果候选节点数太多，无人机根本运不过来，直接放弃
    # 假设每个 sortie 最多服务 2 个节点 (Dual Task)
    max_capacity = num_drones * 2
    if n_cand > max_capacity:
        return []

    # 卡车这一段的基准时间
    truck_leg_time = truck_leg_dist / v_truck

    # [规则 2] 定义时间同步容忍度 (Synchronization Tolerance)
    # 允许无人机比卡车慢一点，但不能慢太多。
    # 设定为 2.0 倍卡车时间 (即允许卡车等待的时间 = 卡车路程时间)
    # 这是一个物理限制：如果等待时间超过路程时间，不如直接让卡车去送。
    max_allowed_drone_time = truck_leg_time * 2.0 + 300  # +300秒作为固定缓冲

    # 预计算：单个节点的可行性 (几何剪枝预处理)
    # 如果某个节点太远，导致单次往返都超时，则包含该节点的任何组合都不可行
    valid_single_indices = []
    for i in range(n_cand):
        node = candidates[i]
        # 估算最小飞行时间：Launch -> Node -> Recover
        min_flight_dist = get_dd(launch_node, node) + get_dd(node, recover_node)
        if min_flight_dist / v_drone <= max_allowed_drone_time:
            valid_single_indices.append(i)

    # 如果有节点连单次任务都做不到，且它必须被分配（candidates必须全部分配），
    # 那么整个 candidates 集合就是不可行的。
    if len(valid_single_indices) < n_cand:
        return []

    results = []
    visited_states = set()

    def _backtrack(idx, partition):
        state = (idx, len(partition))

        if state in visited_states:
            return

        visited_states.add(state)
        # 成功基准：所有候选节点都已分配
        if idx == n_cand:
            sortie_plan = []
            total_flight_dist = 0.0
            max_sortie_time = 0.0

            for group in partition:
                # 构建路径
                path_nodes = group

                # 计算距离
                # Launch -> First
                fd = get_dd(launch_node, path_nodes[0])
                # Inter-nodes
                for k in range(len(path_nodes) - 1):
                    fd += get_dd(path_nodes[k], path_nodes[k + 1])
                # Last -> Recover
                fd += get_dd(path_nodes[-1], recover_node)

                # [规则 3] 严格的时间同步检查 (Strict Time Check)
                # 在生成方案的一瞬间，如果发现某架无人机飞太久，直接丢弃该方案
                st = fd / v_drone
                if st > max_allowed_drone_time:
                    return  # 剪枝：此方案无效

                total_flight_dist += fd
                if st > max_sortie_time:
                    max_sortie_time = st

                sortie_plan.append((launch_node, list(path_nodes), recover_node))

            # 只有当无人机最长飞行时间在可接受范围内时，才接受此方案
            seg_dist = truck_leg_dist + total_flight_dist
            seg_time = max(truck_leg_time, max_sortie_time)

            results.append((sortie_plan, seg_dist, seg_time))
            return

        # 剪枝：如果已用的无人机数量达到上限
        if len(partition) >= num_drones:
            return

        # 尝试分配 candidates[idx]

        # 选项 A: 单任务 (Single Task)
        # 将当前节点作为一个新的 sortie
        partition.append([candidates[idx]])
        _backtrack(idx + 1, partition)
        partition.pop()

        # 选项 B: 双任务 (Dual Task)
        # 尝试将当前节点与下一个节点打包
        # 约束：必须连续，且满足 D->P 约束
        if idx + 1 < n_cand:
            c1 = candidates[idx]
            c2 = candidates[idx + 1]
            if instance.check_dual_task_constraints(c1, c2):
                partition.append([c1, c2])
                _backtrack(idx + 2, partition)
                partition.pop()

    _backtrack(0, [])
    return results
def _reassign_drone_endpoints(truck_route, sorties, instance):
    """
    后处理优化：在确定的卡车路径上，为每个 sortie 重新选择
    起飞点和回收点，允许无人机跨越中间卡车停靠点飞行。

    不改变 truck_route（卡车停靠哪些节点不变）。
    不改变每个 sortie 的 path_nodes（哪些节点由无人机服务不变）。
    只改变每个 sortie 的 launch_node 和 recover_node。

    核心保证：以原始分配（合法）为基准，仅在不违反 NUM_DRONES 约束时才改进。
    约束：任意卡车路径段上同时在途的无人机数 ≤ instance.NUM_DRONES。
    输出格式与输入完全一致：[(launch, [path_nodes], recover), ...]
    """
    if not sorties:
        return sorties

    num_stops = len(truck_route)
    num_drones = instance.NUM_DRONES
    get_dd = instance.get_drone_dist

    def _flight_dist(l_node, path_nodes, r_node):
        """计算一个 sortie 的总飞行距离"""
        fd = get_dd(l_node, path_nodes[0])
        for k in range(len(path_nodes) - 1):
            fd += get_dd(path_nodes[k], path_nodes[k + 1])
        fd += get_dd(path_nodes[-1], r_node)
        return fd

    # --- Step 1: 确定每个 sortie 在 truck_route 上的原始 (l_idx, r_idx) ---
    # 构建节点 → 在 truck_route 中出现的所有位置索引
    node_positions = {}
    for idx, node in enumerate(truck_route):
        if node not in node_positions:
            node_positions[node] = []
        node_positions[node].append(idx)

    original_assignments = []
    for (l_node, path_nodes, r_node) in sorties:
        best_l, best_r = None, None
        for li in node_positions.get(l_node, []):
            for ri in node_positions.get(r_node, []):
                if ri > li:
                    # 选跨度最小的匹配（原始 Split 产生的分配跨度最短）
                    if best_l is None or (ri - li) < (best_r - best_l):
                        best_l, best_r = li, ri
        if best_l is None:
            # 理论上不会发生（Split 产生的 launch/recover 一定在 truck_route 上）
            best_l, best_r = 0, num_stops - 1
        original_assignments.append((best_l, best_r))

    # --- Step 2: 用原始分配初始化 segment_usage（保证合法基线） ---
    segment_usage = [0] * (num_stops - 1)
    for (l_idx, r_idx) in original_assignments:
        for seg in range(l_idx, r_idx):
            segment_usage[seg] += 1

    # --- Step 3: 为每个 sortie 生成候选并尝试改进 ---
    all_candidates = []
    orig_fds = []
    for s_idx, (l_node, path_nodes, r_node) in enumerate(sorties):
        orig_fds.append(_flight_dist(l_node, path_nodes, r_node))
        candidates = []
        for l_i in range(num_stops - 1):
            for r_i in range(l_i + 1, num_stops):
                fd = _flight_dist(truck_route[l_i], path_nodes, truck_route[r_i])
                candidates.append((fd, l_i, r_i))
        candidates.sort()  # 飞行距离升序
        all_candidates.append(candidates)

    # 按「改善潜力」降序处理：原始飞行距离与最佳候选之差越大，优先级越高
    sortie_indices = list(range(len(sorties)))
    sortie_indices.sort(key=lambda s: all_candidates[s][0][0] - orig_fds[s])

    final_assignments = list(original_assignments)  # 默认保持原始

    for s_idx in sortie_indices:
        orig_l, orig_r = original_assignments[s_idx]
        orig_fd = orig_fds[s_idx]

        # 从 segment_usage 中暂时移除当前 sortie 的占用
        for seg in range(orig_l, orig_r):
            segment_usage[seg] -= 1

        # 在所有候选中寻找飞行距离更短且不违反约束的分配
        improved = False
        for (fd, l_i, r_i) in all_candidates[s_idx]:
            if fd >= orig_fd:
                # 后续候选距离只会更大，不可能改进
                break

            can_assign = all(
                segment_usage[seg] < num_drones
                for seg in range(l_i, r_i)
            )
            if can_assign:
                final_assignments[s_idx] = (l_i, r_i)
                for seg in range(l_i, r_i):
                    segment_usage[seg] += 1
                improved = True
                break

        if not improved:
            # 无法改进：恢复原始分配（保证合法性）
            for seg in range(orig_l, orig_r):
                segment_usage[seg] += 1

    # --- Step 4: 构建新 sorties ---
    new_sorties = []
    for s_idx, (_, path_nodes, _) in enumerate(sorties):
        l_idx, r_idx = final_assignments[s_idx]
        new_sorties.append((truck_route[l_idx], list(path_nodes), truck_route[r_idx]))

    # --- 安全门：距离减少 + 可行性不恶化 才采纳 ---
    new_total_fd = sum(_flight_dist(l, pn, r) for (l, pn, r) in new_sorties)
    orig_total_fd = sum(orig_fds)

    if new_total_fd >= orig_total_fd:
        return sorties

    _, _, _, new_violation = fast_timing_and_satisfaction(
        instance, truck_route, new_sorties
    )
    _, _, _, orig_violation = fast_timing_and_satisfaction(
        instance, truck_route, sorties
    )

    if new_violation <= orig_violation + 1e-6:
        return new_sorties
    else:
        return sorties
# ==========================================
# 5. NSGA-II 框架 (保持不变)
# ==========================================
def fast_non_dominated_sort(population):
    """
    带约束的快速非支配排序
    规则：可行解总是支配不可行解；不可行解之间比较违规量
    [优化] 预提取目标值和约束信息为局部数组，减少内循环中的属性查找开销
    """
    n = len(population)

    # [优化] 预提取：将每个个体的关键属性提取到紧凑的局部数组中
    obj0 = [p.objectives[0] for p in population]
    obj1 = [p.objectives[1] for p in population]
    feasible = [p.is_feasible for p in population]
    violation = [p.constraint_violation for p in population]

    domination_count = [0] * n
    dominated_sets = [[] for _ in range(n)]

    first_front_indices = []
    for i in range(n):
        p_o0 = obj0[i]; p_o1 = obj1[i]
        p_feas = feasible[i]; p_viol = violation[i]

        for j in range(n):
            if i == j:
                continue
            q_o0 = obj0[j]; q_o1 = obj1[j]
            q_feas = feasible[j]; q_viol = violation[j]

            p_dominates_q = False
            q_dominates_p = False

            # 规则1：可行解总是支配不可行解
            if p_feas and not q_feas:
                p_dominates_q = True
            elif not p_feas and q_feas:
                q_dominates_p = True
            # 规则2：两者都可行，使用标准Pareto支配
            elif p_feas and q_feas:
                if (p_o0 <= q_o0 and p_o1 <= q_o1) and \
                        (p_o0 < q_o0 or p_o1 < q_o1):
                    p_dominates_q = True
                elif (q_o0 <= p_o0 and q_o1 <= p_o1) and \
                        (q_o0 < p_o0 or q_o1 < p_o1):
                    q_dominates_p = True
            # 规则3：两者都不可行，比较违规量和目标
            else:
                if p_viol < q_viol:
                    p_dominates_q = True
                elif p_viol > q_viol:
                    q_dominates_p = True
                else:
                    if (p_o0 <= q_o0 and p_o1 <= q_o1) and \
                            (p_o0 < q_o0 or p_o1 < q_o1):
                        p_dominates_q = True
                    elif (q_o0 <= p_o0 and q_o1 <= p_o1) and \
                            (q_o0 < p_o0 or q_o1 < p_o1):
                        q_dominates_p = True

            if p_dominates_q:
                dominated_sets[i].append(j)
            elif q_dominates_p:
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 0
            first_front_indices.append(i)

    # [优化] 第二阶段全部基于整数索引操作，避免对象属性查找
    fronts = [[population[i] for i in first_front_indices]]
    current_front_indices = first_front_indices

    while current_front_indices:
        next_front_indices = []
        for p_idx in current_front_indices:
            for q_idx in dominated_sets[p_idx]:
                domination_count[q_idx] -= 1
                if domination_count[q_idx] == 0:
                    population[q_idx].rank = len(fronts)
                    next_front_indices.append(q_idx)
        if next_front_indices:
            fronts.append([population[i] for i in next_front_indices])
        current_front_indices = next_front_indices

    return fronts


def crowding_distance_assignment(front, use_normalized=True):
    l = len(front)
    if l == 0: return
    for p in front: p.crowding_distance = 0

    # [新增 1] 预先计算当前 Front 在每个维度上的极值，用于全局归一化
    # 这一步至关重要，它定义了归一化的"标尺"
    # 使用 Python 原生列表推导式，不引入 numpy 以保持结构纯净
    min_objs = [min(ind.objectives[m] for ind in front) for m in range(2)]
    max_objs = [max(ind.objectives[m] for ind in front) for m in range(2)]

    for m in range(2):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = float('inf')
        front[l - 1].crowding_distance = float('inf')

        # [新增 2] 计算该维度的极差 (Range)
        obj_diff = max_objs[m] - min_objs[m]

        # 鲁棒性检查：防止除以零
        # 如果当前 Front 中所有个体在该目标上的值都相同 (diff=0)，
        # 说明该目标无法提供区分度，直接跳过累加
        if obj_diff == 0:
            continue

        for i in range(1, l - 1):
            # [修改 3] 执行归一化累加
            # 公式：(Next_Obj - Prev_Obj) / Range
            # 这样处理后，Cost (范围3000) 和 Sat (范围100) 的贡献都被拉到了
            # 同一个量级 (百分比)，确保了多目标分布的均匀性

            dist_contribution = front[i + 1].objectives[m] - front[i - 1].objectives[m]
            if use_normalized:
                dist_contribution /= obj_diff
            front[i].crowding_distance += dist_contribution
def crossover(p1, p2, instance):
    size = len(p1.permutation)
    s, e = sorted(random.sample(range(size), 2))

    # 1. 初始化子代路径
    c1_perm = [-1] * size
    c2_perm = [-1] * size

    # 2. 继承核心片段 (Inherit Core Segment)
    c1_perm[s:e] = p1.permutation[s:e]
    c2_perm[s:e] = p2.permutation[s:e]

    # 3. 填充剩余部分 (Fill Rest) - 标准OX交叉
    def fill_remaining(child_perm, fill_parent):
        curr = e
        # [优化] 用 set 替代每次迭代的切片+线性搜索，O(N²) -> O(N)
        inherited_set = set(child_perm[s:e])
        # 填充循环
        for i in range(size):
            cand = fill_parent.permutation[(e + i) % size]
            if cand not in inherited_set:
                child_perm[curr] = cand
                curr = (curr + 1) % size

    # 4. 填充两个子代
    fill_remaining(c1_perm, p2)
    fill_remaining(c2_perm, p1)

    # 5. 创建子代个体
    c1 = Individual(size)
    c1.permutation = c1_perm
    c1.preferred_weight = p1.preferred_weight  # 继承父代1的偏好权重

    c2 = Individual(size)
    c2.permutation = c2_perm
    c2.preferred_weight = p2.preferred_weight  # 继承父代2的偏好权重

    return c1, c2
def mutation(ind, instance, use_alns_mutation=True):
    """
    [修改说明] 简化为经典的序列变异算子
    原因：
    1. 只需对permutation进行变异
    2. 使用VRP领域成熟的变异算子组合
    3. Split算法会自动为变异后的序列找到最优分配
    """
    size = len(ind.permutation)
    r = random.choice([1, 2, 3, 4] if use_alns_mutation else [1, 2, 3])

    if r == 1:
        # Insertion Mutation (插入变异)
        # 随机选择一个节点，移动到另一个位置
        i, j = random.sample(range(size), 2)
        val_node = ind.permutation.pop(i)
        ind.permutation.insert(j, val_node)

    elif r == 2:
        # Swap Mutation (交换变异)
        # 随机交换两个节点的位置
        i, j = random.sample(range(size), 2)
        ind.permutation[i], ind.permutation[j] = ind.permutation[j], ind.permutation[i]

    elif r == 3:
        # Scramble Mutation (打乱变异)
        # 随机选择一段子序列，打乱其内部顺序
        i, j = sorted(random.sample(range(size), 2))
        segment = ind.permutation[i:j + 1]
        random.shuffle(segment)
        ind.permutation[i:j + 1] = segment

    elif r == 4:
        # ALNS: Adaptive Large Neighborhood Search
        # 智能破坏与重构算子

        # 1. 智能破坏：移除时间窗紧迫的节点
        num_remove = max(2, int(size ** 0.5))

        tightness = []
        for idx, node_id in enumerate(ind.permutation):
            start_time = instance.stw[node_id][0]
            tightness.append((idx, start_time))

        # 按时间窗开始时间排序（越早越紧迫）
        tightness.sort(key=lambda x: x[1])

        # 提取要移除的索引（从后往前删除避免索引偏移）
        indices_to_remove = sorted([t[0] for t in tightness[:num_remove]], reverse=True)

        removed_nodes = []
        for idx in indices_to_remove:
            n_id = ind.permutation.pop(idx)
            removed_nodes.append(n_id)

        # 2. 智能重构：最廉价插入（Cheapest Insertion）
        for node_to_insert in removed_nodes:
            best_pos = 0
            min_added_cost = float('inf')

            current_len = len(ind.permutation)
            for pos in range(current_len + 1):
                prev_node = ind.permutation[pos - 1] if pos > 0 else 0
                next_node = ind.permutation[pos] if pos < current_len else 0

                # 计算插入成本（使用卡车距离作为基准）
                cost = instance.get_truck_dist(prev_node, node_to_insert) + \
                       instance.get_truck_dist(node_to_insert, next_node) - \
                       instance.get_truck_dist(prev_node, next_node)

                if cost < min_added_cost:
                    min_added_cost = cost
                    best_pos = pos

            # 执行插入
            ind.permutation.insert(best_pos, node_to_insert)

def perturb_tour(base_tour, strength=2):
    """
    对基准路径施加受控扰动
    strength: 扰动次数，越大偏离越远
    """
    tour = list(base_tour)
    size = len(tour)
    for _ in range(strength):
        op = random.choice(['swap', 'insert', 'segment_shuffle'])
        if op == 'swap':
            i, j = random.sample(range(size), 2)
            tour[i], tour[j] = tour[j], tour[i]
        elif op == 'insert':
            i = random.randrange(size)
            val = tour.pop(i)
            j = random.randrange(size)
            tour.insert(j, val)
        elif op == 'segment_shuffle':
            i, j = sorted(random.sample(range(size), 2))
            seg_len = j - i + 1
            if seg_len > 5:  # 只打乱小片段，保留整体结构
                j = i + 4
            segment = tour[i:j+1]
            random.shuffle(segment)
            tour[i:j+1] = segment
    return tour

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
# ==========================================
# 5. 主程序入口
def run_nsga_solver(common_data, shared_ref_point=None, shared_ref_front=None, ablation=None):
    ablation = ablation or {}
    use_mols_decoder = ablation.get('use_mols_decoder', True)
    use_hybrid_init = ablation.get('use_hybrid_init', True)
    use_low_cost_probes = ablation.get('use_low_cost_probes', True)
    use_low_cost_weight_bias = ablation.get('use_low_cost_weight_bias', True)
    use_alns_mutation = ablation.get('use_alns_mutation', True)
    use_normalized_crowding_distance = ablation.get('use_normalized_crowding_distance', True)
    solver_name = ablation.get('name', 'NSGA-II')
    # 参数配置
    POP_SIZE = ablation.get('pop_size', 300)
    GEN_MAX = ablation.get('gen_max', 600)
    # 从数据中获取客户数量
    NUM_CUSTOMERS = len(common_data['nodes']) - 1
    global CACHE_LIMIT
    CACHE_LIMIT = POP_SIZE * GEN_MAX * 2
    # 1. 初始化并注入数据
    instance = ProblemInstance(common_data, num_customers=NUM_CUSTOMERS)
    data_loader.inject_data_to_nsga(instance, common_data)

    # 自适应剪枝容忍度：基于所有客户时间窗的平均宽度
    customer_nodes = [i for i in instance.nodes if i != 0]
    instance.mean_tw_width = sum(
        instance.stw[i][1] - instance.stw[i][0] for i in customer_nodes
    ) / len(customer_nodes)
    print(f"NSGA-II Data Overwritten with Solomon File. Variant: {solver_name}")
    t_solver_start = time.time()
    print("Initialize Population with First-Principles (Order & Chaos)...")
    population = []

    def evaluate_candidate(ind, progress=0.0):
        if use_mols_decoder:
            decode_and_evaluate(
                ind, instance, progress=progress,
                time_weight=ind.preferred_weight
            )
        else:
            ind.decode_weight = ind.preferred_weight
            import sol_pure_NSGA
            sol_pure_NSGA.greedy_split_decode(ind, instance)

    # --- 辅助函数：最近邻路径 (体现物理几何的最优性) ---
    def get_nn_tour(inst):
        unvisited = set(inst.nodes)
        unvisited.remove(0)  # 移除 Depot
        curr = 0
        tour = []
        while unvisited:
            # 贪婪选择：去离当前点最近的下一个点
            nxt = min(unvisited, key=lambda n: inst.get_truck_dist(curr, n))
            tour.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return tour

    def get_drone_aware_tour(inst, customers):
        """
        构造思路：将 D-P 配对相邻放置，穿插在 SPD 节点之间。
        使 Split 解码器在前瞻窗口 (max_lookahead=NUM_DRONES) 内
        更容易发现有效的双任务和单任务无人机派遣方案。
        """
        # 按客户类型分组 (类型编码与 check_dual_task_constraints 一致)
        d_nodes = [n for n in customers if inst.customer_types[n] == 2]  # Delivery
        p_nodes = [n for n in customers if inst.customer_types[n] == 1]  # Pickup
        spd_nodes = [n for n in customers if inst.customer_types[n] == 3]  # SPD (仅卡车)

        # 各组内部按距 Depot 的卡车距离排序，保持空间局部性
        d_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        p_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        spd_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))

        # 构造 D→P 配对队列 (满足 check_dual_task_constraints 的顺序要求)
        drone_pairs = []
        remaining_d = list(d_nodes)
        remaining_p = list(p_nodes)
        while remaining_d and remaining_p:
            drone_pairs.append((remaining_d.pop(0), remaining_p.pop(0)))
        unpaired = remaining_d + remaining_p  # 未配对的剩余节点

        # 交替插入：SPD 作为卡车锚点，D-P 对插入其间
        tour = []
        pi, si = 0, 0
        while pi < len(drone_pairs) or si < len(spd_nodes):
            # 先放一个 SPD 锚点 (卡车经停)
            if si < len(spd_nodes):
                tour.append(spd_nodes[si])
                si += 1
            # 再放一个 D-P 对 (落入下一段的 drone_candidates 窗口)
            if pi < len(drone_pairs):
                tour.append(drone_pairs[pi][0])  # D 在前
                tour.append(drone_pairs[pi][1])  # P 在后
                pi += 1
        # 未配对节点追加到末尾
        tour.extend(unpaired)
        return tour
    # 预计算物理基准路径 (Ground Truth Approximation)
    nn_tour = get_nn_tour(instance)

    # [新增] 预计算时间基准路径 (Time-Aware Approximation)
    all_customers = [n for n in instance.nodes if n != 0]
    time_sorted_tour = sorted(all_customers, key=lambda n: instance.stw[n][0])

    # 预计算无人机感知路径 (Drone-Aware Approximation)
    drone_aware_tour = get_drone_aware_tour(instance, all_customers)

    # 动态定义四种初始化策略的边界 (避免魔法数字)
    # 将种群分为四份: 空间优先 / 时间优先 / 无人机感知 / 随机
    limit_spatial = POP_SIZE // 4
    limit_temporal = POP_SIZE // 2
    limit_drone = (POP_SIZE * 3) // 4

    for i in range(POP_SIZE):
        # 创建个体
        ind = Individual(NUM_CUSTOMERS)

        # [策略 1: 空间有序] (Geometry / Cost Driven)
        # 每组第 1 个保留原始启发式路径，其余按组内位置递增扰动
        # 形成从"精确启发式"到"近随机"的梯度，兼顾质量与多样性
        if not use_hybrid_init:
            ind.permutation = list(np.random.permutation(NUM_CUSTOMERS) + 1)
        elif i < limit_spatial:
            position_in_group = i
            ind.permutation = perturb_tour(nn_tour, strength=position_in_group)

        # [策略 2: 时间有序] (Time / Feasibility Driven)
        elif i < limit_temporal:
            position_in_group = i - limit_spatial
            ind.permutation = perturb_tour(time_sorted_tour, strength=position_in_group)

        # [策略 3: 无人机感知] (Drone Coordination Driven)
        elif i < limit_drone:
            position_in_group = i - limit_temporal
            ind.permutation = perturb_tour(drone_aware_tour, strength=position_in_group)

        # [策略 4: 混沌] (Chaos / Diversity)
        # 完全随机打乱，防止算法陷入局部最优
        else:
            ind.permutation = list(np.random.permutation(NUM_CUSTOMERS) + 1)

        # ✅ 双重解码：随机选择解码模式
        # 使用个体自身的 preferred_weight 进行解码，保证评估一致性
        evaluate_candidate(ind, progress=0.0)
        population.append(ind)
        # print(ind.permutation)

    _init_dists = [p.objectives[0] for p in population]
    weight_upper = max(_init_dists) / (min(_init_dists) + 1e-10)
    # 校正初始种群的 preferred_weight 到自适应范围，并重新评估以保持一致性
    for idx, p in enumerate(population):
        if use_low_cost_weight_bias and idx % 5 == 0:
            p.preferred_weight = random.uniform(0.0, min(1.0, weight_upper))
        else:
            p.preferred_weight = random.uniform(0.0, weight_upper)
        evaluate_candidate(p, progress=0.0)
    pareto_archive = []
    print("Start Evolution...")
    # 交叉概率：由问题规模推导，N越大越接近1.0
    pc = (NUM_CUSTOMERS - 1.0) / NUM_CUSTOMERS

    for gen in range(GEN_MAX):
        current_progress = gen / float(GEN_MAX)
        offspring = []

        # 自适应变异概率：初期=1.0（强探索），末期=1/N（弱扰动）
        pm = 1.0 / NUM_CUSTOMERS + (1.0 - 1.0 / NUM_CUSTOMERS) * (1.0 - current_progress)

        # 常规 NSGA-II 进化流程
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            # 交叉概率控制
            if random.random() < pc:
                c1, c2 = crossover(p1, p2, instance)
            else:
                # 不交叉时，子代直接继承父代基因
                c1 = Individual(NUM_CUSTOMERS)
                c1.permutation = list(p1.permutation)
                c1.preferred_weight = p1.preferred_weight
                c2 = Individual(NUM_CUSTOMERS)
                c2.permutation = list(p2.permutation)
                c2.preferred_weight = p2.preferred_weight

            # 变异概率控制
            if random.random() < pm:
                mutation(c1, instance, use_alns_mutation=use_alns_mutation)
            if random.random() < pm:
                mutation(c2, instance, use_alns_mutation=use_alns_mutation)

            # 使用子代自身的 preferred_weight 进行解码，保证评估一致性
            # c1.preferred_weight 继承自 p1（交叉第 596 行），c2 继承自 p2（第 600 行）
            evaluate_candidate(c1, progress=current_progress)
            evaluate_candidate(c2, progress=current_progress)
            offspring.extend([c1, c2])

        if use_low_cost_probes and gen % 20 == 0:
            for seed_tour in [nn_tour, drone_aware_tour]:
                probe = Individual(NUM_CUSTOMERS)
                probe.permutation = perturb_tour(seed_tour, strength=random.randint(1, 3))
                probe.preferred_weight = random.uniform(0.0, 0.5)
                evaluate_candidate(probe, progress=current_progress)
                offspring.append(probe)

        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)

        # Refine promising candidates before environmental selection. Running this
        # for every decoded individual is too slow, but delaying it until after
        # selection changes the search pressure and can discard candidates that
        # become strong only after route-level refinement.
        if False and fronts:
            crowding_distance_assignment(
                fronts[0],
                use_normalized=use_normalized_crowding_distance,
            )
            fronts[0].sort(key=lambda x: x.crowding_distance, reverse=True)
            refine_budget = max(8, int(len(population) ** 0.5))
            for elite in fronts[0][:refine_budget]:
                saved_permutation = list(elite.permutation)
                saved_objectives = list(elite.objectives)
                saved_violation = elite.constraint_violation
                saved_feasible = elite.is_feasible
                saved_schedule = elite.decoded_schedule

                pass

                old_obj = saved_objectives
                new_obj = elite.objectives
                rollback = False
                if saved_feasible and not elite.is_feasible:
                    rollback = True
                elif saved_feasible and elite.is_feasible:
                    old_dominates = (old_obj[0] <= new_obj[0] and old_obj[1] <= new_obj[1]) and \
                                    (old_obj[0] < new_obj[0] or old_obj[1] < new_obj[1])
                    if old_dominates:
                        rollback = True

                if rollback:
                    elite.permutation = saved_permutation
                    elite.objectives = saved_objectives
                    elite.constraint_violation = saved_violation
                    elite.is_feasible = saved_feasible
                    elite.decoded_schedule = saved_schedule

            fronts = fast_non_dominated_sort(combined)

        new_pop = []
        for front in fronts:
            crowding_distance_assignment(
                front,
                use_normalized=use_normalized_crowding_distance,
            )
            if len(new_pop) + len(front) <= POP_SIZE:
                new_pop.extend(front)
            else:
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_pop.extend(front[:POP_SIZE - len(new_pop)])
                break
        population = new_pop

        # [新增] 模因算法：对 Pareto 前沿精英进行局部搜索强化
        # 1. 获取所有 Rank 0
        all_rank0 = [p for p in population if p.rank == 0]

        # 2. 按拥挤距离降序排列 (优先优化稀疏区域的解，扩展前沿)
        all_rank0.sort(key=lambda x: x.crowding_distance, reverse=True)

        # 3. 动态计算预算：取种群规模的平方根
        # ls_budget = max(1, int(len(population) ** 0.5 * (1.0 - current_progress)))
        ls_budget = max(1, int(len(population) ** 0.5 * 0.3))
        elite_individuals = all_rank0[:ls_budget]

        for elite in elite_individuals:
            # 保存局部搜索前的完整状态快照
            saved_permutation = list(elite.permutation)
            saved_objectives = list(elite.objectives)
            saved_violation = elite.constraint_violation
            saved_feasible = elite.is_feasible
            saved_schedule = elite.decoded_schedule

            if False:
                pass

            # 回滚保护门：如果局部搜索后旧解支配新解，则回退
            # 这确保精英个体在局部搜索后不会变得更差
            new_obj = elite.objectives
            old_obj = saved_objectives

            rollback = False

            if saved_feasible and not elite.is_feasible:
                # 从可行退化为不可行：必须回滚
                rollback = True
            elif saved_feasible and elite.is_feasible:
                # 都可行：检查旧解是否 Pareto 支配新解
                # 支配定义与 fast_non_dominated_sort 中规则 2（第 483-489 行）一致
                old_dominates = (old_obj[0] <= new_obj[0] and old_obj[1] <= new_obj[1]) and \
                                (old_obj[0] < new_obj[0] or old_obj[1] < new_obj[1])
                if old_dominates:
                    rollback = True

            if rollback:
                elite.permutation = saved_permutation
                elite.objectives = saved_objectives
                elite.constraint_violation = saved_violation
                elite.is_feasible = saved_feasible
                elite.decoded_schedule = saved_schedule
        for p in population:
            if not p.is_feasible or p.rank != 0:
                continue
            new_obj = (p.objectives[0], p.objectives[1])

            is_dominated = False
            to_remove = []
            for idx, arch_entry in enumerate(pareto_archive):
                arch_obj = arch_entry['objectives']
                # 存档解支配新解？
                if (arch_obj[0] <= new_obj[0] and arch_obj[1] <= new_obj[1]) and \
                        (arch_obj[0] < new_obj[0] or arch_obj[1] < new_obj[1]):
                    is_dominated = True
                    break
                # 新解支配存档解？
                if (new_obj[0] <= arch_obj[0] and new_obj[1] <= arch_obj[1]) and \
                        (new_obj[0] < arch_obj[0] or new_obj[1] < arch_obj[1]):
                    to_remove.append(idx)

            if not is_dominated:
                for idx in sorted(to_remove, reverse=True):
                    pareto_archive.pop(idx)
                # 去重：检查是否已存在完全相同的目标值
                if not any(e['objectives'] == new_obj for e in pareto_archive):
                    truck_route, drone_sorties = p.decoded_schedule
                    pareto_archive.append({
                        'objectives': new_obj,
                        'decoded_schedule': (
                            list(truck_route),
                            [(l, list(pn), r) for (l, pn, r) in drone_sorties]
                        )
                    })

        if gen % 10 == 0:
            feasible_pop = [p for p in population if p.is_feasible]
            if feasible_pop:
                # 当代种群中的最优（用于对比观察）
                gen_best_dist = min(p.objectives[0] for p in feasible_pop)
                gen_max_sat = -min(p.objectives[1] for p in feasible_pop)

                # 从全局存档中提取历史最优（单目标维度的极值，非同一个解）
                if pareto_archive:
                    arch_best_dist = min(e['objectives'][0] for e in pareto_archive)
                    arch_max_sat = -min(e['objectives'][1] for e in pareto_archive)
                else:
                    arch_best_dist = gen_best_dist
                    arch_max_sat = gen_max_sat

                print(f"Gen {gen}: Gen Dist={gen_best_dist:.2f}, Gen Sat={gen_max_sat:.2f} | "
                      f"Archive Best Dist={arch_best_dist:.2f}, Archive Best Sat={arch_max_sat:.2f}")
            else:
                # 如果种群中全是不可行解（早期可能出现），则回退到打印所有
                raw_dist = min(p.objectives[0] for p in population)
                print(f"Gen {gen}: No feasible sol yet. Raw Dist={raw_dist:.2f}")

    # 结果可视化
    t_solver_end = time.time()
    solver_runtime = t_solver_end - t_solver_start
    print("\nEvolution Finished.")
    pareto_front = fast_non_dominated_sort(population)[0]
    if not pareto_archive:
        pareto_archive = [{
            'objectives': tuple(p.objectives),
            'decoded_schedule': p.decoded_schedule
        } for p in pareto_front if p.is_feasible]
    if not pareto_archive:
        pareto_archive = [{
            'objectives': tuple(p.objectives),
            'decoded_schedule': p.decoded_schedule
        } for p in pareto_front]

    # 从全局存档中提取两个极端解
    # 1. 找最小成本解 (Min Cost) -> objectives[0] 最小
    arch_min_cost = min(pareto_archive, key=lambda e: e['objectives'][0])

    # 2. 找最大满意度解 (Max Sat) -> objectives[1] 最小 (因为是负值)
    arch_max_sat = min(pareto_archive, key=lambda e: e['objectives'][1])

    print("\n[Phase 1] Determining Bounds (Extracted from Global Pareto Archive)...")
    print("  -> Max Satisfaction Solution:")
    print(f"     Max Sat: {-arch_max_sat['objectives'][1]:.2f} (Cost: {arch_max_sat['objectives'][0]:.2f})")

    ms_truck_route, ms_drone_sorties = arch_max_sat['decoded_schedule']
    print(f"     Truck Route: {ms_truck_route}")
    print(f"     Drone Sorties: {ms_drone_sorties}")

    print("  -> Min Cost Solution:")
    print(f"     Min Cost: {arch_min_cost['objectives'][0]:.2f} (Sat: {-arch_min_cost['objectives'][1]:.2f})")

    mc_truck_route, mc_drone_sorties = arch_min_cost['decoded_schedule']
    print(f"     Truck Route: {mc_truck_route}")
    print(f"     Drone Sorties: {mc_drone_sorties}")

    # 为了调用 print_detailed_schedule，需要构造一个临时 Individual 对象
    # 因为 print_detailed_schedule 要求参数是 Individual（访问 .decoded_schedule 和 .constraint_violation）
    arch_min_cost_ind = Individual(NUM_CUSTOMERS)
    arch_min_cost_ind.decoded_schedule = arch_min_cost['decoded_schedule']
    arch_min_cost_ind.objectives = list(arch_min_cost['objectives'])
    arch_min_cost_ind.constraint_violation = 0.0  # 存档中只保存可行解，违规量为0

    arch_max_sat_ind = Individual(NUM_CUSTOMERS)
    arch_max_sat_ind.decoded_schedule = arch_max_sat['decoded_schedule']
    arch_max_sat_ind.objectives = list(arch_max_sat['objectives'])
    arch_max_sat_ind.constraint_violation = 0.0  # 存档中只保存可行解，违规量为0

    print("\nPrinting detailed schedule for Min Cost Solution:")
    print_detailed_schedule(instance, arch_min_cost_ind)

    print("\nPrinting detailed schedule for Max Sat Solution:")
    print_detailed_schedule(instance, arch_max_sat_ind)

    dists = [p.objectives[0] for p in pareto_front]
    sats = [-p.objectives[1] for p in pareto_front]

    # plt.figure(figsize=(10, 6))
    # plt.scatter(dists, sats, c='red', s=50, label='Pareto Solutions')
    # plt.xlabel('Total Distance (Minimize)')
    # plt.ylabel('Worst-Case Satisfaction (Maximize)')
    # plt.title(f'NSGA-II for Risk-Aware TDRP (N={NUM_CUSTOMERS})')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # 打印一个优选解
    best_sol = pareto_front[0]
    print("\n--- Sample Pareto Solution ---")
    print(f"Distance: {best_sol.objectives[0]:.2f}")
    print(f"Satisfaction: {-best_sol.objectives[1]:.2f}")
    t_route, d_tasks = best_sol.decoded_schedule
    # print(f"Truck Route: {t_route}")
    # print(f"Drone Sorties (Launch, Serve, Recover): {d_tasks}")
    #
    # --- 新增调用绘图函数 ---
    # print("Plotting best solution...")
    # plot_solution(instance, t_route, d_tasks, obj_vals=best_sol.objectives)

    # --- [修改开始] 指标计算 ---
    pareto_front_pop = fast_non_dominated_sort(population)[0]

    # 1. 从全局存档中提取目标值 (这是最全的)
    archive_objs = [list(entry['objectives']) for entry in pareto_archive]

    # 2. 从当前种群提取目标值 (防止最后一代有新发现未入库)
    pop_objs = [list(ind.objectives) for ind in pareto_front_pop]

    # 3. 合并并去重
    all_objs = archive_objs + pop_objs
    nsga_front_objs = []
    seen_objs = set()
    for obj in all_objs:
        # 转为 tuple 以便哈希去重
        t_obj = (float(obj[0]), float(obj[1]))
        if t_obj not in seen_objs:
            seen_objs.add(t_obj)
            nsga_front_objs.append(obj)

    # 2. 动态定义参考点
    if shared_ref_point is not None:
        # [修改] 如果提供了共享参考点（来自 Gurobi），则使用它以保证 HV 可比性
        print(f"\n[Metric] Using Shared Reference Point from Exact Solver: {shared_ref_point}")
        ref_point_nsga = shared_ref_point
    else:
        # [修改] 仅在未提供共享参考点时，才使用自身种群的最差值作为参考点
        max_cost_obs = max(ind.objectives[0] for ind in pareto_front_pop)
        max_neg_sat_obs = max(ind.objectives[1] for ind in pareto_front_pop)
        ref_point_nsga = [max_cost_obs, max_neg_sat_obs]
        print(f"\n[Metric] Using Local Reference Point (Self-Adaptive): {ref_point_nsga}")

    # 3. 计算指标
    # 3. 计算指标
    hv_val, igd_val, sp_val = calculate_metrics(nsga_front_objs, ref_point_nsga, shared_ref_front)

    igd_str = f"{igd_val:.6f}" if igd_val is not None else "N/A (需提供shared_ref_front)"

    print("\n" + "=" * 40)
    print("Performance Metrics (NSGA-II)")
    print("=" * 40)
    print(f"Reference Point        : {ref_point_nsga}")
    print(f"HV  (Hypervolume)      : {hv_val:.4f}")
    print(f"IGD (Inv. Gen. Dist.)  : {igd_str}")
    sp_str = f"{sp_val:.6f}" if sp_val is not None else "N/A (degenerate front)"
    print(f"SP  (Spacing)          : {sp_str}")
    print(f"Runtime (s)            : {solver_runtime:.2f}")
    print("=" * 40 + "\n")

    # 返回 Pareto 前沿的最优解 (例如距离最短的)
    best_dist_sol = min(pareto_front_pop, key=lambda p: p.objectives[0])
    return best_dist_sol.objectives[0], best_dist_sol.objectives[1], nsga_front_objs

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


def fast_timing_and_satisfaction(instance, truck_route, drone_sorties):
    # 1. 数据准备：编码 Drone Sorties 为 Numpy 数组
    # 格式: [launch, n1, n2, recover]。如果 n2 不存在则为 -1
    n_sorties = len(drone_sorties)
    # 必须显式指定 dtype=np.int32 以匹配 JIT 签名
    if n_sorties == 0:
        encoded_sorties = np.empty((0, 4), dtype=np.int32)
    else:
        encoded_sorties = np.full((n_sorties, 4), -1, dtype=np.int32)
        for i, (l, path, r) in enumerate(drone_sorties):
            encoded_sorties[i, 0] = l
            # path 是列表，取出节点
            if len(path) > 0:
                encoded_sorties[i, 1] = path[0]
            if len(path) > 1:
                encoded_sorties[i, 2] = path[1]
            encoded_sorties[i, 3] = r

    # 2. 转换 Truck Route 为 array
    route_arr = np.array(truck_route, dtype=np.int32)

    # 3. 准备时间窗数组和 dtw_width 数组 (防御性编程：处理 dict/list 混合情况)
    num_nodes = instance.num_nodes

    # 预分配数组
    stw_start = np.zeros(num_nodes, dtype=np.float64)
    stw_end = np.zeros(num_nodes, dtype=np.float64)
    dtw_width_arr = np.zeros(num_nodes, dtype=np.float64)

    # 检测 instance.stw 的类型来填充数据
    # 如果是 list (通常情况)
    for i in range(num_nodes):
        # 处理 stw
        if isinstance(instance.stw, dict):
            w = instance.stw.get(i, (0.0, 100000.0))  # 默认宽时间窗防止报错
        else:
            w = instance.stw[i]
        stw_start[i] = w[0]
        stw_end[i] = w[1]

        # 处理 dtw_width (关键修复：如果是dict，转为array)
        if isinstance(instance.dtw_width, dict):
            dtw_width_arr[i] = instance.dtw_width.get(i, 0.0)
        else:
            dtw_width_arr[i] = instance.dtw_width[i]

    # 4. 确保距离矩阵也是 numpy array (防止 data_loader 修改)
    d_mat_truck = instance.dist_matrix_truck
    d_mat_drone = instance.dist_matrix_drone
    # 如果意外变成 dict (极少见但为了保险)，这里可以加类似转换，但在 ProblemInstance 中应该已是 array

    # 5. 调用 JIT
    total_sat, srv_times, arr_times, viol_time, valid = fast_timing_core_jit(
        num_nodes,
        route_arr,
        encoded_sorties,
        stw_start,
        stw_end,
        dtw_width_arr,  # <--- 这里传入的现在确信是 Numpy Array
        d_mat_truck,
        d_mat_drone,
        instance.v_truck,
        instance.v_drone
    )

    if not valid:
        # 0.0 满意度, 空时间字典, 极大违规
        return 0.0, {}, {}, float('inf')

    # 将 numpy array 转回 dict 格式的时间表，以保持接口兼容性（后续打印函数可能需要 dict）
    # 但 fast_timing_core_jit 返回的是 array，如果外部只需要数值，可以直接用。
    # 为了保持原函数返回类型签名 (total_sat, service_time(dict/arr), ...)，这里保持 array 即可
    # 因为 python 中 arr[i] 和 dict[i] 语法一样，通常兼容。

    return total_sat, srv_times, arr_times, viol_time

# ==========================================
# 6. 新增绘图工具函数 (修正版: 包含严格的 ID 时间分配逻辑)
# ==========================================
def plot_solution(instance, truck_route, drone_tasks, obj_vals=None):
    """
    可视化卡车和无人机的路径方案 (修正版)
    核心改进：
    1. 引入严格的时间推演，计算每个任务的精确 Start/End 时间。
    2. 使用贪心策略根据时间分配 Drone ID，确保颜色连续且复用正确。
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 10))

    # --- 步骤 A: 预计算所有任务的时间窗口 ---
    # 我们需要知道每个 sortie 的 (launch_time, land_time) 才能分配 ID

    # 1. 建立数据结构
    # drone_tasks 格式: list of (launch_node, path_nodes, recover_node)
    # 我们给每个任务一个原始索引，以便后续追踪
    task_info_map = []  # 存 {'idx': i, 'l': l, 'path': p, 'r': r}
    for i, (l, p, r) in enumerate(drone_tasks):
        task_info_map.append({'original_idx': i, 'l': l, 'path': p, 'r': r, 'start_t': -1, 'end_t': -1})

    # 辅助查找：launch_node -> tasks starting here
    launch_map = {node: [] for node in instance.nodes}
    for item in task_info_map:
        launch_map[item['l']].append(item)

    # 辅助查找：recover_node -> tasks ending here (用于卡车等待)
    pending_recoveries = {node: [] for node in instance.nodes}

    # 2. 模拟卡车运行以计算时间 (与解码器逻辑一致)
    departure_times = {0: 0.0}
    arrival_times = {}

    # 当前卡车到达某点的时间
    curr_time = 0.0

    for idx, curr_node in enumerate(truck_route):
        # 1. 卡车到达
        if idx == 0:
            arrival_times[curr_node] = 0.0
        else:
            prev_node = truck_route[idx - 1]
            travel_t = instance.get_truck_dist(prev_node, curr_node) / instance.v_truck
            arrival_times[curr_node] = departure_times[prev_node] + travel_t

        truck_arr = arrival_times[curr_node]

        # 2. 等待回收 (Wait for Drones)
        # 必须等待所有计划在此处回收的无人机落地
        max_drone_arr = truck_arr
        if curr_node in pending_recoveries:
            for rec_item in pending_recoveries[curr_node]:
                # 这里的 rec_item 包含该任务最后一跳的飞行时间和倒数第二个点的服务结束时间
                # 我们需要在发射时就计算好这些
                d_arr = rec_item['land_time']
                if d_arr > max_drone_arr:
                    max_drone_arr = d_arr

                # 回写结束时间到 task_info
                rec_item['task_ref']['end_t'] = d_arr

        node_ready_time = max_drone_arr

        # 3. 卡车服务与发射 (Service & Launch)
        # 发射时间 = max(到达时间, 回收完毕时间, 节点最早服务时间)
        # 注意：如果是 Depot，不需要服务时间，但作为中间点可能有限制
        srv_start = node_ready_time
        if curr_node != 0:
            stw_start = instance.stw[curr_node][0]
            srv_start = max(node_ready_time, stw_start)
            departure_times[curr_node] = srv_start
        else:
            # Depot 作为终点或起点
            departure_times[curr_node] = node_ready_time

            # 处理在此处发射的任务
        if curr_node in launch_map:
            launch_base_time = node_ready_time  # 无人机可以在卡车到达/回收完成后立即起飞

            for task in launch_map[curr_node]:
                task['start_t'] = launch_base_time

                # 推演该任务的飞行与服务过程，为了计算 land_time
                curr_aerial_t = launch_base_time
                last_srv_end = 0

                # 遍历无人机路径点
                for step_i, d_node in enumerate(task['path']):
                    prev_loc = curr_node if step_i == 0 else task['path'][step_i - 1]
                    fly_t = instance.get_drone_dist(prev_loc, d_node) / instance.v_drone
                    arr_t = curr_aerial_t + fly_t

                    s_start = max(arr_t, instance.stw[d_node][0])
                    s_end = s_start

                    curr_aerial_t = s_end
                    last_srv_end = s_end

                # 计算最后一段回程
                last_node = task['path'][-1]
                r_node = task['r']
                fly_in = instance.get_drone_dist(last_node, r_node) / instance.v_drone
                land_t = last_srv_end + fly_in

                # 注册到回收点，以便卡车计算等待
                if r_node not in pending_recoveries: pending_recoveries[r_node] = []
                pending_recoveries[r_node].append({
                    'land_time': land_t,
                    'task_ref': task
                })

    # --- 步骤 B: 贪心分配 Drone ID (关键修正) ---
    # 1. 按发射时间排序
    sorted_tasks = sorted(task_info_map, key=lambda x: x['start_t'])

    # 2. 初始化无人机状态 [free_time_drone_0, free_time_drone_1, ...]
    drone_free_times = [0.0] * instance.NUM_DRONES
    task_id_assignment = {}  # original_idx -> assigned_drone_id

    for task in sorted_tasks:
        s_t = task['start_t']
        e_t = task['end_t']

        assigned_id = -1
        # 优先找空闲的
        for d_id in range(instance.NUM_DRONES):
            # 容差 1e-4 避免浮点误差
            if drone_free_times[d_id] <= s_t + 1e-4:
                assigned_id = d_id
                drone_free_times[d_id] = e_t
                break

        # 如果都忙(理论上不应发生，如果通过了decode的可行性检查)，则选最早结束的那个强行复用
        # (这只是为了画图不报错，实际代表解可能有瑕疵，但在修正逻辑后应也是合法的)
        if assigned_id == -1:
            best_id = np.argmin(drone_free_times)
            assigned_id = best_id
            drone_free_times[best_id] = e_t

        task_id_assignment[task['original_idx']] = assigned_id

    # --- 步骤 C: 开始绘图 ---

    # 定义样式
    drone_styles = {
        0: {'color': 'tab:red', 'ls': '--', 'label': 'Drone #1'},
        1: {'color': 'tab:green', 'ls': '-.', 'label': 'Drone #2'},
        2: {'color': 'tab:blue', 'ls': ':'},
    }

    legend_added = set()

    # 1. 绘制无人机
    for item in task_info_map:
        d_id = task_id_assignment[item['original_idx']]
        # 只有前2架无人机有固定样式，超出的(如果出错)用默认
        style = drone_styles.get(d_id, {'color': 'black', 'ls': '-', 'label': f'Drone #{d_id + 1}'})

        full_path = [item['l']] + item['path'] + [item['r']]
        coords = np.array([instance.coords[n] for n in full_path])

        # Line
        lbl = style['label'] if d_id not in legend_added else ""
        plt.plot(coords[:, 0], coords[:, 1], color=style['color'], linestyle=style['ls'],
                 linewidth=2, alpha=0.8, zorder=2, label=lbl)
        legend_added.add(d_id)

        # Arrow
        for k in range(len(coords) - 1):
            p1, p2 = coords[k], coords[k + 1]
            plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.5, (p2[1] - p1[1]) * 0.5,
                      head_width=0.3, color=style['color'], length_includes_head=True, zorder=3)

    # 2. 绘制卡车
    truck_coords = np.array([instance.coords[n] for n in truck_route])
    plt.plot(truck_coords[:, 0], truck_coords[:, 1], color='black', linewidth=3,
             alpha=0.6, label='Truck', zorder=1)
    # Truck Arrows
    for k in range(len(truck_coords) - 1):
        p1, p2 = truck_coords[k], truck_coords[k + 1]
        plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.55, (p2[1] - p1[1]) * 0.55,
                  head_width=0.3, color='black', alpha=0.3, zorder=1)

    # 3. 绘制节点
    for n in instance.nodes:
        c = instance.coords[n]
        if n == 0:
            plt.scatter(c[0], c[1], c='black', marker='s', s=200, zorder=10, label='Depot')
            plt.text(c[0], c[1], "0", color='white', ha='center', va='center', fontweight='bold')
        else:
            ctype = instance.customer_types[n]
            # 1:P(Orange), 2:D(Blue), 3:SPD(Purple)
            col = 'orange' if ctype == 1 else ('dodgerblue' if ctype == 2 else 'purple')
            mark = '^' if ctype == 1 else ('o' if ctype == 2 else 'D')

            plt.scatter(c[0], c[1], c=col, marker=mark, s=120, edgecolors='k', zorder=5)
            plt.text(c[0], c[1], str(n), color='white', ha='center', va='center', fontweight='bold', fontsize=9)

    # 4. 装饰
    title = "Improved NSGA-II Solution Visualization"
    if obj_vals: title += f"\nDist: {obj_vals[0]:.2f} | Sat: {-obj_vals[1]:.2f}"
    plt.title(title)

    # 构造完整图例
    handles, labels = plt.gca().get_legend_handles_labels()
    # 去重
    by_label = dict(zip(labels, handles))
    # 添加客户类型图例
    by_label['Type D'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10)
    by_label['Type P'] = Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10)

    plt.legend(by_label.values(), by_label.keys(), loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_detailed_schedule(instance, individual):
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

        # [修正] 兼容 NumPy 数组访问
        t_service = service_times[node]
        if t_service < -0.5:  # 相当于 .get(default)
            t_service = instance.stw[node][0]

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

    # 1. Depot Start
    depot_start = service_times[0]
    final_data.append({
        'id': 0, 'type': 'Depot', 'arr': depot_start, 'srv': depot_start,
        'sat': '-', 'tw_start': instance.stw[0][0], 'tw_end': instance.stw[0][1]
    })

    # 2. Truck
    for k in range(len(truck_route) - 1):
        u = truck_route[k]
        v = truck_route[k + 1]
        if v == 0: continue

        # [修正] 数组访问
        srv_v = service_times[v]
        arr_v = arrival_times[v]
        if srv_v < -0.5: srv_v = instance.stw[v][0]
        if arr_v < -0.5: arr_v = srv_v

        sat_v = node_satisfaction.get(v, 0.0)

        final_data.append({
            'id': v, 'type': 'Truck', 'arr': arr_v, 'srv': srv_v,
            'sat': sat_v, 'tw_start': instance.stw[v][0], 'tw_end': instance.stw[v][1]
        })

    # 3. Drone
    for (launch_node, path_nodes, recover_node) in drone_sorties:
        # Launch time (Arrival at launch node)
        curr_time = arrival_times[launch_node]
        if curr_time < -0.5: curr_time = instance.stw[launch_node][0]

        prev_node = launch_node

        for d_node in path_nodes:
            fly_t = instance.get_drone_dist(prev_node, d_node) / instance.v_drone
            arr_d = curr_time + fly_t

            # [修正] 数组访问
            srv_d = service_times[d_node]
            if srv_d < -0.5: srv_d = instance.stw[d_node][0]

            sat_d = node_satisfaction.get(d_node, 0.0)

            final_data.append({
                'id': d_node, 'type': 'Drone', 'arr': arr_d, 'srv': srv_d,
                'sat': sat_d, 'tw_start': instance.stw[d_node][0], 'tw_end': instance.stw[d_node][1]
            })

            curr_time = srv_d
            prev_node = d_node

    # 4. Depot End
    if truck_route[-1] == 0:
        last_customer = truck_route[-2]
        travel_t = instance.get_truck_dist(last_customer, 0) / instance.v_truck

        srv_last = service_times[last_customer]
        if srv_last < -0.5: srv_last = instance.stw[last_customer][0]

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
        sat_str = f"{item['sat']:.4f}" if isinstance(item['sat'], float) else item['sat']

        if item['arr'] > item['tw_end'] + 1e-3:
            status = "晚到"
        elif item['srv'] > item['tw_end'] + 1e-3:
            status = "超时"
        else:
            status = "正常"

        print(f"{item['id']:<6} {item['type']:<12} {tw_str:<18} "
              f"{item['arr']:<10.2f} {item['srv']:<10.2f} {sat_str:<8} {status:<8}")
    print("-" * 82)

if __name__ == "__main__":
    common_data = data_loader.load_solomon_data('solomon标准算例-时间窗/c1/c101.txt', n_customers=60,
                                                random_seed=42)
    print(common_data)
    run_nsga_solver(common_data)
