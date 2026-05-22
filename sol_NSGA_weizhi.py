import numpy as np
import random
import matplotlib.pyplot as plt
import data_loader
import time
import os
from numba import jit

DTW_MODES = {
    'centered': {'beta_a': 4, 'beta_b': 4, 'label': 'Centered Preference'},
    'early': {'beta_a': 2, 'beta_b': 5, 'label': 'Early Preference'},
    'late': {'beta_a': 5, 'beta_b': 2, 'label': 'Late Preference'},
    'uniform': {'beta_a': 1, 'beta_b': 1, 'label': 'Uniform/Random Preference'},
}

DTW_NUM_SAMPLES = 100


def compute_satisfaction_by_mode(t_service, tw_start, tw_end, dtw_width, a_samples_scaled):
    delta = (tw_end - tw_start) - dtw_width
    if delta < 1e-5:
        return 1.0

    a = a_samples_scaled
    w = dtw_width
    t = t_service
    s = tw_start
    e = tw_end

    sat = np.zeros_like(a)
    inside = (a <= t) & (t <= a + w)
    sat[inside] = 1.0

    left_mask = (t < a) & (t >= s)
    left_len = a[left_mask] - s
    safe_left = np.where(left_len > 1e-8, left_len, 1.0)
    sat[left_mask] = np.clip((t - s) / safe_left, 0.0, 1.0)

    right_mask = (t > a + w) & (t <= e)
    right_len = e - (a[right_mask] + w)
    safe_right = np.where(right_len > 1e-8, right_len, 1.0)
    sat[right_mask] = np.clip((e - t) / safe_right, 0.0, 1.0)

    return float(np.mean(sat))


def get_optimal_target_by_mode(tw_start, tw_end, dtw_width, dtw_mode):
    delta = (tw_end - tw_start) - dtw_width
    if delta < 1e-5:
        return (tw_start + tw_end) / 2.0

    cfg = DTW_MODES[dtw_mode]
    expected_ratio = cfg['beta_a'] / (cfg['beta_a'] + cfg['beta_b'])
    expected_a = tw_start + delta * expected_ratio
    return expected_a + dtw_width / 2.0


@jit(nopython=True)
def calc_robust_satisfaction_jit(t_service, tw_start, tw_end, dtw_width):
    # 閫昏緫瀹屽叏瀵瑰簲鍘熶唬鐮侊紝浣嗛缂栬瘧涓烘満鍣ㄧ爜
    delta_i = (tw_end - tw_start) - dtw_width

    if delta_i < 1e-5:
        return 1.0

    # 瀵瑰簲璁烘枃 Paper 4 鐨勯瞾妫掓弧鎰忓害鍏紡
    sat_leftmost = (tw_end - t_service) / delta_i
    sat_rightmost = (t_service - tw_start) / delta_i

    # 鎵嬪姩瀹炵幇 max/min 浠ョ‘淇濈被鍨嬬ǔ瀹?
    alpha = min(1.0, sat_leftmost)
    alpha = min(alpha, sat_rightmost)
    alpha = max(0.0, alpha)

    return alpha


@jit(nopython=True)
def calc_dist_matrix_jit(coords):
    n = len(coords)
    # 鍒濆鍖栫煩闃?
    mat_truck = np.zeros((n, n), dtype=np.float64)  # 鏄惧紡鎸囧畾 float64
    mat_drone = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            c1 = coords[i]
            c2 = coords[j]
            # 鏇煎搱椤胯窛绂?
            mat_truck[i, j] = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            # 娆у嚑閲屽緱璺濈
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            mat_drone[i, j] = (dx * dx + dy * dy) ** 0.5

    return mat_truck, mat_drone


# --- [鏂板] Numba 鍔犻€熺殑鏍稿績璁＄畻鍑芥暟 ---

@jit(nopython=True)
def check_dominance_jit(existing_objs, new_dist, new_time):
    """
    妫€鏌?(new_dist, new_time) 鏄惁琚?existing_objs 涓殑浠绘剰瑙ｆ敮閰嶃€?
    existing_objs: (N, 2) 鐨?float64 鏁扮粍
    杩斿洖: True (琚敮閰? 涓㈠純), False (涓嶈鏀厤)
    """
    n = existing_objs.shape[0]
    eps = 1e-5
    for i in range(n):
        e_dist = existing_objs[i, 0]
        e_time = existing_objs[i, 1]

        # 鐜版湁瑙?e 鏀厤 鏂拌В new ?
        # 鏉′欢: e_dist <= new_dist AND e_time <= new_time 涓旇嚦灏戞湁涓€涓弗鏍煎皬浜?
        if e_dist <= new_dist + eps and e_time <= new_time + eps:
            if e_dist < new_dist - eps or e_time < new_time - eps:
                return True
    return False


@jit(nopython=True)
def fast_timing_core_jit(num_nodes, truck_route, sorties_encoded,
                         stw_start, stw_end, dtw_width,
                         target_service,
                         dist_mat_truck, dist_mat_drone,
                         v_truck, v_drone):
    """
    [淇鐗圿 鍖呭惈涓ユ牸鐨勯瞾妫掓湇鍔℃椂闂磋绠楅€昏緫锛屼慨澶嶆弧鎰忓害璁＄畻閿欒銆?
    """
    # 1. 棰勫垎閰嶅唴瀛?
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

    # (A) 鏋勫缓鍗¤溅杈?
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

    # (B) 鏋勫缓鏃犱汉鏈鸿竟
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

    # (C) 鎷撴墤鎺掑簭涓庢椂闂翠紶鎾?
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
                # --- [鍏抽敭淇寮€濮媇 ---
                arr_v = arrival_time[v]
                tw_s = stw_start[v]
                tw_e = stw_end[v]
                w_i = dtw_width[v]

                # 1. 杩濊璁＄畻
                if arr_v > tw_e:
                    total_violation += (arr_v - tw_e)

                # 2. 鏈嶅姟鏃堕棿璁＄畻 (瀹屽叏杩樺師鍘熶唬鐮侀€昏緫)
                # 璁＄畻鏂滃潯闀垮害锛屽垽鏂椂闂寸獥鏄惁杩囩獎
                slope_len = (tw_e - tw_s) - w_i

                opt_srv = arr_v  # 榛樿鍒濆鍖?

                if slope_len < 1e-5:
                    # 鏃堕棿绐楅潪甯哥獎锛屽彧鑳藉敖鏃╂湇鍔?
                    opt_srv = max(arr_v, tw_s)
                else:
                    # 椴佹浼樺寲锛氬皾璇曞湪鏃堕棿绐椾腑闂存湇鍔?
                    optimal_center = target_service[v]

                    if arr_v <= tw_e:
                        val = optimal_center
                        # 闄愬埗閫昏緫锛?
                        # 1. 涓嶈兘鏅氫簬鏃堕棿绐楃粨鏉?(铏界劧center閫氬父鍦ㄤ腑闂达紝浣嗛槻姝_i寮傚父)
                        if tw_e < val: val = tw_e
                        # 2. 鏍稿績锛氬鏋滃埌杈炬椂闂存瘮涓棿鐐规櫄锛岃杩帹杩熸湇鍔℃椂闂?
                        if arr_v > val: val = arr_v
                        # 3. 琛ュ厖锛氫笉鑳芥棭浜庢椂闂寸獥寮€濮?(铏界劧center > tw_s锛屼絾闃叉娴偣璇樊)
                        if val < tw_s: val = tw_s

                        opt_srv = val
                    else:
                        # 宸茬粡杩熷埌锛屽彧鑳界珛鍗虫湇鍔?
                        opt_srv = arr_v

                service_time[v] = opt_srv

                # 3. 婊℃剰搴﹁绠?(閫昏緫鏈彉锛屼絾杈撳叆鍙樹簡)
                delta_i = (tw_e - tw_s) - w_i
                if delta_i < 1e-5:
                    sat = 1.0
                else:
                    sat_left = (tw_e - opt_srv) / delta_i
                    sat_right = (opt_srv - tw_s) / delta_i

                    # 鎵嬪啓 min/max
                    sat = sat_left
                    if sat_right < sat: sat = sat_right
                    if sat > 1.0: sat = 1.0
                    if sat < 0.0: sat = 0.0

                total_sat += sat
                # --- [鍏抽敭淇缁撴潫] ---

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
# 1. 闂瀹炰緥涓庢暟鎹敓鎴?
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

        # weizhi目标：比较不同DTW位置偏好，默认宽度比例保持0.5
        self.dtw_width_ratio = 0.5
        self.stw = common_data['time_windows']

        customer_nodes = [i for i in self.nodes if i != 0]
        if customer_nodes:
            self.mean_tw_width = sum(self.stw[i][1] - self.stw[i][0] for i in customer_nodes) / len(customer_nodes)
        else:
            self.mean_tw_width = 0.0

        raw_coords = common_data['coords']
        ordered_coords = []
        for i in range(self.num_nodes):
            ordered_coords.append(raw_coords[i])
        self.coords = np.array(ordered_coords, dtype=np.float64)

        self.customer_types = common_data['types_int']

        self.dtw_width = np.zeros(self.num_nodes, dtype=np.float64)
        for i in self.nodes:
            width = float(self.stw[i][1] - self.stw[i][0]) * self.dtw_width_ratio
            self.dtw_width[i] = width

        self.dtw_mode = 'robust'
        self._dtw_samples_scaled = {}

        self._precompute_distances()

    def _precompute_distances(self):
        self.dist_matrix_truck, self.dist_matrix_drone = calc_dist_matrix_jit(self.coords)

    def get_truck_dist(self, i, j):
        return self.dist_matrix_truck[i][j]

    def get_drone_dist(self, i, j):
        return self.dist_matrix_drone[i][j]

    def precompute_dtw_samples(self, dtw_mode, num_samples=DTW_NUM_SAMPLES):
        self.dtw_mode = dtw_mode
        cfg = DTW_MODES[dtw_mode]
        rng = np.random.RandomState(42)
        self._dtw_samples_scaled = {}

        for node in self.nodes:
            if node == 0:
                continue

            raw = rng.beta(cfg['beta_a'], cfg['beta_b'], size=num_samples)
            tw_start = self.stw[node][0]
            tw_end = self.stw[node][1]
            delta = (tw_end - tw_start) - self.dtw_width[node]

            if delta < 1e-5:
                self._dtw_samples_scaled[node] = np.full(num_samples, tw_start)
            else:
                self._dtw_samples_scaled[node] = tw_start + raw * delta

    def check_dual_task_constraints(self, j, l):
        if self.customer_types[j] != 2:
            return False
        if self.customer_types[l] != 1:
            return False
        return True

# ==========================================
# 2. 鏌撹壊浣?(淇濇寔涓嶅彉)
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
        self.preferred_weight = random.uniform(0.0, 3.0)  # 鏂板锛氫釜浣撶殑鍋忓ソ鏉冮噸锛屽湪鍒濆鍖栨椂纭畾

# ==========================================
# 3. 鏍稿績瑙ｇ爜鍣?(娣卞害閲嶆瀯)
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

    # [缁撴瀯閲嶆瀯] 浣跨敤鍒嗙鐨勬暟鎹粨鏋勪唬鏇?Object
    labels_objs = [[] for _ in range(num_nodes)]
    labels_info = [[] for _ in range(num_nodes)]

    labels_objs_np = [np.empty((0, 2), dtype=np.float64) for _ in range(num_nodes)]
    # 璧风偣鍒濆鍖?
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

            # 涓存椂缂撳啿鍖猴紝鐢ㄤ簬鏈疆鎵╁睍
            new_entries_obj = []
            new_entries_info = []

            for sortie_plan, segment_dist, segment_time in all_plans:
                # 閬嶅巻褰撳墠鑺傜偣鐨勬墍鏈夋爣绛?(k鏄储寮?
                for k in range(len(curr_front_np)):
                    prev_dist = curr_front_np[k, 0]
                    prev_time = curr_front_np[k, 1]

                    new_cum_dist = prev_dist + segment_dist
                    new_cum_time = prev_time + segment_time

                    if next_truck_node != 0 and new_cum_time > tw_end_j + instance.mean_tw_width:
                        continue

                    # [鏍稿績浼樺寲] 璋冪敤 JIT 鍑芥暟鍒ゆ柇鏀厤
                    # 1. 妫€鏌ユ槸鍚﹁鐩爣鑺傜偣宸叉湁鐨勬爣绛炬敮閰?
                    if check_dominance_jit(target_front_np, new_cum_dist, new_cum_time):
                        continue

                    # 2. 妫€鏌ユ槸鍚﹁鏈疆鏂颁骇鐢熺殑鏍囩鏀厤 (绠€鍗曠殑璐績鍘婚噸锛岄伩鍏?Python 寰幆杩囨參)
                    # 涓烘€ц兘鑰冭檻锛岃繖閲岀暐鍘诲 new_entries 鐨勫叏閲忎袱涓ゆ瘮杈冿紝渚濋潬 step 1 鍜屽悗缁殑鍓灊

                    new_entries_obj.append([new_cum_dist, new_cum_time])
                    new_entries_info.append((i, k, sortie_plan))

            # 鎵归噺鏇存柊鍒扮洰鏍囪妭鐐?
            if new_entries_obj:
                # [浼樺寲 Plan B] 绉婚櫎 MAX_LABELS 纭埅鏂紝閲囩敤涓ユ牸甯曠疮鎵樼瓫閫?
                # 杩欎繚璇佷簡鏈€浼樻€у師鐞嗭紝鍚屾椂鍒╃敤 TDRP 璺濈涓庢椂闂寸殑寮虹浉鍏虫€ц嚜鐒舵帶鍒堕泦鍚堝ぇ灏?

                combined_objs = target_front + new_entries_obj
                combined_info = labels_info[j] + new_entries_info

                # 1. 棰勫鐞嗭細鎸夌涓€鐩爣锛堣窛绂?Dist锛夊崌搴忔帓鍒?
                # 鍦ㄨ窛绂荤浉鍚岀殑鎯呭喌涓嬶紝鎸夋椂闂?Time 鍗囧簭鎺掑垪锛堢‘淇濈浉鍚岃窛绂讳笅鍙繚鐣欐椂闂存渶鐭殑锛?
                combined = list(zip(combined_objs, combined_info))
                combined.sort(key=lambda x: (x[0][0], x[0][1]))

                final_objs = []
                final_info = []

                # 2. 鏍稿績绛涢€?(Pulse/Pareto Filter)锛?
                # 鐢变簬宸茬粡鎸?Dist 鎺掑簭锛屽悗缁殑瑙?Dist 涓€瀹?>= 鍓嶉潰鐨勮В銆?
                # 鍥犳锛屽悗缁В鍙湁鍦?Time < 鍓嶉潰鎵€鏈夎В鐨?min_time 鏃讹紝鎵嶆槸闈炴敮閰嶇殑銆?
                # 杩欐槸涓€涓?O(N) 鐨勭嚎鎬ф壂鎻忚繃绋嬨€?

                min_time_so_far = float('inf')

                for obj, inf in combined:
                    current_dist = obj[0]
                    current_time = obj[1]

                    # 瀹瑰樊澶勭悊锛岄伩鍏嶆诞鐐硅宸鑷寸殑铏氬亣闈炴敮閰?
                    if current_time < min_time_so_far - 1e-5:
                        final_objs.append(obj)
                        final_info.append(inf)
                        min_time_so_far = current_time

                    # 闅愬紡閫昏緫锛氬鏋?current_time >= min_time_so_far锛?
                    # 璇存槑瀛樺湪涓€涓?dist 鏇村皬涓?time 鏇寸煭(鎴栫浉绛?鐨勮В锛屽綋鍓嶈В琚敮閰嶏紝涓㈠純銆?

                # 鏇存柊鏁版嵁缁撴瀯
                labels_objs[j] = final_objs
                labels_info[j] = final_info

                # 淇濇寔 Numpy 鏁扮粍鍚屾锛岀敤浜庝笅涓€娆¤凯浠ｇ殑 JIT 鍔犻€?
                if final_objs:
                    labels_objs_np[j] = np.array(final_objs, dtype=np.float64)
                else:
                    labels_objs_np[j] = np.empty((0, 2), dtype=np.float64)

    # 4. 璺緞鍥炴函 (Selection & Backtracking)
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
            # obj: [dist, time]
            score = obj[0] + obj[1] * instance.v_truck * selection_weight
            if score < best_score:
                best_score = score
                best_idx = k

        # 鍥炴函鏋勫缓
        final_truck_route = []
        accepted_sorties = []

        curr_node_i = final_node_idx
        curr_label_k = best_idx

        trace_nodes = []
        trace_sorties = []

        while curr_node_i > 0:
            trace_nodes.append(extended_perm[curr_node_i])

            # 鑾峰彇褰撳墠鏍囩鐨勪俊鎭?
            prev_i, prev_k, sortie_plan = labels_info[curr_node_i][curr_label_k]

            if sortie_plan:
                trace_sorties = sortie_plan + trace_sorties

            curr_node_i = prev_i
            curr_label_k = prev_k

        final_truck_route = [0] + trace_nodes[::-1]
        accepted_sorties = trace_sorties

    if ind.rank == 0:
        accepted_sorties = _reassign_drone_endpoints(
            final_truck_route, accepted_sorties, instance
        )

    # 6. 鏈€缁堢簿纭瘎浼?[淇敼涓鸿皟鐢?JIT Wrapper]
    total_sat, _, _, total_violation_time = fast_timing_and_satisfaction(
        instance, final_truck_route, accepted_sorties
    )

    ind.constraint_violation = total_violation_time
    ind.is_feasible = (total_violation_time < 1e-6)

    # 璺濈璁＄畻淇濇寔 Python 鍗冲彲锛屽紑閿€涓嶅ぇ
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
    [浼樺寲鐗圿 鍩轰簬鐗╃悊绾︽潫锛堝悓姝ユ€т笌鍑犱綍鎬э級鐨勭敓鎴愬櫒锛屾浛浠ｇ函鏆村姏鏋氫妇銆?

    绗竴鎬у師鐞嗙害鏉燂細
    1. 璧勬簮绾︽潫锛氬€欓€夎妭鐐规暟涓嶈兘瓒呰繃鏃犱汉鏈虹殑鏈€澶ф惡甯﹁兘鍔涖€?
    2. 鏃堕棿鍚屾锛氭棤浜烘満椋炶鏃堕棿涓嶅簲鏄捐憲闀夸簬鍗¤溅琛岄┒鏃堕棿 (閬垮厤鍗¤溅闀挎椂闂寸瓑寰?銆?
    3. 鍑犱綍涓夎锛氳嫢 Drone 璺濈 >> Truck 璺濈锛屽垯璇ュ垎閰嶇墿鐞嗕笂浣庢晥銆?
    """
    num_drones = instance.NUM_DRONES
    get_dd = instance.get_drone_dist
    n_cand = len(candidates)

    # [瑙勫垯 1] 鍩烘暟鍓灊 (Cardinality Pruning)
    # 楦藉发鍘熺悊锛氬鏋滃€欓€夎妭鐐规暟澶锛屾棤浜烘満鏍规湰杩愪笉杩囨潵锛岀洿鎺ユ斁寮?
    # 鍋囪姣忎釜 sortie 鏈€澶氭湇鍔?2 涓妭鐐?(Dual Task)
    max_capacity = num_drones * 2
    if n_cand > max_capacity:
        return []

    # 鍗¤溅杩欎竴娈电殑鍩哄噯鏃堕棿
    truck_leg_time = truck_leg_dist / v_truck

    # [瑙勫垯 2] 瀹氫箟鏃堕棿鍚屾瀹瑰繊搴?(Synchronization Tolerance)
    # 鍏佽鏃犱汉鏈烘瘮鍗¤溅鎱竴鐐癸紝浣嗕笉鑳芥參澶銆?
    # 璁惧畾涓?2.0 鍊嶅崱杞︽椂闂?(鍗冲厑璁稿崱杞︾瓑寰呯殑鏃堕棿 = 鍗¤溅璺▼鏃堕棿)
    # 杩欐槸涓€涓墿鐞嗛檺鍒讹細濡傛灉绛夊緟鏃堕棿瓒呰繃璺▼鏃堕棿锛屼笉濡傜洿鎺ヨ鍗¤溅鍘婚€併€?
    max_allowed_drone_time = truck_leg_time * 2.0 + 300  # +300绉掍綔涓哄浐瀹氱紦鍐?

    # 棰勮绠楋細鍗曚釜鑺傜偣鐨勫彲琛屾€?(鍑犱綍鍓灊棰勫鐞?
    # 濡傛灉鏌愪釜鑺傜偣澶繙锛屽鑷村崟娆″線杩旈兘瓒呮椂锛屽垯鍖呭惈璇ヨ妭鐐圭殑浠讳綍缁勫悎閮戒笉鍙
    valid_single_indices = []
    for i in range(n_cand):
        node = candidates[i]
        # 浼扮畻鏈€灏忛琛屾椂闂达細Launch -> Node -> Recover
        min_flight_dist = get_dd(launch_node, node) + get_dd(node, recover_node)
        if min_flight_dist / v_drone <= max_allowed_drone_time:
            valid_single_indices.append(i)

    # 濡傛灉鏈夎妭鐐硅繛鍗曟浠诲姟閮藉仛涓嶅埌锛屼笖瀹冨繀椤昏鍒嗛厤锛坈andidates蹇呴』鍏ㄩ儴鍒嗛厤锛夛紝
    # 閭ｄ箞鏁翠釜 candidates 闆嗗悎灏辨槸涓嶅彲琛岀殑銆?
    if len(valid_single_indices) < n_cand:
        return []

    results = []
    visited_states = set()

    def _backtrack(idx, partition):
        state = (idx, len(partition))

        if state in visited_states:
            return

        visited_states.add(state)
        # 鎴愬姛鍩哄噯锛氭墍鏈夊€欓€夎妭鐐归兘宸插垎閰?
        if idx == n_cand:
            sortie_plan = []
            total_flight_dist = 0.0
            max_sortie_time = 0.0

            for group in partition:
                # 鏋勫缓璺緞
                path_nodes = group

                # 璁＄畻璺濈
                # Launch -> First
                fd = get_dd(launch_node, path_nodes[0])
                # Inter-nodes
                for k in range(len(path_nodes) - 1):
                    fd += get_dd(path_nodes[k], path_nodes[k + 1])
                # Last -> Recover
                fd += get_dd(path_nodes[-1], recover_node)

                # [瑙勫垯 3] 涓ユ牸鐨勬椂闂村悓姝ユ鏌?(Strict Time Check)
                # 鍦ㄧ敓鎴愭柟妗堢殑涓€鐬棿锛屽鏋滃彂鐜版煇鏋舵棤浜烘満椋炲お涔咃紝鐩存帴涓㈠純璇ユ柟妗?
                st = fd / v_drone
                if st > max_allowed_drone_time:
                    return  # 鍓灊锛氭鏂规鏃犳晥

                total_flight_dist += fd
                if st > max_sortie_time:
                    max_sortie_time = st

                sortie_plan.append((launch_node, list(path_nodes), recover_node))

            # 鍙湁褰撴棤浜烘満鏈€闀块琛屾椂闂村湪鍙帴鍙楄寖鍥村唴鏃讹紝鎵嶆帴鍙楁鏂规
            seg_dist = truck_leg_dist + total_flight_dist
            seg_time = max(truck_leg_time, max_sortie_time)

            results.append((sortie_plan, seg_dist, seg_time))
            return

        # 鍓灊锛氬鏋滃凡鐢ㄧ殑鏃犱汉鏈烘暟閲忚揪鍒颁笂闄?
        if len(partition) >= num_drones:
            return

        # 灏濊瘯鍒嗛厤 candidates[idx]

        # 閫夐」 A: 鍗曚换鍔?(Single Task)
        # 灏嗗綋鍓嶈妭鐐逛綔涓轰竴涓柊鐨?sortie
        partition.append([candidates[idx]])
        _backtrack(idx + 1, partition)
        partition.pop()

        # 閫夐」 B: 鍙屼换鍔?(Dual Task)
        # 灏濊瘯灏嗗綋鍓嶈妭鐐逛笌涓嬩竴涓妭鐐规墦鍖?
        # 绾︽潫锛氬繀椤昏繛缁紝涓旀弧瓒?D->P 绾︽潫
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
    鍚庡鐞嗕紭鍖栵細鍦ㄧ‘瀹氱殑鍗¤溅璺緞涓婏紝涓烘瘡涓?sortie 閲嶆柊閫夋嫨
    璧烽鐐瑰拰鍥炴敹鐐癸紝鍏佽鏃犱汉鏈鸿法瓒婁腑闂村崱杞﹀仠闈犵偣椋炶銆?

    涓嶆敼鍙?truck_route锛堝崱杞﹀仠闈犲摢浜涜妭鐐逛笉鍙橈級銆?
    涓嶆敼鍙樻瘡涓?sortie 鐨?path_nodes锛堝摢浜涜妭鐐圭敱鏃犱汉鏈烘湇鍔′笉鍙橈級銆?
    鍙敼鍙樻瘡涓?sortie 鐨?launch_node 鍜?recover_node銆?

    鏍稿績淇濊瘉锛氫互鍘熷鍒嗛厤锛堝悎娉曪級涓哄熀鍑嗭紝浠呭湪涓嶈繚鍙?NUM_DRONES 绾︽潫鏃舵墠鏀硅繘銆?
    绾︽潫锛氫换鎰忓崱杞﹁矾寰勬涓婂悓鏃跺湪閫旂殑鏃犱汉鏈烘暟 鈮?instance.NUM_DRONES銆?
    杈撳嚭鏍煎紡涓庤緭鍏ュ畬鍏ㄤ竴鑷达細[(launch, [path_nodes], recover), ...]
    """
    if not sorties:
        return sorties

    num_stops = len(truck_route)
    num_drones = instance.NUM_DRONES
    get_dd = instance.get_drone_dist

    def _flight_dist(l_node, path_nodes, r_node):
        """Compute flight distance for one sortie."""
        fd = get_dd(l_node, path_nodes[0])
        for k in range(len(path_nodes) - 1):
            fd += get_dd(path_nodes[k], path_nodes[k + 1])
        fd += get_dd(path_nodes[-1], r_node)
        return fd

    # --- Step 1: 纭畾姣忎釜 sortie 鍦?truck_route 涓婄殑鍘熷 (l_idx, r_idx) ---
    # 鏋勫缓鑺傜偣 鈫?鍦?truck_route 涓嚭鐜扮殑鎵€鏈変綅缃储寮?
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
                    # 閫夎法搴︽渶灏忕殑鍖归厤锛堝師濮?Split 浜х敓鐨勫垎閰嶈法搴︽渶鐭級
                    if best_l is None or (ri - li) < (best_r - best_l):
                        best_l, best_r = li, ri
        if best_l is None:
            # 鐞嗚涓婁笉浼氬彂鐢燂紙Split 浜х敓鐨?launch/recover 涓€瀹氬湪 truck_route 涓婏級
            best_l, best_r = 0, num_stops - 1
        original_assignments.append((best_l, best_r))

    # --- Step 2: 鐢ㄥ師濮嬪垎閰嶅垵濮嬪寲 segment_usage锛堜繚璇佸悎娉曞熀绾匡級 ---
    segment_usage = [0] * (num_stops - 1)
    for (l_idx, r_idx) in original_assignments:
        for seg in range(l_idx, r_idx):
            segment_usage[seg] += 1

    # --- Step 3: 涓烘瘡涓?sortie 鐢熸垚鍊欓€夊苟灏濊瘯鏀硅繘 ---
    all_candidates = []
    orig_fds = []
    for s_idx, (l_node, path_nodes, r_node) in enumerate(sorties):
        orig_fds.append(_flight_dist(l_node, path_nodes, r_node))
        candidates = []
        for l_i in range(num_stops - 1):
            for r_i in range(l_i + 1, num_stops):
                fd = _flight_dist(truck_route[l_i], path_nodes, truck_route[r_i])
                candidates.append((fd, l_i, r_i))
        candidates.sort()  # 椋炶璺濈鍗囧簭
        all_candidates.append(candidates)

    # 鎸夈€屾敼鍠勬綔鍔涖€嶉檷搴忓鐞嗭細鍘熷椋炶璺濈涓庢渶浣冲€欓€変箣宸秺澶э紝浼樺厛绾ц秺楂?
    sortie_indices = list(range(len(sorties)))
    sortie_indices.sort(key=lambda s: all_candidates[s][0][0] - orig_fds[s])

    final_assignments = list(original_assignments)  # 榛樿淇濇寔鍘熷

    for s_idx in sortie_indices:
        orig_l, orig_r = original_assignments[s_idx]
        orig_fd = orig_fds[s_idx]

        # 浠?segment_usage 涓殏鏃剁Щ闄ゅ綋鍓?sortie 鐨勫崰鐢?
        for seg in range(orig_l, orig_r):
            segment_usage[seg] -= 1

        # 鍦ㄦ墍鏈夊€欓€変腑瀵绘壘椋炶璺濈鏇寸煭涓斾笉杩濆弽绾︽潫鐨勫垎閰?
        improved = False
        for (fd, l_i, r_i) in all_candidates[s_idx]:
            if fd >= orig_fd:
                # 鍚庣画鍊欓€夎窛绂诲彧浼氭洿澶э紝涓嶅彲鑳芥敼杩?
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
            # 鏃犳硶鏀硅繘锛氭仮澶嶅師濮嬪垎閰嶏紙淇濊瘉鍚堟硶鎬э級
            for seg in range(orig_l, orig_r):
                segment_usage[seg] += 1

    # --- Step 4: 鏋勫缓鏂?sorties ---
    new_sorties = []
    for s_idx, (_, path_nodes, _) in enumerate(sorties):
        l_idx, r_idx = final_assignments[s_idx]
        new_sorties.append((truck_route[l_idx], list(path_nodes), truck_route[r_idx]))

    # --- 瀹夊叏闂細璺濈鍑忓皯 + 鍙鎬т笉鎭跺寲 鎵嶉噰绾?---
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
# 5. NSGA-II 妗嗘灦 (淇濇寔涓嶅彉)
# ==========================================
def fast_non_dominated_sort(population):
    """
    甯︾害鏉熺殑蹇€熼潪鏀厤鎺掑簭
    瑙勫垯锛氬彲琛岃В鎬绘槸鏀厤涓嶅彲琛岃В锛涗笉鍙瑙ｄ箣闂存瘮杈冭繚瑙勯噺
    [浼樺寲] 棰勬彁鍙栫洰鏍囧€煎拰绾︽潫淇℃伅涓哄眬閮ㄦ暟缁勶紝鍑忓皯鍐呭惊鐜腑鐨勫睘鎬ф煡鎵惧紑閿€
    """
    n = len(population)

    # [浼樺寲] 棰勬彁鍙栵細灏嗘瘡涓釜浣撶殑鍏抽敭灞炴€ф彁鍙栧埌绱у噾鐨勫眬閮ㄦ暟缁勪腑
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

            # 瑙勫垯1锛氬彲琛岃В鎬绘槸鏀厤涓嶅彲琛岃В
            if p_feas and not q_feas:
                p_dominates_q = True
            elif not p_feas and q_feas:
                q_dominates_p = True
            # 瑙勫垯2锛氫袱鑰呴兘鍙锛屼娇鐢ㄦ爣鍑哖areto鏀厤
            elif p_feas and q_feas:
                if (p_o0 <= q_o0 and p_o1 <= q_o1) and \
                        (p_o0 < q_o0 or p_o1 < q_o1):
                    p_dominates_q = True
                elif (q_o0 <= p_o0 and q_o1 <= p_o1) and \
                        (q_o0 < p_o0 or q_o1 < p_o1):
                    q_dominates_p = True
            # 瑙勫垯3锛氫袱鑰呴兘涓嶅彲琛岋紝姣旇緝杩濊閲忓拰鐩爣
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

    # [浼樺寲] 绗簩闃舵鍏ㄩ儴鍩轰簬鏁存暟绱㈠紩鎿嶄綔锛岄伩鍏嶅璞″睘鎬ф煡鎵?
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


def crowding_distance_assignment(front):
    l = len(front)
    if l == 0: return
    for p in front: p.crowding_distance = 0

    # [鏂板 1] 棰勫厛璁＄畻褰撳墠 Front 鍦ㄦ瘡涓淮搴︿笂鐨勬瀬鍊硷紝鐢ㄤ簬鍏ㄥ眬褰掍竴鍖?
    # 杩欎竴姝ヨ嚦鍏抽噸瑕侊紝瀹冨畾涔変簡褰掍竴鍖栫殑"鏍囧昂"
    # 浣跨敤 Python 鍘熺敓鍒楄〃鎺ㄥ寮忥紝涓嶅紩鍏?numpy 浠ヤ繚鎸佺粨鏋勭函鍑€
    min_objs = [min(ind.objectives[m] for ind in front) for m in range(2)]
    max_objs = [max(ind.objectives[m] for ind in front) for m in range(2)]

    for m in range(2):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = float('inf')
        front[l - 1].crowding_distance = float('inf')

        # [鏂板 2] 璁＄畻璇ョ淮搴︾殑鏋佸樊 (Range)
        obj_diff = max_objs[m] - min_objs[m]

        # 椴佹鎬ф鏌ワ細闃叉闄や互闆?
        # 濡傛灉褰撳墠 Front 涓墍鏈変釜浣撳湪璇ョ洰鏍囦笂鐨勫€奸兘鐩稿悓 (diff=0)锛?
        # 璇存槑璇ョ洰鏍囨棤娉曟彁渚涘尯鍒嗗害锛岀洿鎺ヨ烦杩囩疮鍔?
        if obj_diff == 0:
            continue

        for i in range(1, l - 1):
            # [淇敼 3] 鎵ц褰掍竴鍖栫疮鍔?
            # 鍏紡锛?Next_Obj - Prev_Obj) / Range
            # 杩欐牱澶勭悊鍚庯紝Cost (鑼冨洿3000) 鍜?Sat (鑼冨洿100) 鐨勮础鐚兘琚媺鍒颁簡
            # 鍚屼竴涓噺绾?(鐧惧垎姣?锛岀‘淇濅簡澶氱洰鏍囧垎甯冪殑鍧囧寑鎬?

            dist_contribution = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_diff
            front[i].crowding_distance += dist_contribution
def crossover(p1, p2, instance):
    size = len(p1.permutation)
    s, e = sorted(random.sample(range(size), 2))

    # 1. 鍒濆鍖栧瓙浠ｈ矾寰?
    c1_perm = [-1] * size
    c2_perm = [-1] * size

    # 2. 缁ф壙鏍稿績鐗囨 (Inherit Core Segment)
    c1_perm[s:e] = p1.permutation[s:e]
    c2_perm[s:e] = p2.permutation[s:e]

    # 3. 濉厖鍓╀綑閮ㄥ垎 (Fill Rest) - 鏍囧噯OX浜ゅ弶
    def fill_remaining(child_perm, fill_parent):
        curr = e
        # [浼樺寲] 鐢?set 鏇夸唬姣忔杩唬鐨勫垏鐗?绾挎€ф悳绱紝O(N虏) -> O(N)
        inherited_set = set(child_perm[s:e])
        # 濉厖寰幆
        for i in range(size):
            cand = fill_parent.permutation[(e + i) % size]
            if cand not in inherited_set:
                child_perm[curr] = cand
                curr = (curr + 1) % size

    # 4. 濉厖涓や釜瀛愪唬
    fill_remaining(c1_perm, p2)
    fill_remaining(c2_perm, p1)

    # 5. 鍒涘缓瀛愪唬涓綋
    c1 = Individual(size)
    c1.permutation = c1_perm
    c1.preferred_weight = p1.preferred_weight  # 缁ф壙鐖朵唬1鐨勫亸濂芥潈閲?

    c2 = Individual(size)
    c2.permutation = c2_perm
    c2.preferred_weight = p2.preferred_weight  # 缁ф壙鐖朵唬2鐨勫亸濂芥潈閲?

    return c1, c2
def mutation(ind, instance):
    """
    [淇敼璇存槑] 绠€鍖栦负缁忓吀鐨勫簭鍒楀彉寮傜畻瀛?
    鍘熷洜锛?
    1. 鍙渶瀵筽ermutation杩涜鍙樺紓
    2. 浣跨敤VRP棰嗗煙鎴愮啛鐨勫彉寮傜畻瀛愮粍鍚?
    3. Split绠楁硶浼氳嚜鍔ㄤ负鍙樺紓鍚庣殑搴忓垪鎵惧埌鏈€浼樺垎閰?
    """
    size = len(ind.permutation)
    r = random.choice([1, 2, 3, 4])

    if r == 1:
        # Insertion Mutation (鎻掑叆鍙樺紓)
        # 闅忔満閫夋嫨涓€涓妭鐐癸紝绉诲姩鍒板彟涓€涓綅缃?
        i, j = random.sample(range(size), 2)
        val_node = ind.permutation.pop(i)
        ind.permutation.insert(j, val_node)

    elif r == 2:
        # Swap Mutation (浜ゆ崲鍙樺紓)
        # 闅忔満浜ゆ崲涓や釜鑺傜偣鐨勪綅缃?
        i, j = random.sample(range(size), 2)
        ind.permutation[i], ind.permutation[j] = ind.permutation[j], ind.permutation[i]

    elif r == 3:
        # Scramble Mutation (鎵撲贡鍙樺紓)
        # 闅忔満閫夋嫨涓€娈靛瓙搴忓垪锛屾墦涔卞叾鍐呴儴椤哄簭
        i, j = sorted(random.sample(range(size), 2))
        segment = ind.permutation[i:j + 1]
        random.shuffle(segment)
        ind.permutation[i:j + 1] = segment

    elif r == 4:
        # ALNS: Adaptive Large Neighborhood Search
        # 鏅鸿兘鐮村潖涓庨噸鏋勭畻瀛?

        # 1. 鏅鸿兘鐮村潖锛氱Щ闄ゆ椂闂寸獥绱ц揩鐨勮妭鐐?
        num_remove = max(2, int(size ** 0.5))

        tightness = []
        for idx, node_id in enumerate(ind.permutation):
            start_time = instance.stw[node_id][0]
            tightness.append((idx, start_time))

        # 鎸夋椂闂寸獥寮€濮嬫椂闂存帓搴忥紙瓒婃棭瓒婄揣杩級
        tightness.sort(key=lambda x: x[1])

        # 鎻愬彇瑕佺Щ闄ょ殑绱㈠紩锛堜粠鍚庡線鍓嶅垹闄ら伩鍏嶇储寮曞亸绉伙級
        indices_to_remove = sorted([t[0] for t in tightness[:num_remove]], reverse=True)

        removed_nodes = []
        for idx in indices_to_remove:
            n_id = ind.permutation.pop(idx)
            removed_nodes.append(n_id)

        # 2. 鏅鸿兘閲嶆瀯锛氭渶寤変环鎻掑叆锛圕heapest Insertion锛?
        for node_to_insert in removed_nodes:
            best_pos = 0
            min_added_cost = float('inf')

            current_len = len(ind.permutation)
            for pos in range(current_len + 1):
                prev_node = ind.permutation[pos - 1] if pos > 0 else 0
                next_node = ind.permutation[pos] if pos < current_len else 0

                # 璁＄畻鎻掑叆鎴愭湰锛堜娇鐢ㄥ崱杞﹁窛绂讳綔涓哄熀鍑嗭級
                cost = instance.get_truck_dist(prev_node, node_to_insert) + \
                       instance.get_truck_dist(node_to_insert, next_node) - \
                       instance.get_truck_dist(prev_node, next_node)

                if cost < min_added_cost:
                    min_added_cost = cost
                    best_pos = pos

            # 鎵ц鎻掑叆
            ind.permutation.insert(best_pos, node_to_insert)

def perturb_tour(base_tour, strength=2):
    """
    瀵瑰熀鍑嗚矾寰勬柦鍔犲彈鎺ф壈鍔?
    strength: 鎵板姩娆℃暟锛岃秺澶у亸绂昏秺杩?
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
            if seg_len > 5:  # 鍙墦涔卞皬鐗囨锛屼繚鐣欐暣浣撶粨鏋?
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
# 5. 涓荤▼搴忓叆鍙?
def run_nsga_solver(common_data, shared_ref_point=None, shared_ref_front=None, dtw_mode='robust'):
    # 鍙傛暟閰嶇疆
    POP_SIZE = int(os.environ.get('NSGA_POP_SIZE', '300'))
    GEN_MAX = int(os.environ.get('NSGA_GEN_MAX', '120'))
    # 浠庢暟鎹腑鑾峰彇瀹㈡埛鏁伴噺
    NUM_CUSTOMERS = len(common_data['nodes']) - 1
    global CACHE_LIMIT
    CACHE_LIMIT = POP_SIZE * GEN_MAX * 2
    # 1. 鍒濆鍖栧苟娉ㄥ叆鏁版嵁
    instance = ProblemInstance(common_data, num_customers=NUM_CUSTOMERS)
    data_loader.inject_data_to_nsga(instance, common_data)

    global SOLUTION_CACHE
    SOLUTION_CACHE = {}

    if dtw_mode != 'robust' and dtw_mode not in DTW_MODES:
        raise ValueError(f"Unknown dtw_mode: {dtw_mode}")

    if dtw_mode != 'robust':
        instance.precompute_dtw_samples(dtw_mode)
        mode_label = DTW_MODES[dtw_mode]['label']
    else:
        instance.dtw_mode = 'robust'
        mode_label = 'Robust (Worst-Case)'

    print(f"\n{'='*60}")
    print(f"  NSGA-II Solver - DTW Mode: {mode_label}")
    print(f"{'='*60}")

    # 鑷€傚簲鍓灊瀹瑰繊搴︼細鍩轰簬鎵€鏈夊鎴锋椂闂寸獥鐨勫钩鍧囧搴?
    customer_nodes = [i for i in instance.nodes if i != 0]
    instance.mean_tw_width = sum(
        instance.stw[i][1] - instance.stw[i][0] for i in customer_nodes
    ) / len(customer_nodes)
    print("NSGA-II Data Overwritten with Solomon File.")
    t_solver_start = time.time()
    print("Initialize Population with First-Principles (Order & Chaos)...")
    population = []

    # --- 杈呭姪鍑芥暟锛氭渶杩戦偦璺緞 (浣撶幇鐗╃悊鍑犱綍鐨勬渶浼樻€? ---
    def get_nn_tour(inst):
        unvisited = set(inst.nodes)
        unvisited.remove(0)  # 绉婚櫎 Depot
        curr = 0
        tour = []
        while unvisited:
            # 璐┆閫夋嫨锛氬幓绂诲綋鍓嶇偣鏈€杩戠殑涓嬩竴涓偣
            nxt = min(unvisited, key=lambda n: inst.get_truck_dist(curr, n))
            tour.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return tour

    def get_drone_aware_tour(inst, customers):
        """
        鏋勯€犳€濊矾锛氬皢 D-P 閰嶅鐩搁偦鏀剧疆锛岀┛鎻掑湪 SPD 鑺傜偣涔嬮棿銆?
        浣?Split 瑙ｇ爜鍣ㄥ湪鍓嶇灮绐楀彛 (max_lookahead=NUM_DRONES) 鍐?
        鏇村鏄撳彂鐜版湁鏁堢殑鍙屼换鍔″拰鍗曚换鍔℃棤浜烘満娲鹃仯鏂规銆?
        """
        # 鎸夊鎴风被鍨嬪垎缁?(绫诲瀷缂栫爜涓?check_dual_task_constraints 涓€鑷?
        d_nodes = [n for n in customers if inst.customer_types[n] == 2]  # Delivery
        p_nodes = [n for n in customers if inst.customer_types[n] == 1]  # Pickup
        spd_nodes = [n for n in customers if inst.customer_types[n] == 3]  # SPD (浠呭崱杞?

        # 鍚勭粍鍐呴儴鎸夎窛 Depot 鐨勫崱杞﹁窛绂绘帓搴忥紝淇濇寔绌洪棿灞€閮ㄦ€?
        d_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        p_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))
        spd_nodes.sort(key=lambda n: inst.get_truck_dist(0, n))

        # 鏋勯€?D鈫扨 閰嶅闃熷垪 (婊¤冻 check_dual_task_constraints 鐨勯『搴忚姹?
        drone_pairs = []
        remaining_d = list(d_nodes)
        remaining_p = list(p_nodes)
        while remaining_d and remaining_p:
            drone_pairs.append((remaining_d.pop(0), remaining_p.pop(0)))
        unpaired = remaining_d + remaining_p  # 鏈厤瀵圭殑鍓╀綑鑺傜偣

        # 浜ゆ浛鎻掑叆锛歋PD 浣滀负鍗¤溅閿氱偣锛孌-P 瀵规彃鍏ュ叾闂?
        tour = []
        pi, si = 0, 0
        while pi < len(drone_pairs) or si < len(spd_nodes):
            # 鍏堟斁涓€涓?SPD 閿氱偣 (鍗¤溅缁忓仠)
            if si < len(spd_nodes):
                tour.append(spd_nodes[si])
                si += 1
            # 鍐嶆斁涓€涓?D-P 瀵?(钀藉叆涓嬩竴娈电殑 drone_candidates 绐楀彛)
            if pi < len(drone_pairs):
                tour.append(drone_pairs[pi][0])  # D 鍦ㄥ墠
                tour.append(drone_pairs[pi][1])  # P 鍦ㄥ悗
                pi += 1
        # 鏈厤瀵硅妭鐐硅拷鍔犲埌鏈熬
        tour.extend(unpaired)
        return tour
    # 棰勮绠楃墿鐞嗗熀鍑嗚矾寰?(Ground Truth Approximation)
    nn_tour = get_nn_tour(instance)

    # [鏂板] 棰勮绠楁椂闂村熀鍑嗚矾寰?(Time-Aware Approximation)
    all_customers = [n for n in instance.nodes if n != 0]
    time_sorted_tour = sorted(all_customers, key=lambda n: instance.stw[n][0])

    # 棰勮绠楁棤浜烘満鎰熺煡璺緞 (Drone-Aware Approximation)
    drone_aware_tour = get_drone_aware_tour(instance, all_customers)

    # 鍔ㄦ€佸畾涔夊洓绉嶅垵濮嬪寲绛栫暐鐨勮竟鐣?(閬垮厤榄旀硶鏁板瓧)
    # 灏嗙缇ゅ垎涓哄洓浠? 绌洪棿浼樺厛 / 鏃堕棿浼樺厛 / 鏃犱汉鏈烘劅鐭?/ 闅忔満
    limit_spatial = POP_SIZE // 4
    limit_temporal = POP_SIZE // 2
    limit_drone = (POP_SIZE * 3) // 4

    for i in range(POP_SIZE):
        # 鍒涘缓涓綋
        ind = Individual(NUM_CUSTOMERS)

        # [绛栫暐 1: 绌洪棿鏈夊簭] (Geometry / Cost Driven)
        # 姣忕粍绗?1 涓繚鐣欏師濮嬪惎鍙戝紡璺緞锛屽叾浣欐寜缁勫唴浣嶇疆閫掑鎵板姩
        # 褰㈡垚浠?绮剧‘鍚彂寮?鍒?杩戦殢鏈?鐨勬搴︼紝鍏奸【璐ㄩ噺涓庡鏍锋€?
        if i < limit_spatial:
            position_in_group = i
            ind.permutation = perturb_tour(nn_tour, strength=position_in_group)

        # [绛栫暐 2: 鏃堕棿鏈夊簭] (Time / Feasibility Driven)
        elif i < limit_temporal:
            position_in_group = i - limit_spatial
            ind.permutation = perturb_tour(time_sorted_tour, strength=position_in_group)

        # [绛栫暐 3: 鏃犱汉鏈烘劅鐭 (Drone Coordination Driven)
        elif i < limit_drone:
            position_in_group = i - limit_temporal
            ind.permutation = perturb_tour(drone_aware_tour, strength=position_in_group)

        # [绛栫暐 4: 娣锋矊] (Chaos / Diversity)
        # 瀹屽叏闅忔満鎵撲贡锛岄槻姝㈢畻娉曢櫡鍏ュ眬閮ㄦ渶浼?
        else:
            ind.permutation = list(np.random.permutation(NUM_CUSTOMERS) + 1)

        # 鉁?鍙岄噸瑙ｇ爜锛氶殢鏈洪€夋嫨瑙ｇ爜妯″紡
        # 浣跨敤涓綋鑷韩鐨?preferred_weight 杩涜瑙ｇ爜锛屼繚璇佽瘎浼颁竴鑷存€?
        decode_and_evaluate(ind, instance, progress=0.0, time_weight=ind.preferred_weight)
        population.append(ind)
        # print(ind.permutation)

    _init_dists = [p.objectives[0] for p in population]
    weight_upper = max(_init_dists) / (min(_init_dists) + 1e-10)
    # 鏍℃鍒濆绉嶇兢鐨?preferred_weight 鍒拌嚜閫傚簲鑼冨洿锛屽苟閲嶆柊璇勪及浠ヤ繚鎸佷竴鑷存€?
    for p in population:
        p.preferred_weight = random.uniform(0.0, weight_upper)
        decode_and_evaluate(p, instance, progress=0.0, time_weight=p.preferred_weight)
    pareto_archive = []
    print("Start Evolution...")
    # 浜ゅ弶姒傜巼锛氱敱闂瑙勬ā鎺ㄥ锛孨瓒婂ぇ瓒婃帴杩?.0
    pc = (NUM_CUSTOMERS - 1.0) / NUM_CUSTOMERS

    for gen in range(GEN_MAX):
        current_progress = gen / float(GEN_MAX)
        offspring = []

        # 鑷€傚簲鍙樺紓姒傜巼锛氬垵鏈?1.0锛堝己鎺㈢储锛夛紝鏈湡=1/N锛堝急鎵板姩锛?
        pm = 1.0 / NUM_CUSTOMERS + (1.0 - 1.0 / NUM_CUSTOMERS) * (1.0 - current_progress)

        # 甯歌 NSGA-II 杩涘寲娴佺▼
        while len(offspring) < POP_SIZE:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            # 浜ゅ弶姒傜巼鎺у埗
            if random.random() < pc:
                c1, c2 = crossover(p1, p2, instance)
            else:
                # 涓嶄氦鍙夋椂锛屽瓙浠ｇ洿鎺ョ户鎵跨埗浠ｅ熀鍥?
                c1 = Individual(NUM_CUSTOMERS)
                c1.permutation = list(p1.permutation)
                c1.preferred_weight = p1.preferred_weight
                c2 = Individual(NUM_CUSTOMERS)
                c2.permutation = list(p2.permutation)
                c2.preferred_weight = p2.preferred_weight

            # 鍙樺紓姒傜巼鎺у埗
            if random.random() < pm:
                mutation(c1, instance)
            if random.random() < pm:
                mutation(c2, instance)

            # 浣跨敤瀛愪唬鑷韩鐨?preferred_weight 杩涜瑙ｇ爜锛屼繚璇佽瘎浼颁竴鑷存€?
            # c1.preferred_weight 缁ф壙鑷?p1锛堜氦鍙夌 596 琛岋級锛宑2 缁ф壙鑷?p2锛堢 600 琛岋級
            decode_and_evaluate(c1, instance, progress=current_progress, time_weight=c1.preferred_weight)
            decode_and_evaluate(c2, instance, progress=current_progress, time_weight=c2.preferred_weight)
            offspring.extend([c1, c2])

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

        # [鏂板] 妯″洜绠楁硶锛氬 Pareto 鍓嶆部绮捐嫳杩涜灞€閮ㄦ悳绱㈠己鍖?
        # 1. 鑾峰彇鎵€鏈?Rank 0
        all_rank0 = [p for p in population if p.rank == 0]

        # 2. 鎸夋嫢鎸よ窛绂婚檷搴忔帓鍒?(浼樺厛浼樺寲绋€鐤忓尯鍩熺殑瑙ｏ紝鎵╁睍鍓嶆部)
        all_rank0.sort(key=lambda x: x.crowding_distance, reverse=True)

        # 3. 鍔ㄦ€佽绠楅绠楋細鍙栫缇よ妯＄殑骞虫柟鏍?
        # ls_budget = max(1, int(len(population) ** 0.5 * (1.0 - current_progress)))
        ls_budget = max(1, int(len(population) ** 0.5 * 0.3))
        elite_individuals = all_rank0[:ls_budget]

        for elite in elite_individuals:
            # 淇濆瓨灞€閮ㄦ悳绱㈠墠鐨勫畬鏁寸姸鎬佸揩鐓?
            saved_permutation = list(elite.permutation)
            saved_objectives = list(elite.objectives)
            saved_violation = elite.constraint_violation
            saved_feasible = elite.is_feasible
            saved_schedule = elite.decoded_schedule

            # local_search_improvement(elite, instance, progress=current_progress)

            # 鍥炴粴淇濇姢闂細濡傛灉灞€閮ㄦ悳绱㈠悗鏃цВ鏀厤鏂拌В锛屽垯鍥為€€
            # 杩欑‘淇濈簿鑻变釜浣撳湪灞€閮ㄦ悳绱㈠悗涓嶄細鍙樺緱鏇村樊
            new_obj = elite.objectives
            old_obj = saved_objectives

            rollback = False

            if saved_feasible and not elite.is_feasible:
                # 浠庡彲琛岄€€鍖栦负涓嶅彲琛岋細蹇呴』鍥炴粴
                rollback = True
            elif saved_feasible and elite.is_feasible:
                # 閮藉彲琛岋細妫€鏌ユ棫瑙ｆ槸鍚?Pareto 鏀厤鏂拌В
                # 鏀厤瀹氫箟涓?fast_non_dominated_sort 涓鍒?2锛堢 483-489 琛岋級涓€鑷?
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
                # 瀛樻。瑙ｆ敮閰嶆柊瑙ｏ紵
                if (arch_obj[0] <= new_obj[0] and arch_obj[1] <= new_obj[1]) and \
                        (arch_obj[0] < new_obj[0] or arch_obj[1] < new_obj[1]):
                    is_dominated = True
                    break
                # 鏂拌В鏀厤瀛樻。瑙ｏ紵
                if (new_obj[0] <= arch_obj[0] and new_obj[1] <= arch_obj[1]) and \
                        (new_obj[0] < arch_obj[0] or new_obj[1] < arch_obj[1]):
                    to_remove.append(idx)

            if not is_dominated:
                for idx in sorted(to_remove, reverse=True):
                    pareto_archive.pop(idx)
                # 鍘婚噸锛氭鏌ユ槸鍚﹀凡瀛樺湪瀹屽叏鐩稿悓鐨勭洰鏍囧€?
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
                # 褰撲唬绉嶇兢涓殑鏈€浼橈紙鐢ㄤ簬瀵规瘮瑙傚療锛?
                gen_best_dist = min(p.objectives[0] for p in feasible_pop)
                gen_max_sat = -min(p.objectives[1] for p in feasible_pop)

                # 浠庡叏灞€瀛樻。涓彁鍙栧巻鍙叉渶浼橈紙鍗曠洰鏍囩淮搴︾殑鏋佸€硷紝闈炲悓涓€涓В锛?
                arch_best_dist = min(e['objectives'][0] for e in pareto_archive)
                arch_max_sat = -min(e['objectives'][1] for e in pareto_archive)

                print(f"Gen {gen}: Gen Dist={gen_best_dist:.2f}, Gen Sat={gen_max_sat:.2f} | "
                      f"Archive Best Dist={arch_best_dist:.2f}, Archive Best Sat={arch_max_sat:.2f}")
            else:
                # 濡傛灉绉嶇兢涓叏鏄笉鍙瑙ｏ紙鏃╂湡鍙兘鍑虹幇锛夛紝鍒欏洖閫€鍒版墦鍗版墍鏈?
                raw_dist = min(p.objectives[0] for p in population)
                print(f"Gen {gen}: No feasible sol yet. Raw Dist={raw_dist:.2f}")

    # 缁撴灉鍙鍖?
    t_solver_end = time.time()
    solver_runtime = t_solver_end - t_solver_start
    print("\nEvolution Finished.")
    pareto_front = fast_non_dominated_sort(population)[0]

    # 浠庡叏灞€瀛樻。涓彁鍙栦袱涓瀬绔В
    # 1. 鎵炬渶灏忔垚鏈В (Min Cost) -> objectives[0] 鏈€灏?
    arch_min_cost = min(pareto_archive, key=lambda e: e['objectives'][0])

    # 2. 鎵炬渶澶ф弧鎰忓害瑙?(Max Sat) -> objectives[1] 鏈€灏?(鍥犱负鏄礋鍊?
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

    # 涓轰簡璋冪敤 print_detailed_schedule锛岄渶瑕佹瀯閫犱竴涓复鏃?Individual 瀵硅薄
    # 鍥犱负 print_detailed_schedule 瑕佹眰鍙傛暟鏄?Individual锛堣闂?.decoded_schedule 鍜?.constraint_violation锛?
    arch_min_cost_ind = Individual(NUM_CUSTOMERS)
    arch_min_cost_ind.decoded_schedule = arch_min_cost['decoded_schedule']
    arch_min_cost_ind.objectives = list(arch_min_cost['objectives'])
    arch_min_cost_ind.constraint_violation = 0.0  # 瀛樻。涓彧淇濆瓨鍙瑙ｏ紝杩濊閲忎负0

    arch_max_sat_ind = Individual(NUM_CUSTOMERS)
    arch_max_sat_ind.decoded_schedule = arch_max_sat['decoded_schedule']
    arch_max_sat_ind.objectives = list(arch_max_sat['objectives'])
    arch_max_sat_ind.constraint_violation = 0.0  # 瀛樻。涓彧淇濆瓨鍙瑙ｏ紝杩濊閲忎负0

    print("\nPrinting detailed schedule for Min Cost Solution:")
    print_detailed_schedule(instance, arch_min_cost_ind)

    print("\nPrinting detailed schedule for Max Sat Solution:")
    print_detailed_schedule(instance, arch_max_sat_ind)

    # dists = [p.objectives[0] for p in pareto_front]
    # sats = [-p.objectives[1] for p in pareto_front]
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(dists, sats, c='red', s=50, label='Pareto Solutions')
    # plt.xlabel('Total Distance (Minimize)')
    # plt.ylabel('Satisfaction (Maximize)')
    # plt.title(f'NSGA-II for TDRP - DTW Mode: {mode_label} (N={NUM_CUSTOMERS})')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # 鎵撳嵃涓€涓紭閫夎В
    # best_sol = pareto_front[0]
    # print("\n--- Sample Pareto Solution ---")
    # print(f"Distance: {best_sol.objectives[0]:.2f}")
    # print(f"Satisfaction: {-best_sol.objectives[1]:.2f}")
    # t_route, d_tasks = best_sol.decoded_schedule
    # print(f"Truck Route: {t_route}")
    # print(f"Drone Sorties (Launch, Serve, Recover): {d_tasks}")
    #
    # # --- 鏂板璋冪敤缁樺浘鍑芥暟 ---
    # print("Plotting best solution...")
    # plot_solution(instance, t_route, d_tasks, obj_vals=best_sol.objectives)

    # --- [淇敼寮€濮媇 鎸囨爣璁＄畻 ---
    pareto_front_pop = fast_non_dominated_sort(population)[0]

    # 1. 浠庡叏灞€瀛樻。涓彁鍙栫洰鏍囧€?(杩欐槸鏈€鍏ㄧ殑)
    archive_objs = [list(entry['objectives']) for entry in pareto_archive]

    # 2. 浠庡綋鍓嶇缇ゆ彁鍙栫洰鏍囧€?(闃叉鏈€鍚庝竴浠ｆ湁鏂板彂鐜版湭鍏ュ簱)
    pop_objs = [list(ind.objectives) for ind in pareto_front_pop]

    # 3. 鍚堝苟骞跺幓閲?
    all_objs = archive_objs + pop_objs
    nsga_front_objs = []
    seen_objs = set()
    for obj in all_objs:
        # 杞负 tuple 浠ヤ究鍝堝笇鍘婚噸
        t_obj = (float(obj[0]), float(obj[1]))
        if t_obj not in seen_objs:
            seen_objs.add(t_obj)
            nsga_front_objs.append(obj)

    # 2. 鍔ㄦ€佸畾涔夊弬鑰冪偣
    if shared_ref_point is not None:
        # [淇敼] 濡傛灉鎻愪緵浜嗗叡浜弬鑰冪偣锛堟潵鑷?Gurobi锛夛紝鍒欎娇鐢ㄥ畠浠ヤ繚璇?HV 鍙瘮鎬?
        print(f"\n[Metric] Using Shared Reference Point from Exact Solver: {shared_ref_point}")
        ref_point_nsga = shared_ref_point
    else:
        # [淇敼] 浠呭湪鏈彁渚涘叡浜弬鑰冪偣鏃讹紝鎵嶄娇鐢ㄨ嚜韬缇ょ殑鏈€宸€间綔涓哄弬鑰冪偣
        max_cost_obs = max(ind.objectives[0] for ind in pareto_front_pop)
        max_neg_sat_obs = max(ind.objectives[1] for ind in pareto_front_pop)
        ref_point_nsga = [max_cost_obs, max_neg_sat_obs]
        print(f"\n[Metric] Using Local Reference Point (Self-Adaptive): {ref_point_nsga}")

    # 3. 璁＄畻鎸囨爣
    # 3. 璁＄畻鎸囨爣
    hv_val, igd_val, sp_val = calculate_metrics(nsga_front_objs, ref_point_nsga, shared_ref_front)

    igd_str = f"{igd_val:.6f}" if igd_val is not None else "N/A"

    print("\n" + "=" * 40)
    print(f"Performance Metrics (NSGA-II) - {mode_label}")
    print("=" * 40)
    print(f"Reference Point        : {ref_point_nsga}")
    print(f"HV  (Hypervolume)      : {hv_val:.4f}")
    print(f"IGD (Inv. Gen. Dist.)  : {igd_str}")
    print(f"SP  (Spacing)          : {sp_val:.6f}")
    print(f"Runtime (s)            : {solver_runtime:.2f}")
    print("=" * 40 + "\n")
    # 杩斿洖 Pareto 鍓嶆部鐨勬渶浼樿В (渚嬪璺濈鏈€鐭殑)
    best_dist_sol = min(pareto_front_pop, key=lambda p: p.objectives[0])
    metrics_summary = {
        'mode': dtw_mode,
        'label': mode_label,
        'hv': hv_val,
        'igd': igd_val,
        'sp': sp_val,
        'runtime': solver_runtime,
        'best_dist': best_dist_sol.objectives[0],
        'best_sat': -min(p.objectives[1] for p in pareto_front_pop),
    }
    return best_dist_sol.objectives[0], best_dist_sol.objectives[1], nsga_front_objs, metrics_summary
def calculate_metrics(pareto_front, reference_point, reference_front=None):
    """
    璁＄畻鎬ц兘鎸囨爣: HV (瓒呬綋绉?, IGD (鍙嶅悜涓栦唬璺濈), Spacing (闂磋窛)
    pareto_front: list of [obj1, obj2] (鍋囪鍧囦负鏈€灏忓寲鏂瑰悜)
    reference_point: [ref_obj1, ref_obj2] (HV 鎵€闇€, 蹇呴』琚墍鏈夎В鏀厤)
    reference_front: list of [obj1, obj2] (IGD 鎵€闇€鐨勫弬鑰冨墠娌? 鍙€?
    """
    # 1. 璁＄畻 HV (Hypervolume) - 2D Minimization
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

    # 2. 璁＄畻 IGD (Inverted Generational Distance)
    # 瀹氫箟: IGD(A, R) = (1/|R|) * 危_{r鈭圧} min_{a鈭圓} dist(r, a)
    # 闇€瑕佸閮ㄥ弬鑰冨墠娌? 鑻ユ湭鎻愪緵鍒欒繑鍥?None
    # 褰掍竴鍖? 浣跨敤鍙傝€冨墠娌胯嚜韬殑鐩爣鏋佸樊, 娑堥櫎閲忕翰宸紓
    igd = None
    if reference_front is not None and len(reference_front) > 0:
        ref_obj0 = [p[0] for p in reference_front]
        ref_obj1 = [p[1] for p in reference_front]
        range_0 = max(ref_obj0) - min(ref_obj0)
        range_1 = max(ref_obj1) - min(ref_obj1)
        if range_0 < 1e-10: range_0 = 1.0  # 闃叉闄ら浂 (鎵€鏈夎В鍦ㄨ鐩爣涓婄浉鍚?
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

    # 3. 璁＄畻 Spacing (Schott, 1995)
    # 瀹氫箟: SP = sqrt( (1/(n-1)) * 危 (d_bar - d_i)^2 )
    # 鍏朵腑 d_i = min_{j鈮爄} L1_normalized(i, j)
    # 鍊艰秺灏? 瑙ｅ垎甯冭秺鍧囧寑
    # 褰掍竴鍖? 浣跨敤鑷韩鍓嶆部鐨勭洰鏍囨瀬宸?
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


def fast_timing_and_satisfaction(instance, truck_route, drone_sorties):
    # 1. 鏁版嵁鍑嗗锛氱紪鐮?Drone Sorties 涓?Numpy 鏁扮粍
    n_sorties = len(drone_sorties)
    if n_sorties == 0:
        encoded_sorties = np.empty((0, 4), dtype=np.int32)
    else:
        encoded_sorties = np.full((n_sorties, 4), -1, dtype=np.int32)
        for i, (l, path, r) in enumerate(drone_sorties):
            encoded_sorties[i, 0] = l
            if len(path) > 0:
                encoded_sorties[i, 1] = path[0]
            if len(path) > 1:
                encoded_sorties[i, 2] = path[1]
            encoded_sorties[i, 3] = r

    route_arr = np.array(truck_route, dtype=np.int32)
    num_nodes = instance.num_nodes

    stw_start = np.zeros(num_nodes, dtype=np.float64)
    stw_end = np.zeros(num_nodes, dtype=np.float64)
    dtw_width_arr = np.zeros(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        if isinstance(instance.stw, dict):
            w = instance.stw.get(i, (0.0, 100000.0))
        else:
            w = instance.stw[i]
        stw_start[i] = w[0]
        stw_end[i] = w[1]

        if isinstance(instance.dtw_width, dict):
            dtw_width_arr[i] = instance.dtw_width.get(i, 0.0)
        else:
            dtw_width_arr[i] = instance.dtw_width[i]

    mode = getattr(instance, 'dtw_mode', 'robust')
    target_service = np.zeros(num_nodes, dtype=np.float64)
    for i in range(num_nodes):
        if mode == 'robust':
            target_service[i] = (stw_start[i] + stw_end[i]) * 0.5
        else:
            target_service[i] = get_optimal_target_by_mode(
                stw_start[i], stw_end[i], dtw_width_arr[i], mode
            )

    total_sat_robust, srv_times, arr_times, viol_time, valid = fast_timing_core_jit(
        num_nodes,
        route_arr,
        encoded_sorties,
        stw_start,
        stw_end,
        dtw_width_arr,
        target_service,
        instance.dist_matrix_truck,
        instance.dist_matrix_drone,
        instance.v_truck,
        instance.v_drone
    )

    if not valid:
        return 0.0, {}, {}, float('inf')

    if mode == 'robust':
        total_sat = total_sat_robust
    else:
        total_sat = 0.0
        for node in range(num_nodes):
            if node == 0:
                continue

            t_service = srv_times[node]
            if t_service < -0.5:
                continue

            tw_start = stw_start[node]
            tw_end = stw_end[node]
            w_i = dtw_width_arr[node]

            if node in instance._dtw_samples_scaled:
                total_sat += compute_satisfaction_by_mode(
                    t_service,
                    tw_start,
                    tw_end,
                    w_i,
                    instance._dtw_samples_scaled[node]
                )
            else:
                total_sat += calc_robust_satisfaction_jit(t_service, tw_start, tw_end, w_i)

    return total_sat, srv_times, arr_times, viol_time

# ==========================================
# 6. 鏂板缁樺浘宸ュ叿鍑芥暟 (淇鐗? 鍖呭惈涓ユ牸鐨?ID 鏃堕棿鍒嗛厤閫昏緫)
# ==========================================
def plot_solution(instance, truck_route, drone_tasks, obj_vals=None):
    """
    鍙鍖栧崱杞﹀拰鏃犱汉鏈虹殑璺緞鏂规 (淇鐗?
    鏍稿績鏀硅繘锛?
    1. 寮曞叆涓ユ牸鐨勬椂闂存帹婕旓紝璁＄畻姣忎釜浠诲姟鐨勭簿纭?Start/End 鏃堕棿銆?
    2. 浣跨敤璐績绛栫暐鏍规嵁鏃堕棿鍒嗛厤 Drone ID锛岀‘淇濋鑹茶繛缁笖澶嶇敤姝ｇ‘銆?
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    plt.figure(figsize=(12, 10))

    # --- 姝ラ A: 棰勮绠楁墍鏈変换鍔＄殑鏃堕棿绐楀彛 ---
    # 鎴戜滑闇€瑕佺煡閬撴瘡涓?sortie 鐨?(launch_time, land_time) 鎵嶈兘鍒嗛厤 ID

    # 1. 寤虹珛鏁版嵁缁撴瀯
    # drone_tasks 鏍煎紡: list of (launch_node, path_nodes, recover_node)
    # 鎴戜滑缁欐瘡涓换鍔′竴涓師濮嬬储寮曪紝浠ヤ究鍚庣画杩借釜
    task_info_map = []  # 瀛?{'idx': i, 'l': l, 'path': p, 'r': r}
    for i, (l, p, r) in enumerate(drone_tasks):
        task_info_map.append({'original_idx': i, 'l': l, 'path': p, 'r': r, 'start_t': -1, 'end_t': -1})

    # 杈呭姪鏌ユ壘锛歭aunch_node -> tasks starting here
    launch_map = {node: [] for node in instance.nodes}
    for item in task_info_map:
        launch_map[item['l']].append(item)

    # 杈呭姪鏌ユ壘锛歳ecover_node -> tasks ending here (鐢ㄤ簬鍗¤溅绛夊緟)
    pending_recoveries = {node: [] for node in instance.nodes}

    # 2. 妯℃嫙鍗¤溅杩愯浠ヨ绠楁椂闂?(涓庤В鐮佸櫒閫昏緫涓€鑷?
    departure_times = {0: 0.0}
    arrival_times = {}

    # 褰撳墠鍗¤溅鍒拌揪鏌愮偣鐨勬椂闂?
    curr_time = 0.0

    for idx, curr_node in enumerate(truck_route):
        # 1. 鍗¤溅鍒拌揪
        if idx == 0:
            arrival_times[curr_node] = 0.0
        else:
            prev_node = truck_route[idx - 1]
            travel_t = instance.get_truck_dist(prev_node, curr_node) / instance.v_truck
            arrival_times[curr_node] = departure_times[prev_node] + travel_t

        truck_arr = arrival_times[curr_node]

        # 2. 绛夊緟鍥炴敹 (Wait for Drones)
        # 蹇呴』绛夊緟鎵€鏈夎鍒掑湪姝ゅ鍥炴敹鐨勬棤浜烘満钀藉湴
        max_drone_arr = truck_arr
        if curr_node in pending_recoveries:
            for rec_item in pending_recoveries[curr_node]:
                # 杩欓噷鐨?rec_item 鍖呭惈璇ヤ换鍔℃渶鍚庝竴璺崇殑椋炶鏃堕棿鍜屽€掓暟绗簩涓偣鐨勬湇鍔＄粨鏉熸椂闂?
                # 鎴戜滑闇€瑕佸湪鍙戝皠鏃跺氨璁＄畻濂借繖浜?
                d_arr = rec_item['land_time']
                if d_arr > max_drone_arr:
                    max_drone_arr = d_arr

                # 鍥炲啓缁撴潫鏃堕棿鍒?task_info
                rec_item['task_ref']['end_t'] = d_arr

        node_ready_time = max_drone_arr

        # 3. 鍗¤溅鏈嶅姟涓庡彂灏?(Service & Launch)
        # 鍙戝皠鏃堕棿 = max(鍒拌揪鏃堕棿, 鍥炴敹瀹屾瘯鏃堕棿, 鑺傜偣鏈€鏃╂湇鍔℃椂闂?
        # 娉ㄦ剰锛氬鏋滄槸 Depot锛屼笉闇€瑕佹湇鍔℃椂闂达紝浣嗕綔涓轰腑闂寸偣鍙兘鏈夐檺鍒?
        srv_start = node_ready_time
        if curr_node != 0:
            stw_start = instance.stw[curr_node][0]
            srv_start = max(node_ready_time, stw_start)
            departure_times[curr_node] = srv_start
        else:
            # Depot 浣滀负缁堢偣鎴栬捣鐐?
            departure_times[curr_node] = node_ready_time

            # 澶勭悊鍦ㄦ澶勫彂灏勭殑浠诲姟
        if curr_node in launch_map:
            launch_base_time = node_ready_time  # 鏃犱汉鏈哄彲浠ュ湪鍗¤溅鍒拌揪/鍥炴敹瀹屾垚鍚庣珛鍗宠捣椋?

            for task in launch_map[curr_node]:
                task['start_t'] = launch_base_time

                # 鎺ㄦ紨璇ヤ换鍔＄殑椋炶涓庢湇鍔¤繃绋嬶紝涓轰簡璁＄畻 land_time
                curr_aerial_t = launch_base_time
                last_srv_end = 0

                # 閬嶅巻鏃犱汉鏈鸿矾寰勭偣
                for step_i, d_node in enumerate(task['path']):
                    prev_loc = curr_node if step_i == 0 else task['path'][step_i - 1]
                    fly_t = instance.get_drone_dist(prev_loc, d_node) / instance.v_drone
                    arr_t = curr_aerial_t + fly_t

                    s_start = max(arr_t, instance.stw[d_node][0])
                    s_end = s_start

                    curr_aerial_t = s_end
                    last_srv_end = s_end

                # 璁＄畻鏈€鍚庝竴娈靛洖绋?
                last_node = task['path'][-1]
                r_node = task['r']
                fly_in = instance.get_drone_dist(last_node, r_node) / instance.v_drone
                land_t = last_srv_end + fly_in

                # 娉ㄥ唽鍒板洖鏀剁偣锛屼互渚垮崱杞﹁绠楃瓑寰?
                if r_node not in pending_recoveries: pending_recoveries[r_node] = []
                pending_recoveries[r_node].append({
                    'land_time': land_t,
                    'task_ref': task
                })

    # --- 姝ラ B: 璐績鍒嗛厤 Drone ID (鍏抽敭淇) ---
    # 1. 鎸夊彂灏勬椂闂存帓搴?
    sorted_tasks = sorted(task_info_map, key=lambda x: x['start_t'])

    # 2. 鍒濆鍖栨棤浜烘満鐘舵€?[free_time_drone_0, free_time_drone_1, ...]
    drone_free_times = [0.0] * instance.NUM_DRONES
    task_id_assignment = {}  # original_idx -> assigned_drone_id

    for task in sorted_tasks:
        s_t = task['start_t']
        e_t = task['end_t']

        assigned_id = -1
        # 浼樺厛鎵剧┖闂茬殑
        for d_id in range(instance.NUM_DRONES):
            # 瀹瑰樊 1e-4 閬垮厤娴偣璇樊
            if drone_free_times[d_id] <= s_t + 1e-4:
                assigned_id = d_id
                drone_free_times[d_id] = e_t
                break

        # 濡傛灉閮藉繖(鐞嗚涓婁笉搴斿彂鐢燂紝濡傛灉閫氳繃浜哾ecode鐨勫彲琛屾€ф鏌?锛屽垯閫夋渶鏃╃粨鏉熺殑閭ｄ釜寮鸿澶嶇敤
        # (杩欏彧鏄负浜嗙敾鍥句笉鎶ラ敊锛屽疄闄呬唬琛ㄨВ鍙兘鏈夌憰鐤碉紝浣嗗湪淇閫昏緫鍚庡簲涔熸槸鍚堟硶鐨?
        if assigned_id == -1:
            best_id = np.argmin(drone_free_times)
            assigned_id = best_id
            drone_free_times[best_id] = e_t

        task_id_assignment[task['original_idx']] = assigned_id

    # --- 姝ラ C: 寮€濮嬬粯鍥?---

    # 瀹氫箟鏍峰紡
    drone_styles = {
        0: {'color': 'tab:red', 'ls': '--', 'label': 'Drone #1'},
        1: {'color': 'tab:green', 'ls': '-.', 'label': 'Drone #2'},
        2: {'color': 'tab:blue', 'ls': ':'},
    }

    legend_added = set()

    # 1. 缁樺埗鏃犱汉鏈?
    for item in task_info_map:
        d_id = task_id_assignment[item['original_idx']]
        # 鍙湁鍓?鏋舵棤浜烘満鏈夊浐瀹氭牱寮忥紝瓒呭嚭鐨?濡傛灉鍑洪敊)鐢ㄩ粯璁?
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

    # 2. 缁樺埗鍗¤溅
    truck_coords = np.array([instance.coords[n] for n in truck_route])
    plt.plot(truck_coords[:, 0], truck_coords[:, 1], color='black', linewidth=3,
             alpha=0.6, label='Truck', zorder=1)
    # Truck Arrows
    for k in range(len(truck_coords) - 1):
        p1, p2 = truck_coords[k], truck_coords[k + 1]
        plt.arrow(p1[0], p1[1], (p2[0] - p1[0]) * 0.55, (p2[1] - p1[1]) * 0.55,
                  head_width=0.3, color='black', alpha=0.3, zorder=1)

    # 3. 缁樺埗鑺傜偣
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

    # 4. 瑁呴グ
    title = "Improved NSGA-II Solution Visualization"
    if obj_vals: title += f"\nDist: {obj_vals[0]:.2f} | Sat: {-obj_vals[1]:.2f}"
    plt.title(title)

    # 鏋勯€犲畬鏁村浘渚?
    handles, labels = plt.gca().get_legend_handles_labels()
    # 鍘婚噸
    by_label = dict(zip(labels, handles))
    # 娣诲姞瀹㈡埛绫诲瀷鍥句緥
    by_label['Type D'] = Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10)
    by_label['Type P'] = Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10)

    plt.legend(by_label.values(), by_label.keys(), loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_detailed_schedule(instance, individual):
    if not individual.decoded_schedule:
        print("No decoded schedule available.")
        return

    truck_route, drone_sorties = individual.decoded_schedule

    total_sat, service_times, arrival_times, total_violation_time = fast_timing_and_satisfaction(
        instance, truck_route, drone_sorties
    )

    is_feasible = (total_violation_time < 1e-6)
    feasibility_status = "Feasible" if is_feasible else f"Infeasible ({total_violation_time:.2f}s late)"

    print("\n" + "=" * 60)
    print("      DETAILED SCHEDULE ANALYSIS")
    print("=" * 60)
    print(f"Total Satisfaction: {total_sat:.4f}")
    print(f"Feasibility       : {feasibility_status}")
    print(f"Constraint Viol.  : {individual.constraint_violation:.2f} s")
    print(f"Truck Route       : {truck_route}")
    print(f"Drone Sorties     : {drone_sorties}")

    involved_nodes = set(truck_route)
    for (_, p_nodes, _) in drone_sorties:
        for n in p_nodes:
            involved_nodes.add(n)

    print("\nNode   Type    Arrival    Service    Sat")
    print("-" * 50)
    for node in sorted(involved_nodes):
        arr = arrival_times[node] if node < len(arrival_times) and arrival_times[node] >= -0.5 else instance.stw[node][0]
        srv = service_times[node] if node < len(service_times) and service_times[node] >= -0.5 else instance.stw[node][0]

        if node == 0:
            sat = 0.0
            node_type = "Depot"
        else:
            tw_start = instance.stw[node][0]
            tw_end = instance.stw[node][1]
            w_i = instance.dtw_width[node] if not isinstance(instance.dtw_width, dict) else instance.dtw_width.get(node, 0.0)

            if getattr(instance, 'dtw_mode', 'robust') != 'robust' and node in getattr(instance, '_dtw_samples_scaled', {}):
                sat = compute_satisfaction_by_mode(srv, tw_start, tw_end, w_i, instance._dtw_samples_scaled[node])
            else:
                sat = calc_robust_satisfaction_jit(srv, tw_start, tw_end, w_i)

            node_type = "Truck" if node in truck_route else "Drone"

        print(f"{node:<6}{node_type:<8}{arr:>8.2f}{srv:>11.2f}{sat:>9.4f}")
    print("-" * 50)
if __name__ == "__main__":
    common_data = data_loader.load_solomon_data(
        'solomon标准算例-时间窗/c1/c101.txt',
        n_customers=40,
        random_seed=42
    )
    # print(common_data)

    modes_to_run = ['centered', 'early', 'late', 'uniform']#
    all_results = {}

    t_start_g = time.time()

    for mode in modes_to_run:
        print(f"\n{'#' * 70}")
        print(f"#  Running DTW Mode: {DTW_MODES[mode]['label']}")
        print(f"{'#' * 70}\n")

        best_dist, best_sat, front_objs, metrics = run_nsga_solver(
            common_data,
            dtw_mode=mode
        )
        all_results[mode] = {
            'front_objs': front_objs,
            'metrics': metrics
        }

    t_end_g = time.time()
    print(f"\nTotal run time for all modes: {t_end_g - t_start_g:.2f} s")

    print("\n" + "=" * 90)
    print("  COMPARISON SUMMARY - 4 DTW Position Preferences")
    print("=" * 90)
    print(f"{'Mode':<25} {'HV':>12} {'IGD':>12} {'SP':>12} {'Best Dist':>12} {'Best Sat':>12} {'Time(s)':>10}")
    print("-" * 90)

    for mode in modes_to_run:
        m = all_results[mode]['metrics']
        igd_str = f"{m['igd']:.6f}" if m['igd'] is not None else "N/A"
        print(
            f"{m['label']:<25} {m['hv']:>12.4f} {igd_str:>12} {m['sp']:>12.6f} "
            f"{m['best_dist']:>12.2f} {m['best_sat']:>12.2f} {m['runtime']:>10.2f}"
        )

    print("=" * 90)

    plt.figure(figsize=(12, 8))
    colors = {'centered': 'blue', 'early': 'green', 'late': 'red', 'uniform': 'orange'}
    markers = {'centered': 'o', 'early': '^', 'late': 's', 'uniform': 'D'}

    for mode in modes_to_run:
        front = all_results[mode]['front_objs']
        dists = [p[0] for p in front]
        sats = [-p[1] for p in front]
        label = DTW_MODES[mode]['label']
        plt.scatter(
            dists,
            sats,
            c=colors[mode],
            marker=markers[mode],
            s=60,
            alpha=0.7,
            edgecolors='k',
            linewidths=0.5,
            label=label
        )

    plt.xlabel('Total Distance (Minimize)', fontsize=12)
    plt.ylabel('Satisfaction (Maximize)', fontsize=12)
    plt.title('Pareto Front Comparison - 4 DTW Position Preferences', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dtw_modes_pareto_comparison.png', dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mode_labels = [DTW_MODES[m]['label'] for m in modes_to_run]
    bar_colors = [colors[m] for m in modes_to_run]

    hv_vals = [all_results[m]['metrics']['hv'] for m in modes_to_run]
    axes[0].bar(mode_labels, hv_vals, color=bar_colors, edgecolor='k')
    axes[0].set_title('Hypervolume (HV)', fontsize=12)
    axes[0].set_ylabel('HV')
    axes[0].tick_params(axis='x', rotation=20)

    sp_vals = [all_results[m]['metrics']['sp'] for m in modes_to_run]
    axes[1].bar(mode_labels, sp_vals, color=bar_colors, edgecolor='k')
    axes[1].set_title('Spacing (SP)', fontsize=12)
    axes[1].set_ylabel('SP')
    axes[1].tick_params(axis='x', rotation=20)

    rt_vals = [all_results[m]['metrics']['runtime'] for m in modes_to_run]
    axes[2].bar(mode_labels, rt_vals, color=bar_colors, edgecolor='k')
    axes[2].set_title('Runtime (s)', fontsize=12)
    axes[2].set_ylabel('Time (s)')
    axes[2].tick_params(axis='x', rotation=20)

    plt.suptitle('Performance Metrics Comparison - 4 DTW Modes', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('dtw_modes_metrics_comparison.png', dpi=150)
    plt.show()





