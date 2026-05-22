import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import data_loader
import time


# --- 1. 绘图函数 (保持与 gurobi_sat 一致的逻辑，支持虚拟终点) ---
def plot_route_scheme(nodes, coords, x_vals, y_vals, z_vals, y_indices, z_indices, cust_types, title_suffix=""):
    """
    绘制具体的卡车和无人机路径方案
    """
    if not x_vals: return

    plt.figure(figsize=(10, 8))

    # 绘制节点 (注意：nodes 包含虚拟终点，但绘图时只需画一次 Depot)
    # 假设 nodes 列表最后一个是虚拟终点
    real_nodes = [n for n in nodes if n != nodes[-1]]  # 排除虚拟终点用于绘图循环(避免重叠绘制)
    start_depot = nodes[0]

    # 画起点 Depot
    cx, cy = coords[start_depot]
    plt.scatter(cx, cy, c='black', marker='s', s=150, zorder=10, label='Depot')
    plt.text(cx, cy + 1, "Depot", fontsize=10, ha='center')

    # 画客户
    for i in real_nodes:
        if i == start_depot: continue
        cx, cy = coords[i]
        ctype = cust_types[i]
        c_color = 'skyblue' if ctype == 'D' else ('orange' if ctype == 'P' else 'orchid')
        c_mark = 'o' if ctype == 'D' else ('^' if ctype == 'P' else 'D')
        plt.scatter(cx, cy, c=c_color, marker=c_mark, s=100, edgecolors='k', zorder=5)
        plt.text(cx, cy + 1, f"C{i}({ctype})", fontsize=8, ha='center')

    # 2. 绘制卡车路径 (黑色实线)
    for (i, j), val in x_vals.items():
        if val > 0.5:
            p1, p2 = coords[i], coords[j]
            plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                      color='black', width=0.02, length_includes_head=True, zorder=2, alpha=0.7)

    # 3. 绘制无人机路径 (彩色虚线)
    drone_style = dict(linestyle='--', alpha=0.8, linewidth=1.5)

    # 1-Stop
    for (i, j, k), val in y_vals.items():
        if val > 0.5:
            plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], color='red', **drone_style)
            plt.plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]], color='red', **drone_style)
            plt.text(coords[j][0], coords[j][1] - 2, "1-Stop", color='red', fontsize=7, ha='center')

    # 2-Stop
    for (i, j, l, k), val in z_vals.items():
        if val > 0.5:
            path = [coords[i], coords[j], coords[l], coords[k]]
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            plt.plot(px, py, color='green', **drone_style)
            plt.text(coords[j][0], coords[j][1] - 2, "2-Stop", color='green', fontsize=7, ha='center')

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=2),
                    Line2D([0], [0], color='red', linestyle='--', lw=1.5),
                    Line2D([0], [0], color='green', linestyle='--', lw=1.5)]
    plt.legend(custom_lines, ['Truck', 'Drone (1-Stop)', 'Drone (2-Stop)'], loc='best')

    plt.title(f"Route Visualization {title_suffix}", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pareto_frontier(solutions):
    """
    绘制帕累托前沿
    solutions: list of tuples (satisfaction, cost)
    """
    if not solutions:
        print("No solutions to plot.")
        return

    sats, costs = zip(*solutions)

    plt.figure(figsize=(10, 6))
    plt.plot(sats, costs, 'o-', color='darkblue', markersize=8, linewidth=2, label='Pareto Front')

    for s, c in solutions:
        plt.text(s, c + 2, f"({s:.1f}, {c:.0f})", fontsize=9, ha='center')

    plt.title("Pareto Frontier: Cost vs. Satisfaction", fontsize=14)
    plt.xlabel("Total Satisfaction (Constraint)", fontsize=12)
    plt.ylabel("Total Cost (Objective)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


# --- 2. 模型构建器 (已同步 gurobi_sat 的逻辑) ---
def build_fstsp_model(nodes, coords, STW_start, STW_end, cust_types):
    # [同步] 1. 定义虚拟终点仓库
    start_depot = 0
    end_depot = nodes[-1] + 1

    # [同步] 数据拷贝与隔离
    coords = coords.copy()
    STW_start = STW_start.copy()
    STW_end = STW_end.copy()
    cust_types = cust_types.copy()

    # 构建扩展后的节点列表
    all_nodes = nodes + [end_depot]

    # [同步] 2. 复制终点仓库的属性
    coords[end_depot] = coords[start_depot]
    STW_start[end_depot] = STW_start[start_depot]
    STW_end[end_depot] = STW_end[start_depot]
    cust_types[end_depot] = 'D'

    NUM_DRONES = 2

    # [同步] 3. 计算距离矩阵 (包含虚拟终点)
    matrix_size = len(all_nodes)
    d_T = np.zeros((matrix_size, matrix_size))
    d_D = np.zeros((matrix_size, matrix_size))

    for i in all_nodes:
        for j in all_nodes:
            d_T[i, j] = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            d_D[i, j] = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)

    Truck_Speed = 1.0;
    Drone_Speed = 2.0;
    M = 10000;
    DTW_RATIO = 0.5
    Cost_Truck = 1.0
    Cost_Drone = 1

    m = gp.Model("FSTSP_Epsilon_Dis")
    m.params.OutputFlag = 0
    m.params.TimeLimit = 360

    customers = [i for i in nodes if i != 0]

    # [同步] 变量定义范围扩展到 all_nodes
    x = m.addVars(all_nodes, all_nodes, vtype=GRB.BINARY, name="x")
    h = m.addVars(nodes, vtype=GRB.BINARY, name="h")  # h 仅需覆盖原节点
    u = m.addVars(customers, vtype=GRB.BINARY, name="u")
    p = m.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=NUM_DRONES, name="p")
    a = m.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0, name="a")
    t = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, name="t")
    alpha = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sat")

    # [同步] Indices 逻辑
    y_indices = []
    for i in all_nodes:
        if i == end_depot: continue  # 终点不能发射
        for j in customers:
            if i == j: continue
            for k in all_nodes:
                if k == start_depot: continue  # 起点不能回收
                if k == j or k == i: continue
                y_indices.append((i, j, k))

    z_indices = []
    for i in all_nodes:
        if i == end_depot: continue
        for j in customers:
            if cust_types[j] != 'D': continue
            for l in customers:
                if l == j or cust_types[l] != 'P': continue
                for k in all_nodes:
                    if k == start_depot: continue
                    if k != j and k != l and k != i:
                        z_indices.append((i, j, l, k))

    y = m.addVars(y_indices, vtype=GRB.BINARY, name="y")
    z = m.addVars(z_indices, vtype=GRB.BINARY, name="z")

    # 1. Total Cost
    cost_truck = gp.quicksum(Cost_Truck * d_T[i, j] * x[i, j] for i in all_nodes for j in all_nodes)
    cost_y = gp.quicksum(Cost_Drone * (d_D[i, j] + d_D[j, k]) * y[i, j, k] for i, j, k in y_indices)
    cost_z = gp.quicksum(Cost_Drone * (d_D[i, j] + d_D[j, l] + d_D[l, k]) * z[i, j, l, k] for i, j, l, k in z_indices)
    expr_total_cost = cost_truck + cost_y + cost_z

    # 2. Total Satisfaction
    expr_total_sat = gp.quicksum(alpha[c] for c in customers)

    # --- [同步] 约束条件 ---

    # 卡车流：Start出, End入
    m.addConstr(gp.quicksum(x[start_depot, j] for j in all_nodes if j != start_depot) == 1)
    m.addConstr(gp.quicksum(x[j, end_depot] for j in all_nodes if j != end_depot) == 1)
    m.addConstr(gp.quicksum(x[j, start_depot] for j in all_nodes) == 0)
    m.addConstr(gp.quicksum(x[end_depot, j] for j in all_nodes) == 0)

    # 客户流平衡
    for i in customers:
        m.addConstr(gp.quicksum(x[j, i] for j in all_nodes if j != i) == u[i])
        m.addConstr(gp.quicksum(x[i, j] for j in all_nodes if j != i) == u[i])
        m.addConstr(x[i, i] == 0)

    # 服务关系
    for c in customers:
        sy = gp.quicksum(y[i, c, k] for i, j, k in y_indices if j == c)
        sz1 = gp.quicksum(z[i, c, l, k] for i, j, l, k in z_indices if j == c)
        sz2 = gp.quicksum(z[i, j, c, k] for i, j, l, k in z_indices if l == c)
        m.addConstr(u[c] + sy + sz1 + sz2 == 1)
        # m.addConstr(u[c] <= h[c])

        # [同步] Visit Necessity 约束
        # 逻辑修复：防止“隔空发射/回收”。
        # 如果节点 c 是发射点 (Launch) 或回收点 (Recover)，则卡车必须访问该节点。
        # 在当前模型中，卡车访问即意味着 u[c]=1 (由流约束 sum(x)==u 决定)。

        # 1. 统计以 c 为发射点的所有任务 (y 和 z)
        # 注意 y_indices = (i, j, k), z_indices = (i, j, l, k)。 i 是发射点。
        launch_ops = gp.quicksum(y[c, j, k] for i, j, k in y_indices if i == c) + \
                     gp.quicksum(z[c, j, l, k] for i, j, l, k in z_indices if i == c)

        # 2. 统计以 c 为回收点的所有任务 (y 和 z)
        # k 是回收点。
        recover_ops = gp.quicksum(y[i, j, c] for i, j, k in y_indices if k == c) + \
                      gp.quicksum(z[i, j, l, c] for i, j, l, k in z_indices if k == c)

        # 3. 强制同步约束：如果发生发射或回收，则 u[c] 必须为 1
        # 使用已定义的常量 M (M=10000)
        # 逻辑：(Launch + Recover) <= M * u[c]
        # 若左边 > 0，则 u[c] 必须 >= 1/M，即 u[c]=1。
        m.addConstr(launch_ops + recover_ops <= M * u[c], name=f"Sync_Truck_Visit_{c}")

    m.addConstr(p[0] == NUM_DRONES)

    # [同步] 库存与流平衡 (排除 end_depot)
    process_nodes = [n for n in all_nodes if n != end_depot]
    for i in process_nodes:
        L = gp.quicksum(y[i, j, k] for s, j, k in y_indices if s == i) + gp.quicksum(
            z[i, j, l, k] for s, j, l, k in z_indices if s == i)
        R = gp.quicksum(y[k, j, i] for k, j, e in y_indices if e == i) + gp.quicksum(
            z[k, j, l, i] for k, j, l, e in z_indices if e == i)

        if i != start_depot:
            m.addConstr(L <= M * u[i])
            m.addConstr(R <= M * u[i])
        m.addConstr(p[i] + R <= NUM_DRONES)

        for j in all_nodes:
            if i != j and j != start_depot:
                m.addConstr(p[j] >= p[i] - L + R - M * (1 - x[i, j]))
                m.addConstr(p[j] <= p[i] - L + R + M * (1 - x[i, j]))

    # [同步] 仓库发射/回收限制
    dL = gp.quicksum(y[start_depot, j, k] for s, j, k in y_indices if s == start_depot) + gp.quicksum(
        z[start_depot, j, l, k] for s, j, l, k in z_indices if s == start_depot)
    dR = gp.quicksum(y[i, j, end_depot] for i, j, e in y_indices if e == end_depot) + gp.quicksum(
        z[i, j, l, end_depot] for i, j, l, e in z_indices if e == end_depot)
    m.addConstr(dL <= NUM_DRONES)
    m.addConstr(dR <= NUM_DRONES)

    m.addConstr(a[0] >= 0)

    # [同步] 时间约束 (Process Nodes -> All Nodes)
    for i in process_nodes:
        for j in all_nodes:
            if i != j and j != start_depot:
                if i == start_depot:
                    dep_time = a[start_depot]
                else:
                    dep_time = t[i]
                m.addConstr(a[j] >= dep_time + d_T[i, j] / Truck_Speed - M * (1 - x[i, j]))

    for i, j, k in y_indices:
        m.addConstr(t[j] >= a[i] + d_D[i, j] / Drone_Speed - M * (1 - y[i, j, k]))

        if k != end_depot:
            m.addConstr(a[k] >= t[j] + d_D[j, k] / Drone_Speed - M * (1 - y[i, j, k]))
        else:
            m.addConstr(a[k] >= t[j] + d_D[j, k] / Drone_Speed - M * (1 - y[i, j, k]))

        if i == start_depot:
            dep_time = a[start_depot]
        else:
            dep_time = t[i]
        m.addConstr(a[k] >= dep_time + d_T[i, k] / Truck_Speed - M * (1 - y[i, j, k]))

    for i, j, l, k in z_indices:
        m.addConstr(t[j] >= a[i] + d_D[i, j] / Drone_Speed - M * (1 - z[i, j, l, k]))
        m.addConstr(t[l] >= t[j] + d_D[j, l] / Drone_Speed - M * (1 - z[i, j, l, k]))

        m.addConstr(a[k] >= t[l] + d_D[l, k] / Drone_Speed - M * (1 - z[i, j, l, k]))

        if i == start_depot:
            dep_time = a[start_depot]
        else:
            dep_time = t[i]
        m.addConstr(a[k] >= dep_time + d_T[i, k] / Truck_Speed - M * (1 - z[i, j, l, k]))

    # [同步] STW 基础约束
    for c in customers:
        m.addConstr(t[c] >= a[c] - M * (1 - u[c]))
        m.addConstr(t[c] >= STW_start[c])
        m.addConstr(t[c] <= STW_end[c])

    # [同步] 鲁棒满意度约束
    for c in customers:
        S, E = STW_start[c], STW_end[c]
        STW_width = E - S
        w_i = STW_width * DTW_RATIO
        decay_gap = STW_width - w_i
        if decay_gap > 1e-4:
            m.addConstr(alpha[c] <= (E - t[c]) / decay_gap, name=f"Robust_Left_{c}")
            m.addConstr(alpha[c] <= (t[c] - S) / decay_gap, name=f"Robust_Right_{c}")

    # 返回扩展后的数据结构，供后续使用
    return m, expr_total_cost, expr_total_sat, x, y, z, a, t, alpha, y_indices, z_indices, all_nodes, coords, cust_types


def run_gurobi_solver(common_data):
    # 1. 使用适配器转换数据
    nodes, coords, STW_start, STW_end, cust_types = data_loader.adapter_for_gurobi(common_data)

    # 2. 建立增强版模型
    print("Building Model (Updated Topology)...")
    m, expr_cost, expr_sat, x, y, z, a, t, alpha, y_indices, z_indices, ext_nodes, ext_coords, ext_types = build_fstsp_model(
        nodes, coords, STW_start, STW_end, cust_types)

    # --- Phase 1: Determine Bounds (略，保持不变) ---
    print("\n[Phase 1] Determining Bounds...")
    m.setObjective(expr_sat, GRB.MAXIMIZE)
    m.optimize()
    max_sat = m.objVal
    cost_at_max_sat = expr_cost.getValue()
    print(f"     Max Sat: {max_sat:.2f} (Cost: {cost_at_max_sat:.2f})")

    m.setObjective(expr_cost, GRB.MINIMIZE)
    m.optimize()
    min_cost = m.objVal
    sat_at_min_cost = expr_sat.getValue()
    print(f"     Min Cost: {min_cost:.2f} (Sat: {sat_at_min_cost:.2f})")

    # --- Phase 2: Epsilon Loop ---
    print("\n[Phase 2] Generating Pareto Front (Min Cost s.t. Sat >= eps)...")

    min_cost_sol = None
    min_cost_val = float('inf')

    GRID_POINTS = 8
    eps_values = np.linspace(sat_at_min_cost, max_sat, GRID_POINTS)
    solutions = []

    # 设置主目标：Min Cost
    m.setObjective(expr_cost, GRB.MINIMIZE)
    # 添加约束 Sat >= 0 (动态更新)
    eps_constr = m.addConstr(expr_sat >= 0, name="Epsilon_Constraint_Sat")

    print(f"{'Iter':<5} {'Epsilon (Min Sat)':<20} {'Obj (Cost)':<15} {'Actual Sat':<15}")
    print("-" * 60)

    for i, eps in enumerate(eps_values):
        eps_constr.RHS = eps
        m.optimize()

        if m.status == GRB.OPTIMAL:
            curr_sat = expr_sat.getValue()
            curr_cost = expr_cost.getValue()
            solutions.append((curr_sat, curr_cost))
            print(f"{i + 1:<5} {eps:<20.4f} {curr_cost:<15.2f} {curr_sat:<15.4f}")

            # [修改] 优选逻辑：优先选成本低的；成本相同选满意度高的
            is_better = False
            if curr_cost < min_cost_val - 1e-4:
                is_better = True
            elif abs(curr_cost - min_cost_val) < 1e-4:
                if min_cost_sol is None or curr_sat > min_cost_sol['sat']:
                    is_better = True

            if is_better:
                min_cost_val = curr_cost
                min_cost_sol = {
                    'x': m.getAttr('x', x),
                    'y': m.getAttr('x', y),
                    'z': m.getAttr('x', z),
                    # [新增] 抓取时间变量和满意度变量
                    'a': m.getAttr('x', a),
                    't': m.getAttr('x', t),
                    'alpha': m.getAttr('x', alpha),  # <--- 关键：抓取每个客户的满意度
                    'sat': curr_sat,
                    'cost': curr_cost
                }
        else:
            print(f"{i + 1:<5} {eps:<20.4f} {'Infeasible':<15} -")

    # --- Phase 3: Output & Plot ---
    if min_cost_sol:
        print("\n" + "-" * 60)
        print(f"Final Solution Route Scheme (Sat={min_cost_sol['sat']:.4f}, Cost={min_cost_sol['cost']:.2f}):")
        print("  (Displaying the solution with the Global Minimum Cost found)")

        # 1. 打印路径边
        truck_edges = [k for k, v in min_cost_sol['x'].items() if v > 0.5]
        print(f"  Truck Edges (i -> j): {truck_edges}")

        drone_1 = [k for k, v in min_cost_sol['y'].items() if v > 0.5]
        print(f"  Drone 1-Stop: {drone_1}")

        if 'z' in min_cost_sol:
            drone_2 = [k for k, v in min_cost_sol['z'].items() if v > 0.5]
            print(f"  Drone 2-Stop: {drone_2}")

        # 2. [重构] 详细时间表打印 (包含无人机到达时间推算 和 个体满意度)
        print("\n  Time Schedule & Satisfaction Details:")
        print(f"    {'Node':<6} {'Type':<8} {'Arrival':<12} {'Service':<12} {'Sat (Alpha)':<12}")
        print("    " + "-" * 50)

        # 辅助函数：计算两点距离
        def get_dist(n1, n2):
            c1, c2 = ext_coords[n1], ext_coords[n2]
            return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

        # 构建无人机父节点映射表: child -> (parent, arrival_time)
        # 必须手动推算，因为 Gurobi 对无人机节点的 'a' 变量约束较松，可能显示为 0
        drone_arrival_map = {}
        Drone_Speed = 2.0  # 硬编码或从外部传入，需保持一致

        # 解析 1-Stop (y): i -> j -> k
        for (i, j, k) in drone_1:
            # i 是发射点。Launch Time 通常取决于 Truck 在 i 的 Service Start (t[i]) 或 Arrival (a[i])
            # 对于 Depot 0，用 a[0]；对于客户，用 t[i]
            launch_time = min_cost_sol['a'][i] #if i == 0 else min_cost_sol['t'][i]
            fly_time = get_dist(i, j) / Drone_Speed
            drone_arrival_map[j] = launch_time + fly_time

        # 解析 2-Stop (z): i -> j -> l -> k
        if 'z' in min_cost_sol:
            for (i, j, l, k) in drone_2:
                # i -> j
                launch_time = min_cost_sol['a'][i] #if i == 0 else min_cost_sol['t'][i]
                arr_j = launch_time + (get_dist(i, j) / Drone_Speed)
                drone_arrival_map[j] = arr_j

                # j -> l
                # 假设在 j 服务完立即起飞 (Service Start j + Service Duration(0) + Flight)
                # j 的起飞时间 = t[j]
                arr_l = min_cost_sol['t'][j] + (get_dist(j, l) / Drone_Speed)
                drone_arrival_map[l] = arr_l

        # 遍历所有节点打印
        all_node_ids = sorted(min_cost_sol['a'].keys())
        for n in all_node_ids:
            # 跳过虚拟终点，或者打印为 End Depot
            if n == ext_nodes[-1]:
                # 终点
                arr_val = min_cost_sol['a'][n]
                print(f"    {n:<6} {'Depot':<8} {arr_val:<12.2f} {'-':<12} {'-':<12}")
                continue

            # 获取 Service Time (t)
            if n in min_cost_sol['t']:
                srv_val = min_cost_sol['t'][n]
                srv_str = f"{srv_val:.2f}"
            else:
                srv_val = None
                srv_str = "-"

            # 获取 Satisfaction (alpha)
            if n in min_cost_sol['alpha']:
                sat_val = min_cost_sol['alpha'][n]
                sat_str = f"{sat_val:.4f}"
            else:
                sat_str = "-"

            # 获取 Arrival Time (a)
            # 如果在 drone_arrival_map 中，说明是无人机服务的节点，优先使用推算值
            # 否则使用 Gurobi 的变量 a[n] (卡车到达)
            if n in drone_arrival_map:
                arr_val = drone_arrival_map[n]
                node_type = "Drone"
            else:
                arr_val = min_cost_sol['a'][n]
                node_type = "Truck" if n != 0 else "Depot"

            print(f"    {n:<6} {node_type:<8} {arr_val:<12.2f} {srv_str:<12} {sat_str:<12}")

        print("-" * 60)

    # 计算指标 & 绘图 (略，保持不变)
    normalized_front = []
    for s_val, c_val in solutions:
        normalized_front.append([c_val, -s_val])

    # 使用 Phase 1 极值作为参考点
    ref_point_gurobi = [cost_at_max_sat, -sat_at_min_cost]
    qp_val, hv_val = calculate_metrics(normalized_front, ref_point_gurobi)

    print("\n" + "=" * 30)
    print("Performance Metrics (Gurobi)")
    print("=" * 30)
    print(f"Reference Point : {ref_point_gurobi}")
    print(f"QP (Quantity)   : {qp_val}")
    print(f"HV (Hypervolume): {hv_val:.4f}")
    print("=" * 30 + "\n")

    # if min_cost_sol:
    #     plot_route_scheme(
    #         ext_nodes, ext_coords,
    #         min_cost_sol['x'], min_cost_sol['y'], min_cost_sol['z'],
    #         y_indices, z_indices, ext_types,
    #         title_suffix=f"(Sat={min_cost_sol['sat']:.1f}, Cost={min_cost_sol['cost']:.0f})"
    #     )
    return solutions, ref_point_gurobi


def calculate_metrics(pareto_front, reference_point):
    """
    计算非支配解集数量 (QP) 和超体积 (HV)
    """
    qp = len(pareto_front)
    # 按 Cost 从小到大排序
    sorted_front = sorted(pareto_front, key=lambda x: x[0])

    hv = 0.0
    ref_x, ref_y = reference_point

    for i in range(len(sorted_front)):
        curr_x, curr_y = sorted_front[i]
        if curr_x > ref_x or curr_y > ref_y: continue

        if i < len(sorted_front) - 1:
            next_x = sorted_front[i + 1][0]
            width = next_x - curr_x
        else:
            width = ref_x - curr_x

        height = ref_y - curr_y
        if width > 0 and height > 0:
            hv += width * height

    return qp, hv


if __name__ == "__main__":
    # 注意：确保 data_loader 已正确实现 adapter_for_gurobi
    common_data = data_loader.load_solomon_data('solomon标准算例-时间窗/r1/r101.txt', n_customers=6, random_seed=42)
    run_gurobi_solver(common_data)