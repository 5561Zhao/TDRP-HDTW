import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import data_loader
import time


# --- 新增：路径方案绘图函数 ---
def plot_route_scheme(nodes, coords, x_vals, y_vals, z_vals, y_indices, z_indices, cust_types, title_suffix=""):
    """
    绘制具体的卡车和无人机路径方案
    """
    if not x_vals: return

    plt.figure(figsize=(10, 8))

    # 1. 绘制节点
    for i in nodes:
        cx, cy = coords[i]
        if i == 0:
            plt.scatter(cx, cy, c='black', marker='s', s=150, zorder=10, label='Depot')
            plt.text(cx, cy + 1, "Depot", fontsize=10, ha='center')
        else:
            ctype = cust_types[i]
            # 根据类型分配颜色: D=Blue, P=Orange, SPD=Purple
            c_color = 'skyblue' if ctype == 'D' else ('orange' if ctype == 'P' else 'orchid')
            c_mark = 'o' if ctype == 'D' else ('^' if ctype == 'P' else 'D')
            plt.scatter(cx, cy, c=c_color, marker=c_mark, s=100, edgecolors='k', zorder=5)
            plt.text(cx, cy + 1, f"C{i}({ctype})", fontsize=8, ha='center')

    # 2. 绘制卡车路径 (黑色实线)
    # x_vals 是字典: {(i,j): 0.0 or 1.0}
    for (i, j), val in x_vals.items():
        if val > 0.5:
            p1, p2 = coords[i], coords[j]
            plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                      color='black', width=0.02, length_includes_head=True, zorder=2, alpha=0.7)

    # 3. 绘制无人机路径 (彩色虚线)
    # 单任务 y: i -> j -> k
    drone_style = dict(linestyle='--', alpha=0.8, linewidth=1.5)
    for (i, j, k), val in y_vals.items():
        if val > 0.5:
            # i->j
            plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], color='red', **drone_style)
            # j->k
            plt.plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]], color='red', **drone_style)
            # 标记
            plt.text(coords[j][0], coords[j][1] - 2, "1-Stop", color='red', fontsize=7, ha='center')

    # 双任务 z: i -> j -> l -> k
    for (i, j, l, k), val in z_vals.items():
        if val > 0.5:
            # i->j->l->k
            path = [coords[i], coords[j], coords[l], coords[k]]
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            plt.plot(px, py, color='green', **drone_style)
            plt.text(coords[j][0], coords[j][1] - 2, "2-Stop", color='green', fontsize=7, ha='center')

    # 图例构造
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

# --- 2. 绘图函数 (保持不变) ---
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

    # 标注点
    for s, c in solutions:
        plt.text(s, c + 2, f"({s:.1f}, {c:.0f})", fontsize=9, ha='center')

    plt.title("Pareto Frontier: Cost vs. Satisfaction", fontsize=14)
    plt.xlabel("Total Satisfaction (Constraint)", fontsize=12)
    plt.ylabel("Total Cost (Objective)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


# --- 3. 模型构建器 (只建模型，不求解) ---
def build_fstsp_model(nodes, coords, STW_start, STW_end, cust_types):
    # [新增] 1. 定义虚拟终点仓库
    # 假设 nodes 是 [0, 1, 2, ... N]，新节点 ID 为 N+1
    start_depot = 0
    end_depot = nodes[-1] + 1

    # [关键修复] 数据隔离：对字典进行浅拷贝 (.copy())
    # 必须这样做，否则会污染 main.py 中的 common_data，导致后续 NSGA 算法报错
    coords = coords.copy()
    STW_start = STW_start.copy()
    STW_end = STW_end.copy()
    cust_types = cust_types.copy()

    # 构建扩展后的节点列表 (列表相加会生成新列表，本身就是安全的)
    all_nodes = nodes + [end_depot]

    # [新增] 2. 复制终点仓库的属性 (坐标、时间窗)
    # 现在修改的是本地副本，不会影响外部
    coords[end_depot] = coords[start_depot]
    STW_start[end_depot] = STW_start[start_depot]
    STW_end[end_depot] = STW_end[start_depot]
    cust_types[end_depot] = 'D'

    NUM_DRONES = 2

    # [修改] 3. 基于扩展节点列表计算距离矩阵
    # 矩阵维度增加 1
    matrix_size = len(all_nodes)
    d_T = np.zeros((matrix_size, matrix_size))
    d_D = np.zeros((matrix_size, matrix_size))

    # 注意：这里遍历的是 all_nodes 而不是原 nodes
    for i in all_nodes:
        for j in all_nodes:
            # 使用 extend 后的 nodes 索引访问扩展后的 coords
            d_T[i, j] = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            d_D[i, j] = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)

    Truck_Speed = 1.0;
    Drone_Speed = 2.0;
    M = 10000;
    DTW_RATIO = 0.5

    # --- 成本系数 (关键调整) ---
    Cost_Truck = 1.0
    Cost_Drone = 1  # 鼓励使用无人机

    m = gp.Model("FSTSP_Epsilon")
    m.params.OutputFlag = 0  # 关闭日志
    m.params.TimeLimit = 360

    customers = [i for i in nodes if i != 0]

    # [修改] 变量定义范围扩展到 all_nodes
    # x: 卡车路径需包含终点
    x = m.addVars(all_nodes, all_nodes, vtype=GRB.BINARY, name="x")

    # # h 仅用于客户访问逻辑，原 nodes 范围即可，也可以用 all_nodes 但需约束固定值
    # h = m.addVars(nodes, vtype=GRB.BINARY, name="h")

    u = m.addVars(customers, vtype=GRB.BINARY, name="u")

    # p (库存) 和 a (到达时间) 必须包含终点
    p = m.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=NUM_DRONES, name="p")
    a = m.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0, name="a")

    t = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, name="t")
    alpha = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sat")

    # Indices
    # [修改] 无人机路径索引逻辑
    # i (发射): 可以是 start_depot 或 客户 (不能是 end_depot)
    # k (回收): 可以是 客户 或 end_depot (不能是 start_depot)
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
                    if k != j and k != l and k != i:  # 避免自环
                        z_indices.append((i, j, l, k))

    y = m.addVars(y_indices, vtype=GRB.BINARY, name="y")
    z = m.addVars(z_indices, vtype=GRB.BINARY, name="z")

    # 1. Total Cost (需遍历 all_nodes)
    cost_truck = gp.quicksum(Cost_Truck * d_T[i, j] * x[i, j] for i in all_nodes for j in all_nodes)
    cost_y = gp.quicksum(Cost_Drone * (d_D[i, j] + d_D[j, k]) * y[i, j, k] for i, j, k in y_indices)
    cost_z = gp.quicksum(Cost_Drone * (d_D[i, j] + d_D[j, l] + d_D[l, k]) * z[i, j, l, k] for i, j, l, k in z_indices)
    expr_total_cost = cost_truck + cost_y + cost_z

    # 2. Total Satisfaction
    expr_total_sat = gp.quicksum(alpha[c] for c in customers)

    # --- 约束 (复制自原代码) ---
    # [修改] 1. 卡车流约束：起点 0 出，终点 end_depot 入
    # start_depot 只有出边
    m.addConstr(gp.quicksum(x[start_depot, j] for j in all_nodes if j != start_depot) == 1)

    # end_depot 只有入边
    m.addConstr(gp.quicksum(x[j, end_depot] for j in all_nodes if j != end_depot) == 1)

    # start_depot 没有入边
    m.addConstr(gp.quicksum(x[j, start_depot] for j in all_nodes) == 0)

    # end_depot 没有出边
    m.addConstr(gp.quicksum(x[end_depot, j] for j in all_nodes) == 0)

    # 客户节点的流平衡
    for i in customers:
        # 遍历 all_nodes 寻找前驱和后继
        m.addConstr(gp.quicksum(x[j, i] for j in all_nodes if j != i) == u[i])
        m.addConstr(gp.quicksum(x[i, j] for j in all_nodes if j != i) == u[i])
        m.addConstr(x[i, i] == 0)

    for c in customers:
        sy = gp.quicksum(y[i, c, k] for i, j, k in y_indices if j == c)
        sz1 = gp.quicksum(z[i, c, l, k] for i, j, l, k in z_indices if j == c)
        sz2 = gp.quicksum(z[i, j, c, k] for i, j, l, k in z_indices if l == c)
        m.addConstr(u[c] + sy + sz1 + sz2 == 1)
        # m.addConstr(u[c] <= h[c])

        # --- [修正开始] 添加访问必要性约束 ---
        # 逻辑：卡车访问节点 c (h[c]=1) 的前提是：
        # 1. 卡车服务该节点 (u[c]=1)
        # 2. 或者，卡车在该节点发射了无人机 (Launch)
        # 3. 或者，卡车在该节点回收了无人机 (Recover)

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
        # --- [修正结束] ---

    m.addConstr(p[0] == NUM_DRONES)

    # [修改] 库存和流平衡约束需遍历除 end_depot 外的所有节点
    # 因为 end_depot 没有出边 x[end, j]，所以不需要计算流出库存
    # 但需要计算进入 end_depot 的库存 (作为检查)

    process_nodes = [n for n in all_nodes if n != end_depot]

    for i in process_nodes:
        # L, R 的计算需包含 all_nodes 中的索引
        L = gp.quicksum(y[i, j, k] for s, j, k in y_indices if s == i) + gp.quicksum(
            z[i, j, l, k] for s, j, l, k in z_indices if s == i)
        R = gp.quicksum(y[k, j, i] for k, j, e in y_indices if e == i) + gp.quicksum(
            z[k, j, l, i] for k, j, l, e in z_indices if e == i)

        if i != start_depot:
            m.addConstr(L <= M * u[i])
            m.addConstr(R <= M * u[i])
        m.addConstr(p[i] + R <= NUM_DRONES)

        # 核心库存传递：p[j] >= p[i] ...
        for j in all_nodes:
            if i != j and j != start_depot:  # j不能是起点
                m.addConstr(p[j] >= p[i] - L + R - M * (1 - x[i, j]))
                m.addConstr(p[j] <= p[i] - L + R + M * (1 - x[i, j]))

    # [修改] 仓库发射/回收限制
    # 统计从 start_depot 发射的数量
    dL = gp.quicksum(y[start_depot, j, k] for s, j, k in y_indices if s == start_depot) + gp.quicksum(
        z[start_depot, j, l, k] for s, j, l, k in z_indices if s == start_depot)

    # 统计在 end_depot 回收的数量
    dR = gp.quicksum(y[i, j, end_depot] for i, j, e in y_indices if e == end_depot) + gp.quicksum(
        z[i, j, l, end_depot] for i, j, l, e in z_indices if e == end_depot)

    m.addConstr(dL <= NUM_DRONES)
    m.addConstr(dR <= NUM_DRONES)

    m.addConstr(a[0] >= 0)

    # [修改] 时间轴传递
    # i 遍历 process_nodes (不含 end_depot)
    for i in process_nodes:
        # j 遍历 all_nodes (包含 end_depot，因为卡车要开到终点)
        for j in all_nodes:
            if i != j and j != start_depot:
                if i == start_depot:
                    dep_time = a[start_depot]
                else:
                    dep_time = t[i]  # 客户点需等待服务

                m.addConstr(a[j] >= dep_time + d_T[i, j] / Truck_Speed - M * (1 - x[i, j]))

    # [修改] 无人机时间约束 (y)
    for i, j, k in y_indices:
        # 同样的，k 可能是 end_depot
        m.addConstr(t[j] >= a[i] + d_D[i, j] / Drone_Speed - M * (1 - y[i, j, k]))

        # 无论 k 是客户还是终点，都需要满足无人机到达时间
        # 但注意，如果是 end_depot，它没有对应的 t[k] (服务时间)，只有 a[k] (到达时间)
        if k != end_depot:
            m.addConstr(a[k] >= t[j] + d_D[j, k] / Drone_Speed - M * (1 - y[i, j, k]))
        else:
            # 如果回收点是 end_depot，直接约束到达时间 a[end]
            m.addConstr(a[k] >= t[j] + d_D[j, k] / Drone_Speed - M * (1 - y[i, j, k]))

        # 同步逻辑：卡车必须到达 k
        if i == start_depot:
            dep_time = a[start_depot]
        else:
            dep_time = t[i]

        m.addConstr(a[k] >= dep_time + d_T[i, k] / Truck_Speed - M * (1 - y[i, j, k]))

    # [修改] 无人机时间约束 (z) - 类似逻辑
    for i, j, l, k in z_indices:
        # ... (前两段服务时间约束不变)
        m.addConstr(t[j] >= a[i] + d_D[i, j] / Drone_Speed - M * (1 - z[i, j, l, k]))
        m.addConstr(t[l] >= t[j] + d_D[j, l] / Drone_Speed - M * (1 - z[i, j, l, k]))

        # 回收约束
        m.addConstr(a[k] >= t[l] + d_D[l, k] / Drone_Speed - M * (1 - z[i, j, l, k]))

        if i == start_depot:
            dep_time = a[start_depot]
        else:
            dep_time = t[i]
        m.addConstr(a[k] >= dep_time + d_T[i, k] / Truck_Speed - M * (1 - z[i, j, l, k]))

    for c in customers:
        m.addConstr(t[c] >= a[c] - M * (1 - u[c]))
        # 基础 STW 约束：服务必须发生在计划时间窗内
        m.addConstr(t[c] >= STW_start[c])
        m.addConstr(t[c] <= STW_end[c])

        # [修改 2] 实现 Paper 4 Proposition 1 的分布鲁棒满意度约束
    for c in customers:
        S, E = STW_start[c], STW_end[c]
        STW_width = E - S

        # 动态计算该客户的期望时间窗宽度 w_i
        w_i = STW_width * DTW_RATIO

        # 计算鲁棒性衰减的区间长度 (Gap)
        # 该长度即为：STW 长度 - DTW 长度
        decay_gap = STW_width - w_i

        # 防止除以零（即 DTW = STW 的特例，此时 Gap=0，满意度恒为 1）
        if decay_gap > 1e-4:
            # --- 场景 A：左极端偏好 (Left Extreme DTW: [S, S+w_i]) ---
            # 客户希望尽早服务。如果服务时间 t > S+w_i，满意度线性下降。
            # 约束：alpha <= (E - t) / (E - (S + w_i))
            m.addConstr(alpha[c] <= (E - t[c]) / decay_gap, name=f"Robust_Left_{c}")

            # --- 场景 B：右极端偏好 (Right Extreme DTW: [E-w_i, E]) ---
            # 客户希望尽晚服务。如果服务时间 t < E-w_i，满意度线性下降。
            # 约束：alpha <= (t - S) / ((E - w_i) - S)
            m.addConstr(alpha[c] <= (t[c] - S) / decay_gap, name=f"Robust_Right_{c}")

        else:
            # 如果 Gap 极小，说明 STW 与 DTW 几乎重合，只要在 STW 内满意度即为 1
            # 由于 alpha 定义域为 [0,1]，无需额外线性约束，自然取到 1
            pass

    # 返回
    return m, expr_total_cost, expr_total_sat, x, y, z, a, t, alpha, y_indices, z_indices, all_nodes, coords, cust_types


def run_gurobi_solver(common_data):
    # 1. 使用适配器转换数据
    nodes, coords, STW_start, STW_end, cust_types = data_loader.adapter_for_gurobi(common_data)

    # 2. 建立基础模型
    print("Building Model...")
    # [修改] 接收 alpha 变量
    m, expr_cost, expr_sat, x, y, z, a, t, alpha, y_indices, z_indices, ext_nodes, ext_coords, ext_types = build_fstsp_model(
        nodes, coords, STW_start, STW_end, cust_types)

    # --- Phase 1: Determine Bounds ---
    print("\n[Phase 1] Determining Bounds...")

    # 1.1 Maximize Satisfaction
    print("  -> Solving for Max Satisfaction...")
    m.setObjective(expr_sat, GRB.MAXIMIZE)
    m.optimize()
    if m.status != GRB.OPTIMAL:
        print("Error: Infeasible.")
        return
    max_sat = m.objVal
    cost_at_max_sat = expr_cost.getValue()
    print(f"     Max Sat: {max_sat:.2f} (Cost: {cost_at_max_sat:.2f})")

    # 1.2 Minimize Cost
    print("  -> Solving for Min Cost...")
    m.setObjective(expr_cost, GRB.MINIMIZE)
    m.optimize()
    min_cost = m.objVal
    sat_at_min_cost = expr_sat.getValue()
    print(f"     Min Cost: {min_cost:.2f} (Sat: {sat_at_min_cost:.2f})")

    # --- Phase 2: Epsilon Loop ---
    print("\n[Phase 2] Generating Pareto Front (Max Sat s.t. Cost <= eps)...")

    last_best_sol = None

    # 设定 Epsilon (成本预算) 网格
    GRID_POINTS = 8
    eps_values = np.linspace(min_cost, cost_at_max_sat, GRID_POINTS)
    solutions = []

    # 设置主目标：最大化满意度
    m.setObjective(expr_sat, GRB.MAXIMIZE)

    # 添加约束 Cost <= 0 (动态更新)
    eps_constr = m.addConstr(expr_cost <= 0, name="Epsilon_Constraint_Cost")

    print(f"{'Iter':<5} {'Epsilon (Max Cost)':<20} {'Obj (Sat)':<15} {'Actual Cost':<15}")
    print("-" * 60)

    for i, eps in enumerate(eps_values):
        eps_constr.RHS = eps
        m.optimize()

        if m.status == GRB.OPTIMAL:
            curr_sat = expr_sat.getValue()
            curr_cost = expr_cost.getValue()
            solutions.append((curr_sat, curr_cost))
            print(f"{i + 1:<5} {eps:<20.4f} {curr_sat:<15.4f} {curr_cost:<15.2f}")

            # 保存当前可行解 (随着预算 eps 增加，Sat 通常会增加或持平)
            last_best_sol = {
                'x': m.getAttr('x', x),
                'y': m.getAttr('x', y),
                'z': m.getAttr('x', z),
                # [新增] 抓取时间变量和满意度
                'a': m.getAttr('x', a),
                't': m.getAttr('x', t),
                'alpha': m.getAttr('x', alpha),
                'sat': curr_sat,
                'cost': curr_cost
            }
        else:
            print(f"{i + 1:<5} {eps:<20.4f} {'Infeasible':<15} -")

    # --- Phase 3: Output & Details ---
    if last_best_sol:
        print("\n" + "-" * 60)
        print(f"Final Solution Route Scheme (Sat={last_best_sol['sat']:.4f}, Cost={last_best_sol['cost']:.2f}):")

        # 1. 打印拓扑结构
        truck_edges = [k for k, v in last_best_sol['x'].items() if v > 0.5]
        print(f"  Truck Edges (i -> j): {truck_edges}")

        drone_1_stop = [k for k, v in last_best_sol['y'].items() if v > 0.5]
        print(f"  Drone 1-Stop (i -> j -> k): {drone_1_stop}")

        drone_2_stop = [k for k, v in last_best_sol['z'].items() if v > 0.5]
        print(f"  Drone 2-Stop (i -> j -> l -> k): {drone_2_stop}")

        # 2. [新增] 详细时间表与满意度打印
        print("\n  Time Schedule & Satisfaction Details:")
        print(f"    {'Node':<6} {'Type':<8} {'Arrival':<12} {'Service':<12} {'Sat (Alpha)':<12}")
        print("    " + "-" * 50)

        # 辅助计算距离
        def get_dist(n1, n2):
            c1, c2 = ext_coords[n1], ext_coords[n2]
            return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

        # 推算无人机到达时间
        drone_arrival_map = {}
        Drone_Speed = 2.0

        # 1-Stop
        for (i, j, k) in drone_1_stop:
            # i=Depot时用 a[i], i=Customer时用 t[i]
            launch_time = last_best_sol['a'][i] #if i == 0 else last_best_sol['t'][i]
            fly_time = get_dist(i, j) / Drone_Speed
            drone_arrival_map[j] = launch_time + fly_time

        # 2-Stop
        for (i, j, l, k) in drone_2_stop:
            # i -> j
            launch_time = last_best_sol['a'][i] #if i == 0 else last_best_sol['t'][i]
            arr_j = launch_time + (get_dist(i, j) / Drone_Speed)
            drone_arrival_map[j] = arr_j

            # j -> l (假设 j 服务完立即起飞)
            arr_l = last_best_sol['t'][j] + (get_dist(j, l) / Drone_Speed)
            drone_arrival_map[l] = arr_l

        # 遍历所有节点
        all_node_ids = sorted(last_best_sol['a'].keys())
        for n in all_node_ids:
            # 跳过虚拟终点
            if n == ext_nodes[-1]:
                arr_val = last_best_sol['a'][n]
                print(f"    {n:<6} {'Depot':<8} {arr_val:<12.2f} {'-':<12} {'-':<12}")
                continue

            # 获取 Service
            if n in last_best_sol['t']:
                srv_val = last_best_sol['t'][n]
                srv_str = f"{srv_val:.2f}"
            else:
                srv_str = "-"

            # 获取 Sat (Alpha)
            if n in last_best_sol['alpha']:
                sat_val = last_best_sol['alpha'][n]
                sat_str = f"{sat_val:.4f}"
            else:
                sat_str = "-"

            # 获取 Arrival
            if n in drone_arrival_map:
                arr_val = drone_arrival_map[n]
                node_type = "Drone"
            else:
                arr_val = last_best_sol['a'][n]
                node_type = "Truck" if n != 0 else "Depot"

            print(f"    {n:<6} {node_type:<8} {arr_val:<12.2f} {srv_str:<12} {sat_str:<12}")

        print("-" * 60)

    # --- Phase 4: Metrics & Plot ---
    normalized_front = []
    for s_val, c_val in solutions:
        normalized_front.append([c_val, -s_val])

    ref_point_gurobi = [cost_at_max_sat, -sat_at_min_cost]
    qp_val, hv_val = calculate_metrics(normalized_front, ref_point_gurobi)

    print("\n" + "=" * 30)
    print("Performance Metrics (Gurobi - Sat Loop)")
    print("=" * 30)
    print(f"Reference Point : {ref_point_gurobi}")
    print(f"QP (Quantity)   : {qp_val}")
    print(f"HV (Hypervolume): {hv_val:.4f}")
    print("=" * 30 + "\n")

    # if last_best_sol:
    #     plot_route_scheme(
    #         ext_nodes, ext_coords,
    #         last_best_sol['x'], last_best_sol['y'], last_best_sol['z'],
    #         y_indices, z_indices, ext_types,
    #         title_suffix=f"(Sat={last_best_sol['sat']:.1f}, Cost={last_best_sol['cost']:.0f})"
    #     )
    return solutions, ref_point_gurobi


def calculate_metrics(pareto_front, reference_point):
    """
    计算非支配解集数量 (QP) 和超体积 (HV)
    pareto_front: list of [obj1, obj2] (假设均为最小化方向)
    reference_point: [ref_obj1, ref_obj2] (必须被所有解支配)
    """
    # 1. 计算 QP (Quantity of Pareto solutions)
    qp = len(pareto_front)

    # 2. 计算 HV (Hypervolume) - 2D Minimization
    # 逻辑：按第一个目标排序，然后累加每个解形成的矩形面积
    # 矩形宽度 = 下一个解的Obj1 - 当前解的Obj1 (最后一个解则到参考点)
    # 矩形高度 = 参考点Obj2 - 当前解的Obj2

    # 按目标1 (Cost) 从小到大排序
    sorted_front = sorted(pareto_front, key=lambda x: x[0])

    hv = 0.0
    ref_x, ref_y = reference_point

    for i in range(len(sorted_front)):
        curr_x, curr_y = sorted_front[i]

        # 确保解在参考点范围内 (Dominance check)
        if curr_x > ref_x or curr_y > ref_y:
            continue

        # 确定宽度 (Width)
        if i < len(sorted_front) - 1:
            next_x = sorted_front[i + 1][0]
            # 只有当下一个点的 x 更大时才计算切片（避免重复点）
            width = next_x - curr_x
        else:
            # 最后一个点，延伸到参考点 X
            width = ref_x - curr_x

        # 确定高度 (Height)
        height = ref_y - curr_y

        if width > 0 and height > 0:
            hv += width * height

    return qp, hv

if __name__ == "__main__":
    common_data = data_loader.load_solomon_data('solomon标准算例-时间窗/r1/r101.txt', n_customers=15, random_seed=42)
    run_gurobi_solver(common_data)