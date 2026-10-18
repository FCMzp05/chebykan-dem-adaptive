"""
完整 DEM + 相场 staggered 求解框架
适用于各向异性复合材料断裂（基于 Yu 等 2024 论文）
作者：根据用户需求定制
版本：1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg
import re
import warnings

# ========================== 1. Gmsh 网格读取 ==========================
def read_gmsh_mesh(filename):
    """
    读取 Gmsh 生成的 .msh 文件 (ASCII 格式，支持版本 2.2 或 4.1)
    返回:
        node_coords: [N_nodes, 2]  (x, y)  numpy array
        tri_conn:    [N_tri, 3]    (节点全局编号，从0开始) numpy array
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    node_coords = []
    tri_conn = []
    in_nodes = False
    in_elements = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '$Nodes':
            in_nodes = True
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                node_id = int(parts[0]) - 1
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append([x, y])
                i += 1
            in_nodes = False
            continue
        if line == '$Elements':
            in_elements = True
            i += 1
            n_elems = int(lines[i].strip())
            i += 1
            for _ in range(n_elems):
                parts = lines[i].strip().split()
                elem_type = int(parts[1])
                if elem_type == 2:  # 3-node triangle
                    num_tags = int(parts[2])
                    offset = 3 + num_tags
                    nodes = [int(parts[offset + j]) - 1 for j in range(3)]
                    tri_conn.append(nodes)
                i += 1
            in_elements = False
            continue
        i += 1

    node_coords = np.array(node_coords, dtype=np.float64)
    tri_conn = np.array(tri_conn, dtype=np.int64)
    return node_coords, tri_conn


# ========================== 2. 三角形 FEM 数据预计算 ==========================
def compute_triangle_fem_data(node_coords, tri_conn):
    """
    对每个线性三角形单元，计算：
        - 高斯点坐标 (质心)      [N_tri, 2]
        - 高斯权重 (单元面积)    [N_tri]
        - 形函数导数 dN/dx, dN/dy [N_tri, 3]
        - 连接矩阵               [N_tri, 3]
    返回字典
    """
    # 转换为 torch tensor 便于后续操作
    if not isinstance(node_coords, torch.Tensor):
        node_coords = torch.tensor(node_coords, dtype=torch.float64)
    if not isinstance(tri_conn, torch.Tensor):
        tri_conn = torch.tensor(tri_conn, dtype=torch.long)

    device = node_coords.device
    dtype = node_coords.dtype

    p0 = node_coords[tri_conn[:, 0]]
    p1 = node_coords[tri_conn[:, 1]]
    p2 = node_coords[tri_conn[:, 2]]

    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]

    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    area = torch.abs(area2) / 2.0
    area = torch.clamp(area, min=1e-12)

    # 质心
    centroid_x = (x0 + x1 + x2) / 3.0
    centroid_y = (y0 + y1 + y2) / 3.0
    gauss_pts = torch.stack([centroid_x, centroid_y], dim=-1)  # [N_tri, 2]
    gauss_w = area  # [N_tri]

    # 形函数导数（保留符号）
    denom = area2
    denom = torch.where(torch.abs(denom) < 1e-12, torch.ones_like(denom) * 1e-12, denom)
    dN0dx = (y1 - y2) / denom
    dN1dx = (y2 - y0) / denom
    dN2dx = (y0 - y1) / denom
    dN0dy = (x2 - x1) / denom
    dN1dy = (x0 - x2) / denom
    dN2dy = (x1 - x0) / denom

    dN_dx = torch.stack([dN0dx, dN1dx, dN2dx], dim=-1)  # [N_tri, 3]
    dN_dy = torch.stack([dN0dy, dN1dy, dN2dy], dim=-1)

    return {
        'node_coords': node_coords,
        'tri_conn': tri_conn,
        'flat_gauss_pts': gauss_pts,
        'flat_gauss_w': gauss_w,
        'flat_dN_dx': dN_dx,
        'flat_dN_dy': dN_dy,
        'flat_conn': tri_conn,  # 每个单元连接相同
    }


# ========================== 3. 复合材料材料模型 ==========================
class CompositeMaterial:
    """
    正交各向异性复合材料，平面应力（默认），能量分解遵循 Yu 等 2024 论文式 (17)-(25)
    """
    def __init__(self, E1, E2, G12, nu12, theta_func, is_plane_strain=False):
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.theta_func = theta_func
        self.is_plane_strain = is_plane_strain

        denom = 1.0 - nu12 * nu12 * (E2 / E1)
        self.Q11 = E1 / denom
        self.Q22 = E2 / denom
        self.Q12 = nu12 * E2 / denom
        self.Q66 = G12

        if is_plane_strain:
            warnings.warn("平面应变未完全实现，当前为平面应力")

    def get_global_stiffness(self, theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        c2 = c * c
        s2 = s * s
        cs = c * s

        Q_loc = torch.tensor([
            [self.Q11, self.Q12, 0.0],
            [self.Q12, self.Q22, 0.0],
            [0.0, 0.0, self.Q66]
        ], dtype=theta.dtype, device=theta.device)

        # 工程剪应变约定 (eps = (exx, eyy, gxy), Q66 = G12):
        # e11 = c2*exx + s2*eyy + cs*gxy; g12 = -2cs*exx + 2cs*eyy + (c2-s2)*gxy
        row1 = torch.stack([c2, s2, cs], dim=-1)
        row2 = torch.stack([s2, c2, -cs], dim=-1)
        row3 = torch.stack([-2 * cs, 2 * cs, c2 - s2], dim=-1)
        T = torch.stack([row1, row2, row3], dim=-2)

        Q_global = torch.einsum('...ki,...kl,...lj->...ij', T, Q_loc, T)
        return Q_global

    def compute_driving_components(self, eps, theta):
        """
        计算三个驱动能分量（用于历史场更新）
        返回: (psi_TEEDF, psi_TEEDM, psi_TSEDM) 均为 [N]
        对应式 (20)-(22)
        """
        exx, eyy, exy = eps[:, 0], eps[:, 1], eps[:, 2]
        exx_p = torch.relu(exx)
        eyy_p = torch.relu(eyy)

        # 构造 ε⁺ = [⟨εxx⟩₊, ⟨εyy⟩₊, γxy]
        eps_p = torch.stack([exx_p, eyy_p, exy], dim=-1)

        Q = self.get_global_stiffness(theta)          # [N, 3, 3]
        sigma_p = torch.einsum('...ij,...j->...i', Q, eps_p)  # [N, 3]

        # 按式 (20)-(22) 计算三项
        psi_TEEDF = 0.5 * exx_p * sigma_p[:, 0]       # 纤维拉伸
        psi_TEEDM = 0.5 * eyy_p * sigma_p[:, 1]       # 基体拉伸
        psi_TSEDM = 0.5 * exy * sigma_p[:, 2]         # 剪切
        return psi_TEEDF, psi_TEEDM, psi_TSEDM

    def compute_psi(self, eps, d, theta):
        """
        eps:   [N, 3]  (exx, eyy, exy)
        d:     [N] 或标量
        theta: [N] 或标量
        返回: psi_plus [N], psi_minus [N]
        """
        exx, eyy, exy = eps[:, 0], eps[:, 1], eps[:, 2]
        exx_p = torch.relu(exx)
        exx_m = torch.minimum(exx, torch.zeros_like(exx))
        eyy_p = torch.relu(eyy)
        eyy_m = torch.minimum(eyy, torch.zeros_like(eyy))

        # 构造 ε⁺ 和 ε⁻ (式18)
        eps_p = torch.stack([exx_p, eyy_p, exy], dim=-1)          # 剪切始终为正
        eps_m = torch.stack([exx_m, eyy_m, torch.zeros_like(exy)], dim=-1)

        Q = self.get_global_stiffness(theta)          # [N, 3, 3]

        # 计算 ψ₀⁺ = ½ (ε⁺)ᵀ Q (ε⁺)
        psi_plus = 0.5 * torch.einsum('...i,...ij,...j->...', eps_p, Q, eps_p)

        # 计算 ψ₀⁻ = ½ (ε⁻)ᵀ Q (ε⁻)
        psi_minus = 0.5 * torch.einsum('...i,...ij,...j->...', eps_m, Q, eps_m)

        # 退化
        g = (1 - d) ** 2
        return g * psi_plus, psi_minus

# ========================== 4. 位移神经网络（含 Adaptive ReLU） ==========================
class AdaptiveReLU(nn.Module):
    """可训练参数的 ReLU: max(0, m * x)，参考 Manav 等 2024"""
    def __init__(self, in_features, init_m=1.0):
        super().__init__()
        self.m = nn.Parameter(torch.ones(in_features) * init_m)

    def forward(self, x):
        return torch.relu(self.m * x)


class DisplacementNet(nn.Module):
    def __init__(self, dim_x=2, dim_u=2, hidden_layers=6, hidden_neurons=200,
                 init_m=1.0, output_scale=1.0):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.output_scale = output_scale
        layers = []
        layers.append(nn.Linear(dim_x, hidden_neurons))
        layers.append(AdaptiveReLU(hidden_neurons, init_m))
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(AdaptiveReLU(hidden_neurons, init_m))
        layers.append(nn.Linear(hidden_neurons, dim_u))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, dist_func, u_bc_func):
        raw_u = self.net(x)
        dist = dist_func(x)
        u_bc = u_bc_func(x)
        if dist.ndim == 1:
            dist = dist.unsqueeze(-1)
        return u_bc + self.output_scale * dist * raw_u

# ========================== 5. DEM 损失函数 ==========================
class DEMLoss:
    def __init__(self, material, fem_data, dist_func, u_bc_func):
        """
        fem_data: 由 compute_triangle_fem_data 返回的字典
        """
        self.material = material
        self.node_coords = fem_data['node_coords']
        self.flat_conn = fem_data['flat_conn']          # [N_tri, 3]
        self.flat_gauss_pts = fem_data['flat_gauss_pts'] # [N_tri, 2]
        self.flat_gauss_w = fem_data['flat_gauss_w']     # [N_tri]
        self.flat_dN_dx = fem_data['flat_dN_dx']         # [N_tri, 3]
        self.flat_dN_dy = fem_data['flat_dN_dy']         # [N_tri, 3]
        self.dist_func = dist_func
        self.u_bc_func = u_bc_func

        # 预计算纤维角度
        with torch.no_grad():
            theta_vals = self.material.theta_func(
                self.flat_gauss_pts[:, 0],
                self.flat_gauss_pts[:, 1]
            )
        self.flat_theta = theta_vals          # 改为普通属性
        self.node_coords_t = self.node_coords  # 改为普通属性

    def compute_strain(self, u_nodes):
        """
        u_nodes: [N_nodes, 2]
        返回: eps [N_tri, 3] (exx, eyy, exy)
        """
        u_e = u_nodes[self.flat_conn]  # [N_tri, 3, 2]
        ux = u_e[:, :, 0]
        uy = u_e[:, :, 1]
        exx = torch.sum(self.flat_dN_dx * ux, dim=1)
        eyy = torch.sum(self.flat_dN_dy * uy, dim=1)
        gamma_xy = (torch.sum(self.flat_dN_dy * ux, dim=1) +
                    torch.sum(self.flat_dN_dx * uy, dim=1))
        return torch.stack([exx, eyy, gamma_xy], dim=-1)

    def compute_potential(self, net, d_gauss):
        """
        d_gauss: [N_tri] 损伤场在高斯点（单元质心）的值
        返回: log(总势能)
        """
        u_nodes = net(self.node_coords_t, self.dist_func, self.u_bc_func)
        eps = self.compute_strain(u_nodes)
        psi_plus, psi_minus = self.material.compute_psi(eps, d_gauss, self.flat_theta)
        energy = torch.sum(self.flat_gauss_w * (psi_plus + psi_minus))
        return torch.log(energy + 1e-12)


# ========================== 6. 位移训练函数（RPROP） ==========================
def train_displacement_subproblem(net, loss_func, d_gauss, max_epochs=5000,
                                  lr=1e-5, min_step=1e-10, max_step=50.0,
                                  tol=1e-6, patience=10, weight_decay=1e-5,
                                  verbose=True):
    """
    使用 RPROP 优化器训练位移子问题
    引入 L2 权重正则化（参考 Manav 等 2024）

    参数:
        net: 位移神经网络
        loss_func: DEM 损失函数
        d_gauss: 当前损伤场（高斯点值）
        max_epochs: 最大迭代次数
        lr: RPROP 学习率
        min_step, max_step: RPROP 允许的最小/最大步长
        tol: 收敛容差
        patience: 收敛判断窗口
        weight_decay: L2 权重正则化系数
        verbose: 是否打印训练信息
    """
    # RPROP 优化器（与 Manav 文献一致）
    optimizer = optim.Rprop(
        net.parameters(), lr=lr, etas=(0.5, 1.2),
        step_sizes=(min_step, max_step)
    )
    losses = []
    converged = False

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # 计算能量损失
        energy_loss = loss_func.compute_potential(net, d_gauss)

        # ----- 添加 L2 权重正则化 -----
        l2_reg = 0.0
        for param in net.parameters():
            l2_reg += torch.sum(param ** 2)
        reg_loss = weight_decay * l2_reg

        # 总损失 = log(能量) + 正则化项
        loss = energy_loss + reg_loss

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        # 收敛判据：连续 patience 个 epoch 变化小于 tol
        if len(losses) > patience:
            delta = abs(losses[-1] - losses[-patience]) / max(
                1.0, abs(losses[-patience])
            )
            if delta < tol:
                converged = True
                if verbose:
                    print(f"位移子问题收敛于 Epoch {epoch}, loss={current_loss:.4e}, delta={delta:.4e}")
                break

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss:.4e}")

    # 每次训练结束都输出实际经过的 epoch 数 (不依赖 verbose)
    reason = "收敛" if converged else "达到最大 epoch 数"
    print(f"位移训练停止: 共 {epoch + 1}/{max_epochs} 个 epoch ({reason})")
    if converged and epoch + 1 <= 500:
        print(f"  500 步内收敛, 收敛时 loss = {current_loss:.4e}")
    return net, losses


# ========================== 7. 相场有限元求解器 ==========================
class PhaseFieldSolver:
    def __init__(self, node_coords, tri_conn, Gc, l, beta, theta_func):
        """
        node_coords: [Nn, 2] numpy array
        tri_conn:    [Ne, 3] numpy array
        Gc, l, beta: 标量
        theta_func:  函数 theta(x, y) 返回弧度 (输入 [N,2] 输出 [N])
        """
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.tri_conn = np.asarray(tri_conn, dtype=np.int64)
        self.Gc = Gc
        self.l = l
        self.beta = beta
        self.theta_func = theta_func

        self.Nn = self.node_coords.shape[0]
        self.Ne = self.tri_conn.shape[0]

        # 预计算单元常数（使用 numpy）
        self._precompute()

    def _precompute(self):
        coords = self.node_coords
        conn = self.tri_conn
        p0 = coords[conn[:, 0]]
        p1 = coords[conn[:, 1]]
        p2 = coords[conn[:, 2]]
        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]

        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        self.area = np.abs(area2) / 2.0
        self.area = np.maximum(self.area, 1e-12)

        denom = area2
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        dN0dx = (y1 - y2) / denom
        dN1dx = (y2 - y0) / denom
        dN2dx = (y0 - y1) / denom
        dN0dy = (x2 - x1) / denom
        dN1dy = (x0 - x2) / denom
        dN2dy = (x1 - x0) / denom

        self.dNdx = np.stack([dN0dx, dN1dx, dN2dx], axis=1)  # [Ne, 3]
        self.dNdy = np.stack([dN0dy, dN1dy, dN2dy], axis=1)

        self.centroid = np.stack([(x0+x1+x2)/3.0, (y0+y1+y2)/3.0], axis=1)

    def assemble_system(self, H_gauss, theta_gauss=None, fixed_nodes=None, fixed_vals=None):
        """
        H_gauss:     [Ne] 历史场值 (高斯点)
        theta_gauss: [Ne] 纤维角度 (弧度)
        fixed_nodes: list/array 固定自由度节点编号
        fixed_vals:  list/array 对应的固定值
        返回: K (csr_matrix), f (ndarray)
        """
        if theta_gauss is None:
            theta_gauss = self.theta_func(self.centroid[:, 0], self.centroid[:, 1])
        theta_gauss = np.asarray(theta_gauss)

        Ne = self.Ne
        Nn = self.Nn
        Gc = self.Gc
        l = self.l
        beta = self.beta

        rows, cols, vals = [], [], []
        rhs = np.zeros(Nn)

        c = np.cos(theta_gauss)
        s = np.sin(theta_gauss)

        for e in range(Ne):
            nodes = self.tri_conn[e]
            area_e = self.area[e]
            dNdx_e = self.dNdx[e]  # [3]
            dNdy_e = self.dNdy[e]
            H_e = H_gauss[e]
            theta_e = theta_gauss[e]
            a = np.array([c[e], s[e]])

            dN = np.column_stack((dNdx_e, dNdy_e))  # [3,2]
            dN_dot_a = dN @ a  # [3]

            # 各向异性梯度刚度
            K_aniso = Gc * l * area_e * (dN @ dN.T + beta * np.outer(dN_dot_a, dN_dot_a))

            # 质量矩阵
            Ni = np.ones(3) / 3.0
            M_e = area_e * np.outer(Ni, Ni) * (Gc / l + 2.0 * H_e)

            K_e = K_aniso + M_e

            # 右端项
            f_e = 2.0 * H_e * area_e * Ni

            # 组装
            for i in range(3):
                row = nodes[i]
                rhs[row] += f_e[i]
                for j in range(3):
                    col = nodes[j]
                    rows.append(row)
                    cols.append(col)
                    vals.append(K_e[i, j])

        K = coo_matrix((vals, (rows, cols)), shape=(Nn, Nn)).tocsr()

        # Dirichlet 边界
        if fixed_nodes is not None:
            fixed_nodes = np.asarray(fixed_nodes, dtype=int)
            fixed_vals = np.asarray(fixed_vals, dtype=float)
            for i, node in enumerate(fixed_nodes):
                K[node, :] = 0.0
                K[node, node] = 1.0
                rhs[node] = fixed_vals[i]

        return K, rhs

    def solve(self, H_gauss, theta_gauss=None, fixed_nodes=None, fixed_vals=None):
        K, f = self.assemble_system(H_gauss, theta_gauss, fixed_nodes, fixed_vals)
        try:
            d = spsolve(K, f)
        except Exception:
            d, _ = cg(K, f, tol=1e-6, maxiter=1000)
        d = np.clip(d, 0.0, 1.0)
        return d


# ========================== 8. 历史场更新 ==========================
def update_history(H1, H2, H3, psi1, psi2, psi3):
    """
    分别更新三个历史场分量，并返回总历史场
    H1, H2, H3: 当前历史场分量 (numpy array)
    psi1, psi2, psi3: 当前驱动能分量 (numpy array)
    返回: (H1_new, H2_new, H3_new, H_total)
    """
    H1_new = np.maximum(H1, psi1)
    H2_new = np.maximum(H2, psi2)
    H3_new = np.maximum(H3, psi3)
    H_total = H1_new + H2_new + H3_new
    return H1_new, H2_new, H3_new, H_total


# ========================== 9. 主程序示例 ==========================
if __name__ == "__main__":
    # -------------------- 创建简单网格（4个三角形，方形板） --------------------
    # 实际使用时用 read_gmsh_mesh("mesh.msh") 替换
    node_coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ], dtype=np.float64)
    tri_conn = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int64)

    # -------------------- 材料参数（示例） --------------------
    def theta_func(x, y):
        return np.zeros_like(x)  # 0°纤维方向

    E1 = 114.8
    E2 = 11.7
    G12 = 9.66
    nu12 = 0.24
    Gc = 0.622   # N/mm (假设)
    l = 0.01
    beta = 300.0

    # -------------------- 边界条件 --------------------
    # 距离函数：左边界 x=0 固定，内部为正
    def dist_func(x):
        return x[:, 0]  # tensor

    # 边界位移：右边界 x=1 施加 y 方向位移 U_p
    def make_u_bc_func(U_p):
        def u_bc_func(x):
            u_bc = torch.zeros_like(x)
            mask = torch.abs(x[:, 0] - 1.0) < 1e-6
            u_bc[mask, 1] = U_p
            return u_bc
        return u_bc_func

    # 初始损伤场固定节点（例如预置裂纹，这里空置）
    fixed_nodes = None
    fixed_vals = None

    # -------------------- 准备 PyTorch 数据 --------------------
    node_t = torch.tensor(node_coords, dtype=torch.float64)
    tri_t = torch.tensor(tri_conn, dtype=torch.long)
    fem_data = compute_triangle_fem_data(node_t, tri_t)

    # 材料对象（需转换为 torch 可调用）
    def theta_torch(x, y):
        return torch.zeros_like(x)
    material = CompositeMaterial(E1, E2, G12, nu12, theta_torch)

    # 初始损伤场（高斯点）= 0
    d_gauss = torch.zeros(fem_data['flat_gauss_w'].shape[0], dtype=torch.float64)

    # -------------------- 加载步循环（仅演示一步） --------------------
    U_p = 0.01
    u_bc_func = make_u_bc_func(U_p)

    # 位移网络
    net = DisplacementNet(hidden_layers=4, hidden_neurons=100)
    loss_func = DEMLoss(material, fem_data, dist_func, u_bc_func)

    # 相场求解器（numpy 版本）
    pf_solver = PhaseFieldSolver(node_coords, tri_conn, Gc, l, beta, theta_func)

    # 历史场初始化
    H1 = np.zeros(pf_solver.Ne)
    H2 = np.zeros(pf_solver.Ne)
    H3 = np.zeros(pf_solver.Ne)

    # ---------- 位移子问题训练 ----------
    print("===== 位移训练 =====")
    net, _ = train_displacement_subproblem(net, loss_func, d_gauss, max_epochs=1000, verbose=True)

    # 获取位移，计算高斯点应变及驱动能
    with torch.no_grad():
        u_nodes = net(node_t, dist_func, u_bc_func)
        eps = loss_func.compute_strain(u_nodes)  # [Ne, 3]
        # 转为 numpy
        eps_np = eps.numpy()
        theta_np = theta_func(fem_data['flat_gauss_pts'][:, 0].numpy(),
                              fem_data['flat_gauss_pts'][:, 1].numpy())
        # 计算驱动能分量（需要材料的方法，但材料是 torch 版本，暂时用 numpy 重算，或转换）
        # 这里为简便，直接使用 torch 材料计算
        psi1, psi2, psi3 = material.compute_driving_components(eps, loss_func.flat_theta)
        psi1 = psi1.numpy()
        psi2 = psi2.numpy()
        psi3 = psi3.numpy()

    # 更新历史场
    H1, H2, H3, H_total = update_history(H1, H2, H3, psi1, psi2, psi3)

    # ---------- 相场求解 ----------
    print("===== 相场求解 =====")
    d_new = pf_solver.solve(H_total, theta_gauss=theta_np, fixed_nodes=fixed_nodes, fixed_vals=fixed_vals)

    print("相场最大值:", d_new.max())
    print("历史场最大值:", H_total.max())

    # 注意：实际 staggered 迭代需重复位移-相场步骤直至收敛，此处仅演示单步。
