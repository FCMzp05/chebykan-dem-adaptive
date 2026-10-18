"""ChebyKAN-DEM 剪切板：Q4 / 可变节点四边形在线 h 自适应。

云端 CUDA 运行：python ChebyKAN_Q4_Adaptive.py
同目录放 initial_weights.pt、tip_seeded.npz；依赖 torch、numpy、matplotlib。
默认 0–15 μm、41 个接受点；输出网格图、逐步场和可续算检查点。
仅预览：加 --preview-only --output mesh_preview；续算：加 --resume。

由 cheby_quad_initguard_20260929 的已跑通版本合并；物理模型未修改。
保留首步 Rprop 保护、历史传递和同载荷重解。裂纹场精度仍待核对。
本单文件版本使用独立来源校验，只续算自身输出，不接续旧三文件输出。
"""
import argparse
import copy
from functools import lru_cache
import hashlib
from itertools import count
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn as nn


# 1. 材料、加载与训练参数（mm、N）
class Config:
    device = 'cuda'
    Lx, Ly = 1.0, 1.0
    mat_E, mat_nu = 210000.0, 0.3
    Gc, l0, k_residual = 2.7, 0.015, 1e-10
    lmbda = mat_E * mat_nu / ((1 + mat_nu) * (1 - 2 * mat_nu))
    mu = mat_E / (2 * (1 + mat_nu))
    u_max, n_steps = 0.015, 41
    disp_steps = np.linspace(0.0, u_max, n_steps)
    loading = 'shear'
    tol_ir = 1e-2
    kan_layers = [2, 128, 128, 64, 3]
    chebyshev_degree = 3
    n_epochs_pretrain = 1000
    lr, weight_decay = 5e-5, 1e-6
    alpha_grad_penalty, rigid_body_penalty = 0.005, 1.0
    early_stop_patience, early_stop_min_epochs = 100, 200
    early_stop_tol = 1e-6
    seed = 1234


conf = Config()



def log(msg):
    """打印日志"""
    print(msg, flush=True)


# 2. 网络、不可逆约束与谱分解


class NonsmoothSigmoid(nn.Module):
    def __init__(self, support=2.0, coeff=1e-3):
        super().__init__()
        self.support = support
        self.coeff = coeff

    def forward(self, x):
        a = x > self.support
        b = x < -self.support
        c = ~(a | b)
        return (a * (self.coeff * (x - self.support) + 1.0) +
                b * (self.coeff * (x + self.support)) +
                c * (x / (2 * self.support) + 0.5))


class IrreversiblePhaseField(nn.Module):
    def __init__(self, beta=50.0):
        super().__init__()
        self.beta = beta

    def forward(self, alpha_net, alpha_history):
        diff = alpha_net - alpha_history
        alpha_increment = torch.nn.functional.softplus(diff, beta=self.beta)
        alpha_final = alpha_history + alpha_increment
        return torch.clamp(alpha_final, 0.0, 1.0)


class MaterialProperties:
    def __init__(self):
        self.E = torch.tensor(conf.mat_E, device=conf.device)
        self.nu = torch.tensor(conf.mat_nu, device=conf.device)
        self.Gc = torch.tensor(conf.Gc, device=conf.device)
        self.l0 = torch.tensor(conf.l0, device=conf.device)
        self.lmbda = torch.tensor(conf.lmbda, device=conf.device)
        self.mu = torch.tensor(conf.mu, device=conf.device)
        self.k = torch.tensor(conf.k_residual, device=conf.device)


class PFFModel:
    def degrade(self, alpha):
        return (1 - alpha)**2, 2 * (alpha - 1)
    def damage_fun(self, alpha):
        return alpha**2, 2.0
    def irr_penalty_weight(self):
        return 1.0 / conf.tol_ir**2 - 1.0


class ChebyKANLayer(nn.Module):
    """Chebyshev-KAN 层"""
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.expanded_dim = in_features * (degree + 1)
        self.linear = nn.Linear(self.expanded_dim, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x_norm = torch.tanh(x)
        T_list = [torch.ones_like(x_norm), x_norm]
        for k in range(2, self.degree + 1):
            T_next = 2 * x_norm * T_list[-1] - T_list[-2]
            T_list.append(T_next)
        T_concat = torch.cat(T_list, dim=-1)
        return self.linear(T_concat)


class ChebyKANNetwork(nn.Module):
    """ChebyKAN 网络"""
    def __init__(self, layer_sizes=None, degree=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = conf.kan_layers
        if degree is None:
            degree = conf.chebyshev_degree

        self.layer_sizes = layer_sizes
        self.degree = degree

        self.kan_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.kan_layers.append(ChebyKANLayer(layer_sizes[i], layer_sizes[i+1], degree))

        self._print_info()

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        log(f"  ChebyKAN Network: {' → '.join(map(str, self.layer_sizes))}, Degree: {self.degree}, Params: {total_params:,}")

    def forward(self, x):
        for i, layer in enumerate(self.kan_layers[:-1]):
            x = layer(x)
            x = torch.tanh(x)
        return self.kan_layers[-1](x)


class DEMSolverBase:
    def __init__(self, network, nodes, conn, areas):
        self.network = network
        self.alpha_constraint = NonsmoothSigmoid()
        self.irreversible = IrreversiblePhaseField(beta=50.0)
        self.mat = MaterialProperties()
        self.pff = PFFModel()

        # 转换为 PyTorch tensors
        self.inp = torch.tensor(nodes, dtype=torch.float32).to(conf.device)
        self.conn = torch.tensor(conn, dtype=torch.long).to(conf.device)
        self.area = torch.tensor(areas, dtype=torch.float32).to(conf.device)

        # 边界掩码
        tol = 1e-5
        self.bottom_mask = torch.abs(self.inp[:, 1] - (-conf.Ly/2)) < tol
        self.top_mask = torch.abs(self.inp[:, 1] - (conf.Ly/2)) < tol

        # 中心节点
        bottom_nodes = self.inp[self.bottom_mask]
        if len(bottom_nodes) > 0:
            center_idx = torch.argmin(torch.abs(bottom_nodes[:, 0]))
            self.center_node_idx = torch.where(self.bottom_mask)[0][center_idx].item()
        else:
            self.center_node_idx = 0

        # 历史场
        self.alpha_history = torch.zeros(self.inp.shape[0], device=conf.device)

    def compute_fields(self, lmbda_val):
        out = self.network(self.inp)
        if not torch.is_tensor(lmbda_val):
            lmbda_val = torch.tensor(lmbda_val, device=conf.device, dtype=torch.float32)

        x, y = self.inp[:, 0], self.inp[:, 1]
        y0, yL = -conf.Ly/2, conf.Ly/2
        bubble_y = (y - y0) * (yL - y)
        linear_y = (y - y0) / (yL - y0)

        # Mode I 顶边竖向拉开；Mode II 顶边水平剪切。上下边的气泡项为零。
        if conf.loading == 'tension':
            u = bubble_y * out[:, 0] * lmbda_val
            v = (bubble_y * out[:, 1] + linear_y) * lmbda_val
        else:
            u = (bubble_y * out[:, 0] + linear_y) * lmbda_val
            v = bubble_y * out[:, 1] * lmbda_val

        alpha_raw = out[:, 2]
        alpha_net = self.alpha_constraint(alpha_raw)
        alpha = self.irreversible(alpha_net, self.alpha_history)
        return u, v, alpha

    def spectral_split(self, eps_11, eps_22, eps_12):
        lmbda, mu = self.mat.lmbda, self.mat.mu
        trace = eps_11 + eps_22
        dev = (eps_11 - eps_22) / 2.0
        radius = torch.linalg.vector_norm(torch.stack((dev, eps_12), dim=-1), dim=-1)  # Finite subgradient at zero.
        eps_1 = trace / 2.0 + radius
        eps_2 = trace / 2.0 - radius
        eps_1_plus, eps_2_plus = torch.relu(eps_1), torch.relu(eps_2)
        eps_1_minus, eps_2_minus = eps_1 - eps_1_plus, eps_2 - eps_2_plus
        trace_plus = eps_1_plus + eps_2_plus
        trace_minus = eps_1_minus + eps_2_minus
        psi_plus = 0.5 * lmbda * trace_plus**2 + mu * (eps_1_plus**2 + eps_2_plus**2)
        psi_minus = 0.5 * lmbda * trace_minus**2 + mu * (eps_1_minus**2 + eps_2_minus**2)
        return psi_plus, psi_minus


# 3. 四边形网格、可变节点积分、标记与细化
def initial_mesh():
    x = np.linspace(-.5, .5, 21)
    y = np.r_[np.linspace(-.5, -.0075, 11), np.linspace(.0075, .5, 11)]
    nodes = np.array([(a, b) for b in y for a in x])
    cells = []
    for j in range(len(y)-1):
        for i in range(len(x)-1):
            if j == 10 and i < 10:
                continue
            a = j*len(x)+i
            cells.append([a, a+1, a+1+len(x), a+len(x)])
    cells = np.asarray(cells, dtype=np.int64)
    used, inverse = np.unique(cells, return_inverse=True)
    return dict(nodes=np.round(nodes[used], 14), cells=inverse.reshape(-1, 4),
                level=np.zeros(len(cells), dtype=np.int64), h0=.05)


def geometry(mesh):
    p = mesh['nodes'][mesh['cells']]
    widths = p[:, 2]-p[:, 0]
    assert np.all(widths > 0)
    assert np.allclose(p[:, 1], np.c_[p[:, 2, 0], p[:, 0, 1]], atol=1e-13)
    assert np.allclose(p[:, 3], np.c_[p[:, 0, 0], p[:, 2, 1]], atol=1e-13)
    return widths.prod(axis=1), widths.max(axis=1), widths


def connectivity(mesh):
    """Return corners + extra edge nodes, and true atomic edges with their owners."""
    nodes = mesh['nodes']
    lines = [{}, {}]
    for axis in (0, 1):
        for value in np.unique(nodes[:, axis]):
            ids = np.flatnonzero(nodes[:, axis] == value)
            ids = ids[np.argsort(nodes[ids, 1-axis])]
            lines[axis][float(value)] = ids
    supports, atoms = [], {}
    for e, corners in enumerate(mesh['cells']):
        extra = []
        for a, b in zip(corners, np.roll(corners, -1)):
            axis = int(nodes[a, 1] == nodes[b, 1])
            ids = lines[axis][float(nodes[a, axis])]
            coord = nodes[ids, 1-axis]
            lo, hi = sorted((nodes[a, 1-axis], nodes[b, 1-axis]))
            ids = ids[np.searchsorted(coord, lo):np.searchsorted(coord, hi, side='right')]
            extra.extend(ids[1:-1].tolist())
            for u, v in zip(ids[:-1], ids[1:]):
                atoms.setdefault(tuple(sorted((int(u), int(v)))), []).append(e)
        supports.append(np.r_[corners, np.asarray(extra, dtype=np.int64)])
    return supports, atoms


def local_nodes(mesh, e, support):
    corners = mesh['nodes'][mesh['cells'][e]]
    return 2*(mesh['nodes'][support]-corners[0])/(corners[2]-corners[0])-1


def basis(points, extra):
    x, y = np.asarray(points).T
    f = [np.ones_like(x), x, y, x*y]
    dx = [np.zeros_like(x), np.ones_like(x), np.zeros_like(x), y]
    dy = [np.zeros_like(x), np.zeros_like(x), np.ones_like(x), x]
    for a, b in extra:
        if abs(abs(b)-1) < 1e-9:
            f.append(abs(x-a)*(y+np.sign(b)))
            dx.append(np.sign(x-a)*(y+np.sign(b)))
            dy.append(abs(x-a))
        else:
            assert abs(abs(a)-1) < 1e-9
            f.append(abs(y-b)*(x+np.sign(a)))
            dx.append(abs(y-b))
            dy.append(np.sign(y-b)*(x+np.sign(a)))
    return np.stack(f, axis=1), np.stack(dx, axis=1), np.stack(dy, axis=1)


def shape(local, points):
    """MATLAB author's A^-1 f basis, using a solve instead of explicit inverse."""
    extra = local[4:]
    a = basis(local, extra)[0].T
    f, dx, dy = basis(points, extra)
    return tuple(np.linalg.solve(a, z.T).T for z in (f, dx, dy))


@lru_cache(maxsize=64)
def reference_rule(key):
    local = np.asarray(key).reshape(-1, 2)
    # Resolve every abs-basis kink before applying 2x2 Gauss integration.
    cuts = [np.unique(np.r_[-1., local[4:, k], 1.]) for k in (0, 1)]
    points, weights = [], []
    g = 1/np.sqrt(3)
    for xa, xb in zip(cuts[0][:-1], cuts[0][1:]):
        for ya, yb in zip(cuts[1][:-1], cuts[1][1:]):
            for u, v in ((-g, -g), (g, -g), (g, g), (-g, g)):
                points.append([(xa+xb)/2+u*(xb-xa)/2, (ya+yb)/2+v*(yb-ya)/2])
                weights.append((xb-xa)*(yb-ya)/4)
    n, dx, dy = shape(local, points)
    center = shape(local, [[0., 0.]])[0][0]
    # Each smooth patch is bilinear, so its extrema occur at these vertices.
    vertices = [(x, y) for y in cuts[1] for x in cuts[0]]
    bounds_n = shape(local, vertices)[0]
    return np.asarray(points), np.asarray(weights), n, np.stack((dx, dy), axis=-1), center, bounds_n


def quadrature(mesh):
    supports, atoms = connectivity(mesh)
    area, h, widths = geometry(mesh)
    max_nodes = max(map(len, supports))
    conn = np.zeros((len(supports), max_nodes), dtype=np.int64)
    mask = np.zeros_like(conn, dtype=bool)
    center_n = np.zeros_like(conn, dtype=float)
    qconn, qn, qgrad, qw, qe, qxy = [], [], [], [], [], []
    bconn, bn, be = [], [], []
    for e, ids in enumerate(supports):
        local = np.round(local_nodes(mesh, e, ids), 12)
        points, weights, n, grad, center, bounds_n = reference_rule(tuple(local.ravel()))
        conn[e, :len(ids)], mask[e, :len(ids)] = ids, True
        center_n[e, :len(ids)] = center
        pad = max_nodes-len(ids)
        qconn.extend(np.tile(conn[e], (len(weights), 1)))
        qn.extend(np.pad(n, ((0, 0), (0, pad))))
        qgrad.extend(np.pad(grad*2/widths[e], ((0, 0), (0, pad), (0, 0))))
        qw.extend(weights*area[e]/4)
        qe.extend([e]*len(weights))
        qxy.extend(mesh['nodes'][mesh['cells'][e, 0]]+(points+1)*widths[e]/2)
        bconn.extend(np.tile(conn[e], (len(bounds_n), 1)))
        bn.extend(np.pad(bounds_n, ((0, 0), (0, pad))))
        be.extend([e]*len(bounds_n))
    return dict(conn=conn, mask=mask, center_n=center_n, area=area, h=h,
                qconn=np.asarray(qconn), N=np.asarray(qn), grad=np.asarray(qgrad),
                weight=np.asarray(qw), owner=np.asarray(qe), points=np.asarray(qxy),
                supports=supports, atoms=atoms, bound_conn=np.asarray(bconn),
                bound_N=np.asarray(bn), bound_owner=np.asarray(be))


def admissibility(damage, lower, data, tolerance=1e-6):
    """Mark cells violating damage bounds/irreversibility anywhere in a patch."""
    ids, n = data['bound_conn'], data['bound_N']
    values = (n*damage[ids]).sum(axis=1)
    increments = (n*(damage-lower)[ids]).sum(axis=1)
    invalid = (values < -tolerance) | (values > 1+tolerance) | (increments < -tolerance)
    marked = np.zeros(len(data['area']), dtype=bool)
    marked[data['bound_owner'][invalid]] = True
    return marked, dict(minimum_damage=float(values.min()), maximum_damage=float(values.max()),
                        minimum_increment=float(increments.min()), invalid_cells=int(marked.sum()))


def near_tip(mesh, radius, tip=(0.0, 0.0)):
    """单元矩形到缺口尖端的距离不超过 radius。"""
    corners = mesh['nodes'][mesh['cells']]
    lo, hi = corners[:, 0], corners[:, 2]
    dx = np.maximum(0., np.maximum(lo[:, 0]-tip[0], tip[0]-hi[:, 0]))
    dy = np.maximum(0., np.maximum(lo[:, 1]-tip[1], tip[1]-hi[:, 1]))
    return np.hypot(dx, dy) <= radius


def damage_neighborhood(mesh, hot, radius):
    """把与损伤单元间隙不超过 radius 的单元一并标出，不指定裂纹方向。"""
    if radius <= 0 or not np.any(hot):
        return hot
    corners = mesh['nodes'][mesh['cells']]
    lo, hi = corners[:, 0], corners[:, 2]
    near = np.array(hot, dtype=bool, copy=True)
    sources = np.flatnonzero(hot)
    for start in range(0, len(sources), 64):
        ids = sources[start:start+64]
        dx = np.maximum(0., np.maximum(lo[ids, 0, None]-hi[None, :, 0], lo[None, :, 0]-hi[ids, 0, None]))
        dy = np.maximum(0., np.maximum(lo[ids, 1, None]-hi[None, :, 1], lo[None, :, 1]-hi[ids, 1, None]))
        near |= np.hypot(dx, dy).min(axis=0) <= radius
    return near


def mark(mesh, damage, data, indicator='maximum'):
    if not np.isfinite(damage).all():
        raise ValueError('Nonfinite damage')
    if indicator == 'center':
        dc = (damage[data['conn']]*data['center_n']).sum(axis=1)
    elif indicator == 'maximum':
        values = (damage[data['bound_conn']]*data['bound_N']).sum(axis=1)
        dc = np.full(len(data['area']), -np.inf)
        np.maximum.at(dc, data['bound_owner'], values)
    else:
        raise ValueError(f'Unknown damage indicator: {indicator}')
    # 损伤超过 0.1 即要求 h0/8，并外扩 0.03 mm。
    target = np.full(len(dc), np.inf)
    target[damage_neighborhood(mesh, dc > .1+1e-6, 0.03)] = mesh['h0']/8
    required = data['h'] > target*(1+1e-10)
    if np.any(required & (mesh['level'] >= 3)):
        raise RuntimeError('Damage-size target is unresolved at maximum refinement level')
    return required, dc, target


def refine(mesh, marked):
    """Four children per marked rectangle; closure limits adjacent levels to 2:1."""
    marked = np.asarray(marked, dtype=bool).copy()
    if marked.shape != (len(mesh['cells']),) or not marked.any():
        raise ValueError('Invalid or empty marked set')
    supports, atoms = connectivity(mesh)
    pairs = np.asarray([v for v in atoms.values() if len(v) == 2], dtype=np.int64)
    while len(pairs):
        future = mesh['level']+marked
        delta = future[pairs[:, 0]]-future[pairs[:, 1]]
        add = np.r_[pairs[delta > 1, 1], pairs[delta < -1, 0]]
        if not len(add):
            break
        marked[add] = True
    if np.any(marked & (mesh['level'] >= 3)):
        raise RuntimeError('Transition closure exceeds maximum refinement level')
    nodes = mesh['nodes'].tolist()
    lookup = {tuple(v): i for i, v in enumerate(nodes)}
    parent_for_node, new_cells, parents = [], [], []
    def node_at(p, parent):
        key = tuple(np.round(p, 14))
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(list(key))
            parent_for_node.append(parent)
        return lookup[key]
    for e, corners in enumerate(mesh['cells']):
        if not marked[e]:
            new_cells.append(corners.tolist()); parents.append(e)
            continue
        p = mesh['nodes'][corners]
        a, b, c, d = corners
        ab, bc, cd, da = [node_at((p[j]+p[(j+1)%4])/2, e) for j in range(4)]
        mid = node_at((p[0]+p[2])/2, e)
        new_cells.extend(((a, ab, mid, da), (ab, b, bc, mid),
                          (mid, bc, c, cd), (da, mid, cd, d)))
        parents.extend([e]*4)
    parents = np.asarray(parents, dtype=np.int64)
    new = dict(nodes=np.asarray(nodes), cells=np.asarray(new_cells, dtype=np.int64),
               level=mesh['level'][parents]+marked[parents], h0=mesh['h0'])
    max_nodes = max(map(len, supports))
    interp_ids = np.zeros((len(parent_for_node), max_nodes), dtype=np.int64)
    interp_n = np.zeros_like(interp_ids, dtype=float)
    for j, e in enumerate(parent_for_node):
        ids = supports[e]
        corners = mesh['nodes'][mesh['cells'][e]]
        point = 2*(new['nodes'][len(mesh['nodes'])+j]-corners[0])/(corners[2]-corners[0])-1
        interp_ids[j, :len(ids)] = ids
        interp_n[j, :len(ids)] = shape(local_nodes(mesh, e, ids), [point])[0][0]
    mapping = dict(old_count=len(mesh['nodes']), ids=interp_ids, N=interp_n,
                   child_parent=parents, marked=np.flatnonzero(marked))
    return new, mapping


def prolong(values, mapping):
    values = np.asarray(values)
    if len(values) != mapping['old_count']:
        raise ValueError('Field and old mesh do not match')
    add = np.einsum('ij,ij...->i...', mapping['N'], values[mapping['ids']])
    return np.concatenate((values.copy(), add))


def validate(mesh, data=None):
    data = quadrature(mesh) if data is None else data
    nodes = mesh['nodes']
    assert np.isfinite(nodes).all() and len(np.unique(nodes, axis=0)) == len(nodes)
    assert abs(data['area'].sum()-.9925) < 1e-10
    assert np.all(data['weight'] > 0) and abs(data['weight'].sum()-.9925) < 1e-10
    assert np.allclose(data['N'].sum(axis=1), 1, atol=1e-10)
    assert np.allclose(data['bound_N'].sum(axis=1), 1, atol=1e-10)
    assert np.allclose(data['grad'].sum(axis=1), 0, atol=1e-9)
    assert np.allclose(np.einsum('qi,qij->qj', data['N'], nodes[data['qconn']]), data['points'], atol=1e-11)
    perimeter = 0.
    for edge, owners in data['atoms'].items():
        assert len(owners) in (1, 2), 'Nonmanifold edge'
        if len(owners) == 2:
            assert abs(int(mesh['level'][owners[0]])-int(mesh['level'][owners[1]])) <= 1
            continue
        p = nodes[list(edge)]
        x, y = p.T
        physical = (np.allclose(y, -.5) or np.allclose(y, .5) or
                    np.allclose(x, -.5) or np.allclose(x, .5) or
                    (x.max() <= 1e-12 and (np.allclose(y, -.0075) or np.allclose(y, .0075))) or
                    (np.allclose(x, 0) and np.max(abs(y)) <= .0075+1e-12))
        assert physical, 'Unmatched interior edge'
        perimeter += np.linalg.norm(p[1]-p[0])
    assert abs(perimeter-5) < 1e-10
    return dict(nodes=len(nodes), elements=len(mesh['cells']),
                transition_elements=sum(len(s)>4 for s in data['supports']),
                h_min=float(data['h'].min()), h_max=float(data['h'].max()),
                level_max=int(mesh['level'].max()), quadrature_points=len(data['weight']))



# 4. Q4 能量、状态传递、优化与加载循环
class QuadSolver(DEMSolverBase):
    def __init__(self, mesh, network=None, lower=None):
        self.mesh = mesh
        self.data = quadrature(mesh)
        self.stats = validate(mesh, self.data)
        network = (network if network is not None else ChebyKANNetwork()).to(conf.device)
        super().__init__(network,
                         mesh['nodes'], mesh['cells'], self.data['area'])
        device = conf.device
        for name, key, dtype in [('qconn', 'qconn', torch.long), ('N', 'N', torch.float32),
                                  ('grad', 'grad', torch.float32), ('qw', 'weight', torch.float32),
                                  ('owner', 'owner', torch.long), ('support', 'conn', torch.long),
                                  ('support_mask', 'mask', torch.bool)]:
            setattr(self, name, torch.as_tensor(self.data[key], dtype=dtype, device=device))
        if lower is not None:
            self.alpha_history = torch.as_tensor(lower, dtype=torch.float32, device=device).clone()

    def fields(self, load):
        return torch.stack(self.compute_fields(load), dim=1)

    def at_quadrature(self, fields):
        nodal = fields[self.qconn]
        values = torch.einsum('qi,qij->qj', self.N, nodal)
        grads = torch.einsum('qia,qib->qab', nodal, self.grad)
        pp, pm = self.spectral_split(grads[:, 0, 0], grads[:, 1, 1],
                                     .5*(grads[:, 0, 1]+grads[:, 1, 0]))
        return values, grads, pp, pm

    def energy_of(self, fields, history):
        values, grads, pp, pm = self.at_quadrature(fields)
        d = values[:, 2]
        g, _ = self.pff.degrade(d)
        w, cw = self.pff.damage_fun(d)
        gc, ell = self.mat.Gc, self.mat.l0
        elastic = (((g+self.mat.k)*pp+pm)*self.qw).sum()
        fracture = ((gc/(cw*ell))*(w+ell**2*grads[:, 2].square().sum(dim=1))*self.qw).sum()
        hg = (self.N*history[self.qconn]).sum(dim=1)
        drive = torch.maximum(hg, pp)
        d_h = torch.clamp(2*ell*drive/(gc+2*ell*drive+1e-12), 0, 1)
        penalty = (.5*gc*self.pff.irr_penalty_weight()*torch.relu(-(d-.9*d_h)).square()*self.qw).sum()
        return elastic, fracture, penalty

    def get_total_energy(self, history, load):
        return self.energy_of(self.fields(load), history)

    def loss_total(self, history, load):
        fields = self.fields(load)
        elastic, fracture, penalty = self.energy_of(fields, history)
        smooth = self.smooth_energy(fields)
        rigid = conf.rigid_body_penalty*fields[self.center_node_idx, 0].square()
        return torch.log10(elastic+fracture+penalty+smooth+rigid+1e-12)

    def smooth_energy(self, fields):
        _, grad, _, _ = self.at_quadrature(fields)
        return conf.alpha_grad_penalty*(grad[:, 2].square().sum(dim=1)*self.qw).sum()/self.qw.sum()

    def next_history(self, fields, old_history):
        with torch.no_grad():
            _, _, pp, _ = self.at_quadrature(fields)
            means = torch.zeros_like(self.area)
            means.scatter_add_(0, self.owner, pp*self.qw)
            means /= self.area
            sums, counts = torch.zeros_like(old_history), torch.zeros_like(old_history)
            for j in range(self.support.shape[1]):
                valid = self.support_mask[:, j]
                ids = self.support[valid, j]
                sums.scatter_add_(0, ids, means[valid])
                counts.scatter_add_(0, ids, torch.ones_like(means[valid]))
            assert torch.all(counts > 0)
            return torch.maximum(old_history, sums/counts)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save_state(path, state):
    temp = Path(str(path)+'.tmp')
    torch.save(state, temp)
    temp.replace(path)


def field_error(actual, target, load):
    delta = actual-target
    return dict(displacement_relative=float(torch.sqrt(delta[:, :2].square().mean())/max(abs(load), 1e-6)),
                damage_max=float(delta[:, 2].abs().max()))


class IncompleteStep(RuntimeError):
    pass


def guarded_rprop_step(optimizer, objective, loss):
    """Backtrack a sign-gradient update that overshoots the current objective."""
    params = [p for group in optimizer.param_groups for p in group['params']]
    before = [p.detach().clone() for p in params]
    optimizer.step()
    delta = [p.detach()-old for p, old in zip(params, before)]
    factor = 1.
    roundoff = 8*torch.finfo(loss.dtype).eps*loss.detach().abs().clamp_min(1)
    with torch.no_grad():
        for backtracks in range(25):
            candidate = objective()
            if torch.isfinite(candidate) and candidate <= loss.detach()+roundoff:
                for state in optimizer.state.values():
                    state['step_size'].mul_(factor)
                return backtracks
            factor *= .5
            for p, old, change in zip(params, before, delta):
                p.copy_(old+factor*change)
        for p, old in zip(params, before):
            p.copy_(old)
    raise IncompleteStep('Rprop backtracking failed; last accepted state preserved')


def optimize(solver, history, load, optimizer, deadline, pretrain=False):
    c = conf
    epochs = c.n_epochs_pretrain
    stale, best = 0, float('inf')
    reference = solver.fields(load).detach().clone()
    change = dict(displacement_relative=float('inf'), damage_max=float('inf'))
    start = time.monotonic()
    backtracks = 0

    def objective():
        if pretrain:
            return solver.fields(0).square().sum(dim=1).mean()
        return (solver.loss_total(history, load)
                + c.weight_decay*sum(p.square().sum() for p in solver.network.parameters()))

    for epoch in (range(epochs) if pretrain else count()):
        if time.monotonic() >= deadline:
            raise IncompleteStep('wall_time_budget; last accepted checkpoint preserved')
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite objective')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in solver.network.parameters()):
            raise FloatingPointError('Nonfinite network gradient')
        # A refined mesh inherits network weights but resets Rprop's step sizes.
        # Protect that initial update; retain ordinary Rprop once its state exists.
        if pretrain or optimizer.state:
            optimizer.step()
        else:
            backtracks += guarded_rprop_step(optimizer, objective, loss)
        value = float(loss.detach())
        if epoch >= c.early_stop_min_epochs:
            if best-value > c.early_stop_tol:
                best, stale = value, 0
            else:
                stale += 1
        if (epoch+1) % 100 == 0:
            torch.cuda.synchronize()
            current = solver.fields(load).detach()
            change = field_error(current, reference, load)
            reference = current.clone()
            print(json.dumps(dict(stage='pretrain' if pretrain else 'solve',
                                  load=load, epoch=epoch+1, loss=value, change=change)), flush=True)
            if not pretrain and stale >= c.early_stop_patience and max(change.values()) <= 1e-3:
                return dict(reason='loss_stagnation_and_field_change', epochs=epoch+1, loss=value,
                            change=change, backtracks=backtracks, seconds=time.monotonic()-start)
    return dict(reason='legacy_pretrain_budget', epochs=epochs, loss=value, seconds=time.monotonic()-start)


def optimizer_for(solver):
    return torch.optim.Rprop(solver.network.parameters(), lr=conf.lr, step_sizes=(1e-10, 50))


def transfer(old, new_mesh, mapping, trial, history, load):
    old_lower = old.alpha_history.detach().cpu().numpy()
    lower = prolong(old_lower, mapping)
    h_new = prolong(history.detach().cpu().numpy(), mapping)
    target_np = prolong(trial.detach().cpu().numpy(), mapping)
    if lower.min() < -1e-7 or lower.max() > 1+1e-7 or h_new.min() < -1e-7:
        raise IncompleteStep('Transferred history overshoot; no clipping or hidden state correction')
    new = QuadSolver(new_mesh, network=copy.deepcopy(old.network), lower=lower)
    h_new = torch.as_tensor(h_new, dtype=torch.float32, device=new.inp.device)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=new.inp.device)
    # Evaluate the old variable-node polynomial at every new integration point.
    reference = np.empty((len(new.data['weight']), 3))
    old_values = trial.detach().cpu().numpy()
    offsets = np.r_[0, np.cumsum(np.bincount(new.data['owner'], minlength=len(new_mesh['cells'])))]
    for e in range(len(new_mesh['cells'])):
        parent = int(mapping['child_parent'][e])
        ids = old.data['supports'][parent]
        corners = old.mesh['nodes'][old.mesh['cells'][parent]]
        take = slice(offsets[e], offsets[e+1])
        points = 2*(new.data['points'][take]-corners[0])/(corners[2]-corners[0])-1
        n = shape(local_nodes(old.mesh, parent, ids), points)[0]
        reference[take] = n@old_values[ids]
    prolonged_gp = np.einsum('qi,qij->qj', new.data['N'], target_np[new.data['qconn']])
    interp_error = float(abs(reference-prolonged_gp).max())
    if interp_error > 2e-6:
        raise IncompleteStep(f'Variable-node field prolongation error {interp_error}')
    before = field_error(new.fields(load).detach(), target, load)
    # The trial is an initial guess, not irreversible history. The new mesh must
    # be solved at this same load before it can be accepted.
    return new, h_new, optimizer_for(new), dict(interpolation_max=interp_error,
               network_initial_error=before, initialization='inherited weights; same-load solve required',
               projected_trial_lower_gap_min=float(np.min(target_np[:, 2]-lower)))


def plot_mesh(mesh, destination, title):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    _, atoms = connectivity(mesh)
    segments = mesh['nodes'][np.asarray(list(atoms))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.add_collection(LineCollection(segments, colors='black', linewidths=.35))
        ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    axes[0].set(xlim=(-.51, .51), ylim=(-.51, .51), title=title)
    axes[1].set(xlim=(-.12, .18), ylim=(-.15, .15), title='Notch-tip detail')
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results_q4_adaptive'))
    parser.add_argument('--budget-seconds', type=float, default=7200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--case', choices=('fixed', 'center', 'seeded', 'cold', 'uniform'), default='seeded')
    parser.add_argument('--initial-weights', type=Path, default=Path(__file__).with_name('initial_weights.pt'))
    parser.add_argument('--seed-mesh', type=Path, default=Path(__file__).with_name('tip_seeded.npz'))
    parser.add_argument('--stop-step', type=int, default=40)
    parser.add_argument('--loading', choices=('shear', 'tension'), default='shear')
    parser.add_argument('--pre-refine', type=int, default=0)
    parser.add_argument('--tip-seed', type=float, default=0.0)
    parser.add_argument('--max-cycles', type=int, default=13)
    args = parser.parse_args()
    if args.loading == 'tension':
        conf.u_max, conf.n_steps = 0.008, 41
    else:
        conf.u_max, conf.n_steps = 0.015, 41
    conf.loading = args.loading
    conf.disp_steps = np.linspace(0.0, conf.u_max, conf.n_steps)
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error('budget-seconds must be positive and finite')
    if not 0 <= args.stop_step < len(conf.disp_steps):
        parser.error('stop-step must be in the prescribed load array')
    if args.max_cycles < 1:
        parser.error('max-cycles must be positive')
    if args.pre_refine < 0:
        parser.error('pre-refine must be zero or positive')
    if args.tip_seed < 0:
        parser.error('tip-seed must be zero or positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA cloud execution is required')
    torch.set_num_threads(4)
    torch.manual_seed(conf.seed); np.random.seed(conf.seed)
    conf.device = 'cuda'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    use_weights = args.case not in ('cold', 'uniform')
    use_seed = args.case == 'seeded'
    identity = dict(source={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                           for p in (Path(__file__).name,)},
                    initial_h0=.05, thresholds=[[.1, 8]],
                    halo_threshold=.1, halo_radius=0.03,
                    pre_refine=args.pre_refine,
                    tip_seed=args.tip_seed, max_level=3,
                    field_change_gate=1e-3,
                    training_limit='wall time budget; convergence gates unchanged',
                    state_transfer='exact accepted d/H prolongation; inherited network weights; same-load solve',
                    admissibility_refinement='damage bounds and irreversibility at every smooth-patch vertex',
                    material=dict(E=conf.mat_E, nu=conf.mat_nu, Gc=conf.Gc, l0=conf.l0),
                    loads=conf.disp_steps.tolist(),
                    integration='2x2 Gauss per smooth reference subrectangle; true variable-node quads',
                    case=args.case, loading=args.loading, max_refine_cycles=args.max_cycles,
                    marker='center' if args.case == 'center' else 'maximum',
                    marker_tolerance=1e-6, smooth='0.005 * domain area average of damage gradient squared',
                    initial_weights_sha256=(hashlib.sha256(args.initial_weights.read_bytes()).hexdigest()
                                            if use_weights else None),
                    seed_mesh_sha256=(hashlib.sha256(args.seed_mesh.read_bytes()).hexdigest()
                                      if use_seed else None),
                    scope='Matched Q4 validation; revised spatial regularizer; legacy spectral model retained')
    if args.resume:
        if json.loads((out/'manifest.json').read_text())['identity'] != identity:
            raise RuntimeError('Resume source/config mismatch')
        accepted = torch.load(out/'accepted.pt', weights_only=False, map_location='cuda')
    else:
        if (out/'manifest.json').exists():
            raise FileExistsError('Output exists; use a new directory or --resume')
        write_json(out/'manifest.json', dict(identity=identity, gpu=torch.cuda.get_device_name(),
                                           torch=torch.__version__, cuda=torch.version.cuda))
        mesh = initial_mesh()
        if args.case in ('fixed', 'uniform'):
            for _ in range(3):
                mesh, _ = refine(mesh, np.ones(len(mesh['cells']), dtype=bool))
        elif args.case == 'seeded':
            with np.load(args.seed_mesh) as saved:
                mesh = {k: saved[k] for k in saved.files}
        elif args.pre_refine:
            for _ in range(args.pre_refine):
                mesh, _ = refine(mesh, np.ones(len(mesh['cells']), dtype=bool))
        if args.tip_seed > 0 and args.case not in ('fixed', 'uniform', 'seeded'):
            for _ in range(3):
                hot = near_tip(mesh, args.tip_seed) & (mesh['level'] < 3)
                if not np.any(hot):
                    break
                mesh, _ = refine(mesh, hot)
        solver = QuadSolver(mesh)
        if use_weights:
            initial = torch.load(args.initial_weights, weights_only=False, map_location='cuda')
            solver.network.load_state_dict(initial['network'])
        plot_mesh(mesh, out/'initial_mesh.png', f'Initial Q4 mesh: {len(mesh["cells"])} cells')
        np.savez_compressed(out/'initial_mesh.npz', **mesh)
        write_json(out/'initial_mesh_stats.json', solver.stats)
        if args.preview_only:
            write_json(out/'status.json', dict(status='preview_only', stats=solver.stats)); return
        accepted = dict(mesh=mesh, network=copy.deepcopy(solver.network.state_dict()),
                        damage=np.zeros(len(mesh['nodes']), dtype=np.float32),
                        H=np.zeros(len(mesh['nodes']), dtype=np.float32), next_step=0, rows=[],
                        pretrained=use_weights)
        save_state(out/'accepted.pt', accepted)
    started = time.monotonic(); deadline = started+args.budget_seconds
    write_json(out/'status.json', dict(status='running', pid=os.getpid(), next_step=accepted['next_step']))
    try:
        solver = QuadSolver(accepted['mesh'], lower=accepted['damage'])
        solver.network.load_state_dict(accepted['network'])
        history = torch.as_tensor(accepted['H'], dtype=torch.float32, device='cuda')
        opt = optimizer_for(solver)
        if 'optimizer' in accepted:
            opt.load_state_dict(accepted['optimizer'])
        if not accepted['pretrained']:
            optimize(solver, history, 0., opt, deadline, pretrain=True)
            accepted.update(network=copy.deepcopy(solver.network.state_dict()), pretrained=True,
                            optimizer=copy.deepcopy(opt.state_dict()))
            save_state(out/'accepted.pt', accepted)
        for step in range(accepted['next_step'], args.stop_step+1):
            load = float(conf.disp_steps[step]); rounds = []
            solver.alpha_history = torch.as_tensor(accepted['damage'], dtype=torch.float32, device='cuda').clone()
            history = torch.as_tensor(accepted['H'], dtype=torch.float32, device='cuda')
            for cycle in range(args.max_cycles):
                training = (dict(reason='legacy_zero_load_skip', epochs=0) if load == 0 else
                            optimize(solver, history, load, opt, deadline))
                trial = solver.fields(load).detach()
                marked, dc, target = mark(solver.mesh, trial[:, 2].cpu().numpy(), solver.data,
                                             indicator='center' if args.case == 'center' else 'maximum')
                if args.case in ('fixed', 'uniform') and marked.any():
                    raise IncompleteStep('Fixed reference does not meet damage resolution target')
                invalid, bounds = admissibility(trial[:, 2].cpu().numpy(),
                                                   solver.alpha_history.cpu().numpy(), solver.data)
                if args.case in ('fixed', 'uniform') and invalid.any():
                    raise IncompleteStep('Fixed reference violates internal admissibility')
                indicator_count = int(marked.sum())
                marked |= invalid
                if not marked.any():
                    break
                if cycle == args.max_cycles-1:
                    raise IncompleteStep('Refinement round limit; step not accepted')
                new_mesh, mapping = refine(solver.mesh, marked)
                if len(new_mesh['cells']) > 50000:
                    raise IncompleteStep('Element cap; step not accepted')
                np.savez_compressed(out/f'mapping_{step:03d}_{cycle}.npz', **mapping,
                                    old_nodes=solver.mesh['nodes'], old_cells=solver.mesh['cells'],
                                    new_nodes=new_mesh['nodes'], new_cells=new_mesh['cells'])
                solver, history, opt, metrics = transfer(solver, new_mesh, mapping, trial, history, load)
                rounds.append(dict(cycle=cycle, stats=solver.stats, transfer=metrics,
                                   damage_marked=indicator_count, admissibility_marked=int(invalid.sum())))
                write_json(out/f'refine_{step:03d}_{cycle}.json', rounds[-1])
                print('REFINED', json.dumps(rounds[-1]), flush=True)
            if not torch.isfinite(trial).all() or torch.any(trial[:, 2] < solver.alpha_history-1e-6):
                raise IncompleteStep('Finite/bounded/irreversibility gate failed')
            if torch.any(trial[:, 2] < 0) or torch.any(trial[:, 2] > 1):
                raise IncompleteStep('Nodal damage bound failed')
            with torch.no_grad():
                # Variable-node bases need not be positive: nodal bounds alone are insufficient.
                dg = (solver.N*trial[solver.qconn, 2]).sum(dim=1)
                lower_g = (solver.N*solver.alpha_history[solver.qconn]).sum(dim=1)
                if dg.min() < -1e-6 or dg.max() > 1+1e-6 or (dg-lower_g).min() < -1e-6:
                    raise IncompleteStep('Quadrature damage bound/irreversibility gate failed')
                energies = [float(e) for e in solver.energy_of(trial, history)]
                eps = 1e-6
                if load == 0:
                    force = 0.
                else:
                    plus = sum(solver.get_total_energy(history, load+eps))
                    minus = sum(solver.get_total_energy(history, load-eps))
                    force = float((plus-minus)/(2*eps))
                new_history = solver.next_history(trial, history)
            row = dict(step=step, load_mm=load, damage_max=float(trial[:, 2].max()),
                       force_N=force, E_elastic=energies[0], E_fracture=energies[1], E_penalty=energies[2],
                       S_area=float(solver.smooth_energy(trial)),
                       physical_scaling_force_N=2*energies[0]/load if load else 0.,
                       mesh=solver.stats, training=training, refinements=len(rounds),
                       field_bounds=bounds,
                       elapsed_seconds=time.monotonic()-started)
            accepted = dict(mesh=solver.mesh, network=copy.deepcopy(solver.network.state_dict()),
                            optimizer=copy.deepcopy(opt.state_dict()), damage=trial[:, 2].cpu().numpy(),
                            decoder_lower=solver.alpha_history.cpu().numpy(), H=new_history.cpu().numpy(),
                            next_step=step+1, rows=accepted['rows']+[row], pretrained=True)
            save_state(out/'accepted.pt', accepted)
            np.savez_compressed(out/f'step_{step:03d}.npz', **solver.mesh, fields=trial.cpu().numpy(),
                                H=accepted['H'], decoder_lower=accepted['decoder_lower'], load=load)
            write_json(out/'results.json', accepted['rows'])
            write_json(out/'status.json', dict(status='running', pid=os.getpid(), next_step=step+1, last=row))
            if rounds or step in (0, 40):
                plot_mesh(solver.mesh, out/f'mesh_{step:03d}.png', f'Q4/variable-node mesh, u={load:.6f} mm')
            print('ACCEPTED', json.dumps(row), flush=True)
        write_json(out/'status.json', dict(status='completed' if args.stop_step == 40 else 'stage_complete',
                   stop_step=args.stop_step, accepted_steps=accepted['next_step'],
                   elapsed_seconds=time.monotonic()-started,
                   qualification='training and mesh gates passed; no independent force/path reference validation yet'))
    except BaseException as exc:
        write_json(out/'status.json', dict(status='incomplete' if isinstance(exc, IncompleteStep) else 'failed',
                   error=repr(exc), next_step=accepted['next_step'], elapsed_seconds=time.monotonic()-started))
        traceback.print_exc(); raise


if __name__ == '__main__':
    main()
