"""Anisotropic SENT with ChebyKAN on a Q4 mesh.

Triangle SENT results are the benchmark. This solver mirrors the proven
isotropic ChebyKAN_Q4_Adaptive.py recipe that fractures tension and shear:
ONE ChebyKAN outputs (u, v, d); ONE joint energy (elastic + fracture +
irreversibility penalty + smooth + rigid) is minimized with guarded Rprop so
displacement and damage grow together. The crack is a resolved geometric slit,
not a sub-mesh wedge. The damage field is not a finite-element solve.

Only the physics is anisotropic: the elastic energy is the full orthotropic
energy degraded by (1-d)^2 (triangle assumption psi^- = 0), the fracture
energy uses the A1 structural tensor, and the damage driving force is the sum
of the Yu fiber/matrix/shear components.
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "composite"))
import ChebyKAN_Q4_Adaptive as cq
import dsolve

# anisotropic material (same as the triangle benchmark)
E11, E22, G12, NU12 = 114.8e3, 11.7e3, 9.66e3, 0.24
GC, L, BETA = 0.2774, 0.01, 300.0
# Two-stage mark: a compact tip disk nucleates localization, then the crack
# core is tracked with a tight halo so the broad beta-smeared shoulder and the
# horizontal ligament are left coarse.
SEED_RADIUS, MARK_SWITCH = 0.03, 0.3
MARK_CORE, HALO_FACTOR = 0.3, 1.0
# Fiber-aligned neighborhoods: the anisotropic regularization reaches
# l*sqrt(1+beta) along the fiber and l across it, so the seed and the core
# halo are ellipses in that metric instead of Euclidean disks.
ELLIPSE = False
SEED_ACROSS = 2.0 * L
# Semi-axis of the seed behind the tip (s < 0 along the fiber); None keeps it
# symmetric. The crack only runs forward, so the rear lobe sees little damage.
SEED_BACK = None
HALO_ALONG = L * np.sqrt(1.0 + BETA)
FIBER = 0.0
U_MAX, N_STEPS = 0.02, 40


class AnisoSolver(cq.QuadSolver):
    def __init__(self, mesh, network=None, lower=None):
        super().__init__(mesh, network=network, lower=lower)
        # Padded support slots (zero shape weight) all index node 0; the
        # gather backward then serializes over tens of thousands of duplicates.
        # Point them at a real node of the same element instead.
        pad = ~torch.as_tensor(self.data["mask"], device=self.qconn.device)[self.owner]
        if bool(pad.any()):
            assert float(self.N[pad].abs().max()) == 0.0 and float(self.grad[pad].abs().max()) == 0.0
            self.qconn = torch.where(pad, self.qconn[:, :1].expand_as(self.qconn), self.qconn)

    def set_material(self, theta_rad):
        device = cq.conf.device
        self.theta = float(theta_rad)
        mat = dsolve.CompositeMaterial(
            E11, E22, G12, NU12, lambda x, y: torch.full_like(x, float(theta_rad)))
        self.Q11, self.Q12, self.Q22, self.Q66 = mat.Q11, mat.Q12, mat.Q22, mat.Q66
        theta_t = torch.tensor(float(theta_rad), dtype=torch.float32, device=device)
        self.Qmat = mat.get_global_stiffness(theta_t).to(device)  # [3,3]
        self.cos_t = float(np.cos(theta_rad))
        self.sin_t = float(np.sin(theta_rad))

    def compute_fields(self, load):
        out = self.network(self.inp)
        if not torch.is_tensor(load):
            load = torch.tensor(load, device=cq.conf.device, dtype=torch.float32)
        y = self.inp[:, 1]
        y0, yL = -cq.conf.Ly / 2, cq.conf.Ly / 2
        rise = y - y0                     # y + 0.5, zero on the bottom edge
        bubble = (y - y0) * (yL - y)      # zero on both loaded edges
        linear = (y - y0) / (yL - y0)     # 0 at bottom, 1 at top
        # orig SENT BC: u_x free except on the bottom; u_y = load at the top.
        u = rise * out[:, 0] * load
        v = (bubble * out[:, 1] + linear) * load
        alpha_net = self.alpha_constraint(out[:, 2])
        alpha = self.irreversible(alpha_net, self.alpha_history)
        return u, v, alpha

    def _eps_values_grads(self, fields):
        nodal = fields[self.qconn]
        values = torch.einsum("qi,qij->qj", self.N, nodal)
        grads = torch.einsum("qia,qib->qab", nodal, self.grad)  # [q,3,2]
        eps = torch.stack(
            (grads[:, 0, 0], grads[:, 1, 1], grads[:, 0, 1] + grads[:, 1, 0]), dim=1)
        return eps, values, grads

    def _driving(self, eps):
        exx, eyy, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
        c, s = self.cos_t, self.sin_t
        c2, s2, cs = c * c, s * s, c * s
        e11 = c2 * exx + cs * gxy + s2 * eyy
        e22 = s2 * exx - cs * gxy + c2 * eyy
        g12 = -2.0 * cs * exx + 2.0 * cs * eyy + (c2 - s2) * gxy
        e11p, e22p = torch.relu(e11), torch.relu(e22)
        s11p = self.Q11 * e11p + self.Q12 * e22p
        s22p = self.Q12 * e11p + self.Q22 * e22p
        tau = self.Q66 * g12
        return 0.5 * e11p * s11p + 0.5 * e22p * s22p + 0.5 * g12 * tau

    def energy_of(self, fields, history):
        eps, values, grads = self._eps_values_grads(fields)
        d = values[:, 2]
        g = (1.0 - d) ** 2
        gc, ell = self.mat.Gc, self.mat.l0
        psi_full = 0.5 * torch.einsum("qi,ij,qj->q", eps, self.Qmat, eps)
        elastic = ((g + self.mat.k) * psi_full * self.qw).sum()
        gx, gy = grads[:, 2, 0], grads[:, 2, 1]
        along = self.cos_t * gx + self.sin_t * gy
        struct = gx.square() + gy.square() + BETA * along.square()
        fracture = (gc * (d.square() / (2.0 * ell) + 0.5 * ell * struct) * self.qw).sum()
        hg = (self.N * history[self.qconn]).sum(dim=1)
        drive = torch.maximum(hg, self._driving(eps))
        d_h = torch.clamp(2.0 * ell * drive / (gc + 2.0 * ell * drive + 1e-12), 0.0, 1.0)
        penalty = (0.5 * gc * self.pff.irr_penalty_weight()
                   * torch.relu(-(d - 0.9 * d_h)).square() * self.qw).sum()
        return elastic, fracture, penalty

    def next_history(self, fields, old_history):
        with torch.no_grad():
            eps, _values, _grads = self._eps_values_grads(fields)
            drive = self._driving(eps)
            means = torch.zeros_like(self.area)
            means.scatter_add_(0, self.owner, drive * self.qw)
            means /= self.area
            sums = torch.zeros_like(old_history)
            counts = torch.zeros_like(old_history)
            for j in range(self.support.shape[1]):
                valid = self.support_mask[:, j]
                ids = self.support[valid, j]
                sums.scatter_add_(0, ids, means[valid])
                counts.scatter_add_(0, ids, torch.ones_like(means[valid]))
            assert torch.all(counts > 0)
            return torch.maximum(old_history, sums / counts)


def build_fixed_mesh(refine_times):
    mesh = cq.initial_mesh()
    for _ in range(refine_times):
        mesh, _ = cq.refine(mesh, np.ones(len(mesh["cells"]), dtype=bool))
    return mesh


def fiber_metric(along, across, fiber=None):
    fiber = FIBER if fiber is None else fiber
    a = np.array([np.cos(fiber), np.sin(fiber)])
    n = np.array([-a[1], a[0]])
    return np.outer(a, a) / along ** 2 + np.outer(n, n) / across ** 2


def box_min_quadratic(xl, xh, yl, yh, M):
    """min g^T M g over the boxes [xl,xh] x [yl,yh] (M symmetric positive definite)."""
    m00, m01, m11 = M[0, 0], M[0, 1], M[1, 1]
    best = np.full(np.broadcast(xl, yl).shape, np.inf)
    for x0 in (xl, xh):
        y = np.clip(-m01 * x0 / m11, yl, yh)
        best = np.minimum(best, m00 * x0 * x0 + 2 * m01 * x0 * y + m11 * y * y)
    for y0 in (yl, yh):
        x = np.clip(-m01 * y0 / m00, xl, xh)
        best = np.minimum(best, m00 * x * x + 2 * m01 * x * y0 + m11 * y0 * y0)
    inside = (xl <= 0) & (xh >= 0) & (yl <= 0) & (yh >= 0)
    return np.where(inside, 0.0, best)


def ellipse_near_tip(mesh, along, across, tip=(0.0, 0.0), fiber=None):
    corners = mesh["nodes"][mesh["cells"]]
    lo, hi = corners[:, 0], corners[:, 2]
    M = fiber_metric(along, across, fiber)
    return box_min_quadratic(lo[:, 0] - tip[0], hi[:, 0] - tip[0],
                             lo[:, 1] - tip[1], hi[:, 1] - tip[1], M) <= 1.0


def seed_near_tip(mesh, along, across, back, fiber=None):
    """Tip seed: fiber ellipse ahead of the tip, shorter ellipse behind it."""
    if back is None:
        return ellipse_near_tip(mesh, along, across, fiber=fiber)
    fiber = FIBER if fiber is None else fiber
    corners = mesh["nodes"][mesh["cells"]]
    s_max = (corners @ np.array([np.cos(fiber), np.sin(fiber)])).max(axis=1)
    ahead = ellipse_near_tip(mesh, along, across, fiber=fiber) & (s_max >= 0.0)
    return ahead | ellipse_near_tip(mesh, back, across, fiber=fiber)


def ellipse_neighborhood(mesh, hot, along, across, fiber=None, targets=None):
    """Cells whose rectangle meets a fiber-aligned ellipse around any hot cell.

    Only cells in ``targets`` (default: all) are tested; the rest keep their
    ``hot`` value.
    """
    near = np.array(hot, dtype=bool, copy=True)
    tested = ~near if targets is None else (~near & targets)
    if not near.any() or not tested.any():
        return near
    corners = mesh["nodes"][mesh["cells"]]
    lo, hi = corners[:, 0], corners[:, 2]
    M = fiber_metric(along, across, fiber)
    cen = 0.5 * (lo + hi)
    half = 0.5 * np.hypot(*(hi - lo).T)
    src, tgt = np.flatnonzero(near), np.flatnonzero(tested)
    reach = max(along, across) + half[src].max() + half[tgt].max()
    lists = cKDTree(cen[tgt]).query_ball_point(cen[src], reach)
    s = np.repeat(src, [len(x) for x in lists])
    t = tgt[np.concatenate([np.asarray(x, dtype=int) for x in lists])] if len(s) else np.zeros(0, int)
    if len(s):
        q = box_min_quadratic(lo[t, 0] - hi[s, 0], hi[t, 0] - lo[s, 0],
                              lo[t, 1] - hi[s, 1], hi[t, 1] - lo[s, 1], M)
        near[t[q <= 1.0]] = True
    return near


def aniso_mark(mesh, damage, data):
    """Two-stage mark: a geometric tip seed, then a tight crack-core halo.

    Localization needs early refinement: on the coarse mesh damage first
    plateaus near 0.1 and only climbs to 1 once the notch region is already
    refined. A damage-value seed (d > 0.1) refines the whole beta-smeared
    shoulder and the horizontal ligament ahead of the notch, which is the
    diffuse over-refinement we want to avoid. Instead, while the damage peak
    is still low we refine only a compact disk at the notch tip; that is
    enough to nucleate the crack. Once the peak passes MARK_SWITCH the crack
    has localized and refinement is monotone, so we track only the crack core
    (d > MARK_CORE) with a tight HALO_FACTOR*l halo and leave the shoulder
    and ligament coarse.
    """
    values = (damage[data["bound_conn"]] * data["bound_N"]).sum(axis=1)
    dc = np.full(len(data["area"]), -np.inf)
    np.maximum.at(dc, data["bound_owner"], values)
    finite = np.isfinite(dc)
    dmax = float(dc[finite].max()) if finite.any() else 0.0
    if ELLIPSE and dmax <= MARK_SWITCH:
        hot = seed_near_tip(mesh, SEED_RADIUS, SEED_ACROSS, SEED_BACK)
    elif ELLIPSE:
        hot = ellipse_neighborhood(mesh, dc > MARK_CORE + 1e-6, HALO_ALONG, HALO_FACTOR * L,
                                   targets=data["h"] > mesh["h0"] / 8.0 * (1.0 + 1e-10))
    elif dmax <= MARK_SWITCH:
        hot = cq.near_tip(mesh, SEED_RADIUS)
    else:
        hot = cq.damage_neighborhood(mesh, dc > MARK_CORE + 1e-6, HALO_FACTOR * L)
    target = np.full(len(dc), np.inf)
    target[hot] = mesh["h0"] / 8.0
    required = data["h"] > target * (1.0 + 1e-10)
    if np.any(required & (mesh["level"] >= 3)):
        raise RuntimeError("Damage-size target is unresolved at maximum refinement level")
    return required


def aniso_transfer(old, new_mesh, mapping, trial, history, load, theta):
    old_lower = old.alpha_history.detach().cpu().numpy()
    lower = cq.prolong(old_lower, mapping)
    h_new = cq.prolong(history.detach().cpu().numpy(), mapping)
    target_np = cq.prolong(trial.detach().cpu().numpy(), mapping)
    if lower.min() < -1e-7 or lower.max() > 1 + 1e-7 or h_new.min() < -1e-7:
        raise cq.IncompleteStep("Transferred history overshoot")
    new = AnisoSolver(new_mesh, network=copy.deepcopy(old.network), lower=lower)
    new.set_material(theta)
    h_new = torch.as_tensor(h_new, dtype=torch.float32, device=new.inp.device)
    reference = np.empty((len(new.data["weight"]), 3))
    old_values = trial.detach().cpu().numpy()
    offsets = np.r_[0, np.cumsum(np.bincount(new.data["owner"], minlength=len(new_mesh["cells"])))]
    for element in range(len(new_mesh["cells"])):
        parent = int(mapping["child_parent"][element])
        ids = old.data["supports"][parent]
        corners = old.mesh["nodes"][old.mesh["cells"][parent]]
        take = slice(offsets[element], offsets[element + 1])
        points = 2 * (new.data["points"][take] - corners[0]) / (corners[2] - corners[0]) - 1
        basis = cq.shape(cq.local_nodes(old.mesh, parent, ids), points)[0]
        reference[take] = basis @ old_values[ids]
    prolonged = np.einsum("qi,qij->qj", new.data["N"], target_np[new.data["qconn"]])
    interp_error = float(abs(reference - prolonged).max())
    if interp_error > 2e-6:
        raise cq.IncompleteStep("Variable-node field prolongation error %s" % interp_error)
    return new, h_new, optimizer_for(new), interp_error


FAST = False


def optimizer_for(solver):
    if not FAST:
        return cq.optimizer_for(solver)
    return torch.optim.Rprop(solver.network.parameters(), lr=cq.conf.lr,
                             step_sizes=(1e-10, 50), capturable=True)


def preseed(mesh):
    """Refine the geometric tip seed before the first solve."""
    while True:
        marked = aniso_mark(mesh, np.zeros(len(mesh["nodes"])), cq.quadrature(mesh))
        if not marked.any():
            return mesh
        mesh, _ = cq.refine(mesh, marked)


def fast_optimize(solver, history, load, optimizer, deadline, check=None, check_every=0):
    """cq.optimize with each epoch replayed from a CUDA graph.

    Epoch sequence, Rprop update, guarded first step, stagnation counter and
    100-epoch field-change test are those of cq.optimize; losses are buffered
    on the device and read once per 100 epochs. ``check`` (called every
    ``check_every`` epochs) may end the solve early so the caller can refine.
    """
    c = cq.conf
    block = 100
    params = list(solver.network.parameters())
    load_t = torch.tensor(float(load), device="cuda", dtype=torch.float32)

    def objective():
        return (solver.loss_total(history, load_t)
                + c.weight_decay * sum(p.square().sum() for p in params))

    reference = solver.fields(load_t).detach().clone()
    start = time.monotonic()
    backtracks, stale, best = 0, 0, float("inf")
    values = []
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=False)
            loss = objective()
            loss.backward()
            if optimizer.state:
                optimizer.step()
            else:
                backtracks += cq.guarded_rprop_step(optimizer, objective, loss)
            values.append(loss.detach().clone())
    torch.cuda.current_stream().wait_stream(side)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    buf = torch.zeros(block, device="cuda")
    slot = torch.zeros(1, dtype=torch.long, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        static_loss = objective()
        static_loss.backward()
        optimizer.step()
        buf.index_copy_(0, slot, static_loss.detach().reshape(1))
        slot.add_(1)

    epoch = 0
    pending = [float(v) for v in values]
    while True:
        if time.monotonic() >= deadline:
            raise cq.IncompleteStep("wall_time_budget; last accepted checkpoint preserved")
        todo = block - len(pending)
        slot.zero_()
        for _ in range(todo):
            graph.replay()
        losses = pending + buf[:todo].tolist()
        pending = []
        for value in losses:
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite objective")
            if epoch >= c.early_stop_min_epochs:
                if best - value > c.early_stop_tol:
                    best, stale = value, 0
                else:
                    stale += 1
            epoch += 1
        if not all(torch.isfinite(p).all() for p in params):
            raise FloatingPointError("Nonfinite network parameters")
        current = solver.fields(load_t).detach()
        change = cq.field_error(current, reference, load)
        reference = current.clone()
        print(json.dumps(dict(stage="solve", load=load, epoch=epoch, loss=losses[-1],
                              change=change)), flush=True)
        if stale >= c.early_stop_patience and max(change.values()) <= 1e-3:
            return dict(reason="loss_stagnation_and_field_change", epochs=epoch, loss=losses[-1],
                        change=change, backtracks=backtracks, seconds=time.monotonic() - start)
        if check is not None and check_every and epoch % check_every == 0 and check(current):
            return dict(reason="early_refine", epochs=epoch, loss=losses[-1], change=change,
                        backtracks=backtracks, seconds=time.monotonic() - start)


def solve(solver, history, load, optimizer, deadline, check=None, check_every=0):
    if FAST:
        return fast_optimize(solver, history, load, optimizer, deadline, check, check_every)
    return cq.optimize(solver, history, load, optimizer, deadline)


def save_step(out_dir, step, solver, trial, history, load, t0, extra=None):
    """Per-step field and wall-time record; does not change the computation."""
    np.savez_compressed(out_dir / ("step_%03d.npz" % step), **solver.mesh,
                        fields=trial.detach().cpu().numpy(),
                        H=history.detach().cpu().numpy(), load=load)
    row = dict(step=step, U=load, cells=int(len(solver.mesh["cells"])),
               nodes=int(len(solver.mesh["nodes"])),
               quadrature_points=int(len(solver.data["weight"])),
               seconds=time.monotonic() - t0)
    row.update(extra or {})
    with (out_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_status(out_dir, t0, **extra):
    status = dict(status="completed", elapsed_seconds=time.monotonic() - t0,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  mark_switch=MARK_SWITCH, mark_core=MARK_CORE, seed_radius=SEED_RADIUS,
                  halo_factor=HALO_FACTOR, ellipse=ELLIPSE, seed_across=SEED_ACROSS,
                  halo_along=HALO_ALONG, seed_back=SEED_BACK)
    status.update(extra)
    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def run(angle, out_dir, refine_times, budget):
    t0 = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cq.conf.device = "cuda"
    cq.conf.loading = "tension"
    cq.conf.u_max, cq.conf.n_steps = U_MAX, N_STEPS + 1
    cq.conf.Gc, cq.conf.l0 = GC, L
    cq.conf.disp_steps = np.linspace(0.0, U_MAX, N_STEPS + 1)
    torch.manual_seed(cq.conf.seed)
    np.random.seed(cq.conf.seed)

    mesh = build_fixed_mesh(refine_times)
    solver = AnisoSolver(mesh)
    solver.set_material(np.deg2rad(float(angle)))
    n_params = sum(p.numel() for p in solver.network.parameters())
    print("ChebyKAN Q4 sent %g: %d cells, %d nodes, %d params, h_min %.5f" % (
        angle, len(mesh["cells"]), len(mesh["nodes"]), n_params,
        solver.stats["h_min"]), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    force_path = out_dir / "force.csv"
    with force_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["step", "U", "force", "force_traction", "elements", "damage_max"])

    opt = optimizer_for(solver)
    deadline = time.monotonic() + budget
    history = torch.zeros(solver.inp.shape[0], device="cuda")
    cq.optimize(solver, history, 0.0, opt, deadline, pretrain=True)

    peak = 0.0
    damage_lower = torch.zeros(solver.inp.shape[0], device="cuda")
    for step in range(1, N_STEPS + 1):
        load = float(cq.conf.disp_steps[step])
        solver.alpha_history = damage_lower.clone()
        info = solve(solver, history, load, opt, deadline)
        with torch.no_grad():
            trial = solver.fields(load).detach()
            eps = 1e-6
            plus = sum(solver.get_total_energy(history, load + eps))
            minus = sum(solver.get_total_energy(history, load - eps))
            force = float((plus - minus) / (2.0 * eps))
            # Clapeyron reaction: for single-parameter displacement loading the
            # vertical reaction equals 2*E_elastic/U, i.e. the degraded traction
            # integral used by the triangle benchmark and the Julia FEM. The
            # energy central difference above also carries the irreversibility
            # penalty derivative, which inflates it once damage lags.
            elastic = float(solver.energy_of(trial, history)[0])
            force_traction = 2.0 * elastic / load if load > 0 else 0.0
            d_max = float(trial[:, 2].max())
            history = solver.next_history(trial, history)
            damage_lower = trial[:, 2].clone()
        save_step(out_dir, step, solver, trial, history, load, t0,
                  dict(epochs=info.get("epochs")))
        with force_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [step, load, force, force_traction, len(solver.mesh["cells"]), d_max])
        print("ACCEPTED step %d U %.5f F %.4e Ftrac %.4e cells %d d %.3f epochs %s" % (
            step, load, force, force_traction, len(solver.mesh["cells"]), d_max,
            info.get("epochs")), flush=True)
        peak = max(peak, abs(force))
        if d_max > 0.9 and peak > 20.0 and abs(force) < 5.0:
            print("STOP force %.4e near zero after fracture" % force, flush=True)
            break
    np.savez_compressed(out_dir / "final_mesh.npz", **solver.mesh,
                        damage=damage_lower.cpu().numpy())
    write_status(out_dir, t0, mode="fixed", angle=float(angle), refine=refine_times, fast=FAST)
    return 0


def accept_row(solver, history, trial, load):
    eps = 1e-6
    plus = sum(solver.get_total_energy(history, load + eps))
    minus = sum(solver.get_total_energy(history, load - eps))
    force = float((plus - minus) / (2.0 * eps))
    elastic = float(solver.energy_of(trial, history)[0])
    traction = 2.0 * elastic / load if load > 0 else 0.0
    return force, traction, float(trial[:, 2].max())


def run_adaptive(angle, out_dir, budget, max_cycles=13, seed_first=False, check_every=0):
    t0 = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cq.conf.device = "cuda"
    cq.conf.loading = "tension"
    cq.conf.u_max, cq.conf.n_steps = U_MAX, N_STEPS + 1
    cq.conf.Gc, cq.conf.l0 = GC, L
    cq.conf.disp_steps = np.linspace(0.0, U_MAX, N_STEPS + 1)
    torch.manual_seed(cq.conf.seed)
    np.random.seed(cq.conf.seed)

    global FIBER
    theta = np.deg2rad(float(angle))
    FIBER = theta
    mesh = cq.initial_mesh()
    if seed_first:
        mesh = preseed(mesh)
    solver = AnisoSolver(mesh)
    solver.set_material(theta)
    print("ChebyKAN Q4 adaptive sent %g: %d cells, h0 %.3f, halo %.3f, fast %s, check %d" % (
        angle, len(mesh["cells"]), mesh["h0"], 2.0 * L, FAST, check_every), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    force_path = out_dir / "force.csv"
    with force_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["step", "U", "force", "force_traction", "elements", "damage_max"])

    opt = optimizer_for(solver)
    deadline = time.monotonic() + budget
    history = torch.zeros(solver.inp.shape[0], device="cuda")
    cq.optimize(solver, history, 0.0, opt, deadline, pretrain=True)
    damage_lower = torch.zeros(solver.inp.shape[0], device="cuda")
    peak = 0.0

    def marks(current):
        nodal = current[:, 2].cpu().numpy()
        marked = aniso_mark(solver.mesh, nodal, solver.data)
        invalid, _bounds = cq.admissibility(nodal, solver.alpha_history.cpu().numpy(), solver.data)
        return marked | invalid

    for step in range(1, N_STEPS + 1):
        load = float(cq.conf.disp_steps[step])
        solver.alpha_history = damage_lower.clone()
        info = None
        trial = None
        for cycle in range(max_cycles):
            info = solve(solver, history, load, opt, deadline,
                         check=lambda current: bool(marks(current).any()), check_every=check_every)
            with torch.no_grad():
                trial = solver.fields(load).detach()
            marked = marks(trial)
            if not marked.any():
                if info.get("reason") == "early_refine":
                    raise RuntimeError("early refine without marks")
                break
            if cycle == max_cycles - 1 or len(solver.mesh["cells"]) > 50000:
                raise cq.IncompleteStep("Adaptive step %d did not resolve" % step)
            new_mesh, mapping = cq.refine(solver.mesh, marked)
            solver, history, opt, err = aniso_transfer(
                solver, new_mesh, mapping, trial, history, load, theta)
            print("REFINED step %d cycle %d cells %d interp %.2e" % (
                step, cycle, len(new_mesh["cells"]), err), flush=True)
        with torch.no_grad():
            force, traction, d_max = accept_row(solver, history, trial, load)
            history = solver.next_history(trial, history)
            damage_lower = trial[:, 2].clone()
        save_step(out_dir, step, solver, trial, history, load, t0,
                  dict(epochs=info.get("epochs"), refine_cycles=cycle))
        with force_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [step, load, force, traction, len(solver.mesh["cells"]), d_max])
        print("ACCEPTED step %d U %.5f F %.4e Ftrac %.4e cells %d d %.3f epochs %s" % (
            step, load, force, traction, len(solver.mesh["cells"]), d_max,
            info.get("epochs")), flush=True)
        peak = max(peak, abs(force))
        if d_max > 0.9 and peak > 20.0 and abs(force) < 5.0:
            print("STOP force %.4e near zero after fracture" % force, flush=True)
            break
    np.savez_compressed(out_dir / "final_mesh.npz", **solver.mesh,
                        damage=damage_lower.cpu().numpy())
    write_status(out_dir, t0, mode="adaptive", angle=float(angle), fast=FAST,
                 preseed=seed_first, check_every=check_every)
    return 0


def main():
    global MARK_SWITCH, MARK_CORE, SEED_RADIUS, FAST, ELLIPSE, SEED_ACROSS, HALO_ALONG, SEED_BACK
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refine", type=int, default=3)
    parser.add_argument("--budget", type=float, default=36000.0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mark-switch", type=float, default=MARK_SWITCH)
    parser.add_argument("--mark-core", type=float, default=MARK_CORE)
    parser.add_argument("--seed-radius", type=float, default=SEED_RADIUS)
    parser.add_argument("--fast", action="store_true", help="CUDA-graph epochs")
    parser.add_argument("--preseed", action="store_true", help="refine the tip seed before step 1")
    parser.add_argument("--check-every", type=int, default=0,
                        help="test refinement marks every N epochs (multiple of 100; needs --fast)")
    parser.add_argument("--ellipse", action="store_true",
                        help="fiber-aligned elliptic seed and core halo")
    parser.add_argument("--seed-across", type=float, default=SEED_ACROSS)
    parser.add_argument("--halo-along", type=float, default=HALO_ALONG)
    parser.add_argument("--seed-back", type=float, default=SEED_BACK)
    args = parser.parse_args()
    MARK_SWITCH, MARK_CORE, SEED_RADIUS = args.mark_switch, args.mark_core, args.seed_radius
    FAST, ELLIPSE = args.fast, args.ellipse
    SEED_ACROSS, HALO_ALONG, SEED_BACK = args.seed_across, args.halo_along, args.seed_back
    if args.check_every and (not FAST or args.check_every % 100):
        raise ValueError("--check-every needs --fast and a multiple of 100")
    if args.output is not None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Nonempty output directory: %s" % args.output)
    if args.adaptive:
        out = args.output or (ROOT / "results" / "cheby_q4_adapt" / ("sent_%g" % args.angle))
        return run_adaptive(args.angle, out, args.budget,
                            max_cycles=40 if args.check_every else 13,
                            seed_first=args.preseed, check_every=args.check_every)
    out = args.output or (ROOT / "results" / "cheby_q4_v3" / ("sent_%g" % args.angle))
    return run(args.angle, out, args.refine, args.budget)


if __name__ == "__main__":
    sys.exit(main())
