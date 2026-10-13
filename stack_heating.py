"""Reference solution: laser-pulse heating of a layered metal stack.

Method
------
Vertex-centred finite volumes on a graded 1-D mesh. Every layer interface is a
node, so no segment crosses an interface and the conductance between two nodes
uses a single conductivity. The control volume of an interface node is split
between the two materials, which gives flux continuity with perfect contact.

The laser source is integrated exactly over each control volume, so the
discrete scheme deposits exactly F * integral(g) of energy. With adiabatic
boundaries the semi-discrete system conserves energy exactly; only the time
integrator's tolerance limits the energy balance.

The system C dT/dt = -K T + q(t) is stiff (cells range from sub-nanometre to
sub-micrometre), so it is integrated with the implicit BDF method, a constant
sparse Jacobian, and a step cap during the pulse so the source is resolved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

_REQUIRED_KEYS = ("thickness", "k", "rho", "cp")


def _check_inputs(layers, F, fwhm, t_peak, delta, t_out, probes, T0):
    if not isinstance(layers, (list, tuple)) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list of dicts")
    for i, layer in enumerate(layers):
        for key in _REQUIRED_KEYS:
            if key not in layer:
                raise ValueError(f"layer {i} is missing '{key}'")
            v = float(layer[key])
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"layer {i}: '{key}' must be positive and finite")
    for name, v in (("pulse_fwhm", fwhm), ("penetration_depth", delta), ("T0", T0)):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be positive and finite")
    if not (math.isfinite(F) and F >= 0):
        raise ValueError("absorbed_fluence must be finite and >= 0")
    if not (math.isfinite(t_peak) and t_peak >= 3.0 * fwhm * (1.0 - 1e-12)):
        raise ValueError("t_peak must be >= 3 * pulse_fwhm")
    if t_out.ndim != 1 or t_out.size == 0:
        raise ValueError("t_out must be a non-empty 1-D array")
    if not np.all(np.isfinite(t_out)) or t_out[0] <= 0 or np.any(np.diff(t_out) <= 0):
        raise ValueError("t_out must be strictly increasing and > 0")
    L = sum(float(layer["thickness"]) for layer in layers)
    if probes.ndim != 1 or probes.size == 0:
        raise ValueError("probe_depths must be a non-empty 1-D array")
    if not np.all(np.isfinite(probes)) or probes.min() < 0 or probes.max() > L * (1.0 + 1e-12):
        raise ValueError("probe_depths must lie in [0, L]")


def _build_mesh(edges, h0, h_max, growth=1.08):
    """Graded node positions: fine at the surface, growing geometrically.

    The step size carries over from one layer to the next. Each layer ends
    exactly on its interface; a remainder between h and 1.5 h is split in two
    so no cell is much larger than the local target size.
    """
    nodes = [0.0]
    h = h0
    for a, b in zip(edges[:-1], edges[1:]):
        x = a
        while b - x > 1.5 * h:
            x += h
            nodes.append(x)
            h = min(h * growth, h_max)
        if b - x > h:
            nodes.append(x + 0.5 * (b - x))
        nodes.append(b)
    return np.asarray(nodes)


def simulate_stack_heating(layers, absorbed_fluence, pulse_fwhm, t_peak,
                           penetration_depth, t_out, probe_depths, T0=300.0):
    F = float(absorbed_fluence)
    fwhm = float(pulse_fwhm)
    t_peak = float(t_peak)
    delta = float(penetration_depth)
    T0 = float(T0)
    t_out = np.asarray(t_out, dtype=float)
    probes = np.asarray(probe_depths, dtype=float)
    _check_inputs(layers, F, fwhm, t_peak, delta, t_out, probes, T0)

    thick = np.array([float(l["thickness"]) for l in layers])
    k = np.array([float(l["k"]) for l in layers])
    rc = np.array([float(l["rho"]) * float(l["cp"]) for l in layers])
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    L = edges[-1]

    # Mesh: resolve the absorption depth and the shortest thermal diffusion
    # length during the pulse at the surface; cap the coarsest cell at L/300.
    alpha_min = float(np.min(k / rc))
    h0 = min(delta / 5.0, math.sqrt(alpha_min * fwhm) / 10.0, L / 300.0)
    x = _build_mesh(edges, h0, L / 300.0)
    n = x.size
    dx = np.diff(x)                                   # segment lengths, n-1
    xmid = 0.5 * (x[:-1] + x[1:])
    seg_layer = np.clip(np.searchsorted(edges, xmid, side="right") - 1, 0, len(layers) - 1)

    # Heat capacity per node (J m^-2 K^-1): half of each adjacent segment.
    seg_cap = rc[seg_layer] * dx
    C = np.zeros(n)
    C[:-1] += 0.5 * seg_cap
    C[1:] += 0.5 * seg_cap

    # Conductance between neighbouring nodes (W m^-2 K^-1).
    G = k[seg_layer] / dx
    main = np.zeros(n)
    main[:-1] += G
    main[1:] += G
    K = sparse.diags([-G, main, -G], [-1, 0, 1], format="csc")
    A = sparse.diags(1.0 / C) @ K                     # dT/dt = -A T + q/C
    A = A.tocsc()

    # Fraction of the absorbed energy landing in each control volume (exact).
    faces = np.concatenate([[0.0], xmid, [L]])
    norm = -math.expm1(-L / delta)
    w = (np.exp(-faces[:-1] / delta) - np.exp(-faces[1:] / delta)) / norm
    src = F * w / C                                   # K per unit of g(t)

    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gnorm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    def g(t):
        return gnorm * math.exp(-0.5 * ((t - t_peak) / sigma) ** 2)

    def rhs(t, T):
        return -(A @ T) + src * g(t)

    jac = -A
    # Solve for the rise above T0; the problem is linear so this is exact.
    y0 = np.zeros(n)
    t_end = float(t_out[-1])
    t_pulse_end = t_peak + 4.0 * fwhm
    rtol, atol = 1e-8, 1e-8

    results = []
    if t_end <= t_pulse_end:
        sol = solve_ivp(rhs, (0.0, t_end), y0, method="BDF", jac=jac, t_eval=t_out,
                        max_step=fwhm / 20.0, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        results.append(sol.y)
    else:
        # Strict split: a requested time equal to t_pulse_end belongs to the
        # second segment, otherwise t_eval would hold it twice.
        early = t_out[t_out < t_pulse_end]
        late = t_out[t_out >= t_pulse_end]
        sol1 = solve_ivp(rhs, (0.0, t_pulse_end), y0, method="BDF", jac=jac,
                         t_eval=np.concatenate([early, [t_pulse_end]]),
                         max_step=fwhm / 20.0, rtol=rtol, atol=atol)
        if not sol1.success:
            raise RuntimeError(sol1.message)
        results.append(sol1.y[:, :-1])
        sol2 = solve_ivp(rhs, (t_pulse_end, t_end), sol1.y[:, -1], method="BDF",
                         jac=jac, t_eval=late, rtol=rtol, atol=atol)
        if not sol2.success:
            raise RuntimeError(sol2.message)
        results.append(sol2.y)

    rise = np.concatenate(results, axis=1).T          # (n_t, n_nodes)
    out = np.empty((t_out.size, probes.size))
    for i in range(t_out.size):
        out[i] = T0 + np.interp(np.minimum(probes, L), x, rise[i])
    return out
