"""Hidden tests for simulate_stack_heating.

Expected values come from sources independent of the reference solution:
- the Duhamel integral for a semi-infinite solid under a surface heat flux,
  evaluated with scipy.integrate.quad;
- the exact image-series solution for a film on a semi-infinite substrate,
  built from that same integral;
- energy conservation (deposited energy is known in closed form);
- the uniform temperature reached once an adiabatic stack equilibrates;
- linearity of the heat equation in the source.
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from stack_heating import simulate_stack_heating

AU = dict(thickness=100e-9, k=317.0, rho=19300.0, cp=129.0)
NI = dict(thickness=5e-6, k=90.9, rho=8900.0, cp=444.0)
CU = dict(thickness=35e-6, k=401.0, rho=8960.0, cp=385.0)
ENIG = [AU, NI, CU]

F_REF = 500.0          # J m^-2 absorbed
FWHM_REF = 8e-9        # s
TPEAK_REF = 24e-9      # s
DELTA_AU = 13e-9       # m, optical penetration depth of Au at 1064 nm


def _sigma(fwhm):
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def semi_infinite_rise(x, t, F, fwhm, t_peak, k, rho, cp):
    """Temperature rise in a semi-infinite solid with surface flux F*g(t).

    T(x,t) - T0 = 1/(e sqrt(pi)) * int_0^t q(s) exp(-x^2/(4 a (t-s))) / sqrt(t-s) ds,
    e = sqrt(k rho cp), a = k/(rho cp). Integrated with quad, independent of
    any mesh or time stepper.
    """
    sig = _sigma(fwhm)
    e = math.sqrt(k * rho * cp)
    a = k / (rho * cp)

    def q(s):
        return F * math.exp(-0.5 * ((s - t_peak) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))

    lo = max(0.0, t_peak - 12 * sig)
    hi = t_peak + 12 * sig
    if t <= lo:
        return 0.0
    if t <= hi:
        # (t - s)^(-1/2) singularity at the upper end handled by an algebraic weight.
        def f(s):
            return q(s) * math.exp(-x * x / (4 * a * (t - s))) if s < t else (q(s) if x == 0 else 0.0)
        val, _ = quad(f, lo, t, weight="alg", wvar=(0.0, -0.5), limit=400)
    else:
        def f(s):
            return q(s) * math.exp(-x * x / (4 * a * (t - s))) / math.sqrt(t - s)
        val, _ = quad(f, lo, hi, limit=400, points=[t_peak])
    return val / (e * math.sqrt(math.pi))


def _analytic_grid(times, depths, material, F, fwhm, t_peak):
    return np.array([[semi_infinite_rise(x, t, F, fwhm, t_peak,
                                         material["k"], material["rho"], material["cp"])
                      for x in depths] for t in times])


def _assert_close_to_peak(T, T0, expected_rise, frac=0.01):
    tol = frac * np.max(np.abs(expected_rise))
    err = np.abs((T - T0) - expected_rise)
    assert np.max(err) <= tol, f"max error {np.max(err):.4g} K exceeds {tol:.4g} K"


# 1. Standard regime, limiting case: thick homogeneous Cu behaves as a
#    semi-infinite solid; very small delta makes the source a surface flux.
def test_semi_infinite_copper_surface_and_depth():
    cu_thick = dict(CU, thickness=200e-6)
    times = np.array([16e-9, 24e-9, 30e-9, 50e-9, 200e-9])
    depths = np.array([0.0, 0.5e-6, 2e-6])
    T = simulate_stack_heating([cu_thick], F_REF, FWHM_REF, TPEAK_REF, 0.5e-9,
                               times, depths, T0=300.0)
    expected = _analytic_grid(times, depths, CU, F_REF, FWHM_REF, TPEAK_REF)
    _assert_close_to_peak(T, 300.0, expected)


# 2. Interface bookkeeping: one material split into three layers at the ENIG
#    interface depths must give the same answer as the homogeneous solid.
def test_split_identical_layers_match_homogeneous():
    layers = [dict(CU, thickness=100e-9), dict(CU, thickness=5e-6), dict(CU, thickness=194.9e-6)]
    times = np.array([20e-9, 30e-9, 100e-9, 1e-6])
    depths = np.array([0.0, 100e-9, 5.1e-6, 10e-6])
    T = simulate_stack_heating(layers, F_REF, FWHM_REF, TPEAK_REF, 0.5e-9, times, depths)
    expected = _analytic_grid(times, depths, CU, F_REF, FWHM_REF, TPEAK_REF)
    _assert_close_to_peak(T, 300.0, expected)


# 3. Numerically hard case: 1 ns pulse on gold, steep surface gradient,
#    output times spanning three decades.
def test_short_pulse_gold_stiff_case():
    au_thick = dict(AU, thickness=100e-6)
    times = np.array([2.5e-9, 3e-9, 3.5e-9, 5e-9, 1e-6])
    depths = np.array([0.0, 0.2e-6, 1e-6])
    T = simulate_stack_heating([au_thick], 200.0, 1e-9, 3e-9, 0.5e-9, times, depths)
    expected = _analytic_grid(times, depths, AU, 200.0, 1e-9, 3e-9)
    _assert_close_to_peak(T, 300.0, expected)


def _stored_energy(T, T0, depths, layers):
    """Integral of rho*cp*(T - T0) dx using the trapezoid rule within each layer."""
    edges = np.concatenate([[0.0], np.cumsum([l["thickness"] for l in layers])])
    total = 0.0
    for j, layer in enumerate(layers):
        m = (depths >= edges[j]) & (depths <= edges[j + 1])
        total += layer["rho"] * layer["cp"] * np.trapezoid(T[m] - T0, depths[m])
    return total


def film_on_substrate_rise(x_where, t, F, fwhm, t_peak, d, film, substrate, n_terms=40):
    """Exact image-series solution for a film of thickness d on a semi-infinite
    substrate, surface flux F*g(t), perfect contact.

    From the Laplace-domain solution, with Gamma = (e1 - e2) / (e1 + e2) and
    u(x, t) the semi-infinite response of the film material:
      surface:   T(0, t) = u(0, t) + 2 * sum_{n>=1} Gamma^n u(2 n d, t)
      interface: T(d, t) = (1 + Gamma) * sum_{n>=0} Gamma^n u((2n+1) d, t)
    """
    e1 = math.sqrt(film["k"] * film["rho"] * film["cp"])
    e2 = math.sqrt(substrate["k"] * substrate["rho"] * substrate["cp"])
    gam = (e1 - e2) / (e1 + e2)

    def u(x):
        return semi_infinite_rise(x, t, F, fwhm, t_peak, film["k"], film["rho"], film["cp"])

    if x_where == "surface":
        return u(0.0) + 2.0 * sum(gam ** n * u(2 * n * d) for n in range(1, n_terms))
    return (1.0 + gam) * sum(gam ** n * u((2 * n + 1) * d) for n in range(n_terms))


# 3b. Two different materials: 100 nm Au on thick Ni against the exact
#     image-series solution, at the surface and at the Au/Ni interface.
def test_gold_film_on_nickel_matches_image_series():
    d = 100e-9
    layers = [dict(AU, thickness=d), dict(NI, thickness=200e-6)]
    times = np.array([16e-9, 24e-9, 30e-9, 50e-9, 100e-9])
    T = simulate_stack_heating(layers, F_REF, FWHM_REF, TPEAK_REF, 0.5e-9, times, [0.0, d])
    expected = np.array([[film_on_substrate_rise(w, t, F_REF, FWHM_REF, TPEAK_REF, d, AU, NI)
                          for w in ("surface", "interface")] for t in times])
    _assert_close_to_peak(T, 300.0, expected)


# 4. Conservation: with adiabatic faces, stored energy equals deposited energy,
#    mid-pulse and after the pulse.
@pytest.mark.parametrize("t", [20e-9, 100e-9])
def test_energy_balance_enig(t):
    L = sum(l["thickness"] for l in ENIG)
    edges = np.cumsum([l["thickness"] for l in ENIG])
    depths = np.unique(np.concatenate([[0.0], np.geomspace(1e-11, L, 20000), edges]))
    depths = depths[depths <= L]
    T = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, [t], depths)[0]
    sig = _sigma(FWHM_REF)
    deposited = F_REF * (ndtr((t - TPEAK_REF) / sig) - ndtr(-TPEAK_REF / sig))
    stored = _stored_energy(T, 300.0, depths, ENIG)
    assert abs(stored - deposited) / deposited < 5e-3


# 5. Late-time limit: the adiabatic stack relaxes to one uniform temperature
#    set by total heat capacity.
def test_late_time_equilibrium_enig():
    depths = np.array([0.0, 50e-9, 100e-9, 2e-6, 5.1e-6, 20e-6, 40.1e-6])
    T = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, [3e-4], depths)[0]
    heat_cap = sum(l["rho"] * l["cp"] * l["thickness"] for l in ENIG)
    rise = F_REF / heat_cap
    assert np.max(np.abs((T - 300.0) - rise)) / rise < 1e-3


# 6. Linearity in the source, and the rise does not depend on T0.
def test_linear_in_fluence_and_independent_of_T0():
    times = [24e-9, 60e-9, 1e-6]
    depths = [0.0, 100e-9, 3e-6]
    T1 = simulate_stack_heating(ENIG, 250.0, FWHM_REF, TPEAK_REF, DELTA_AU, times, depths, T0=300.0)
    T2 = simulate_stack_heating(ENIG, 500.0, FWHM_REF, TPEAK_REF, DELTA_AU, times, depths, T0=77.0)
    r1, r2 = T1 - 300.0, T2 - 77.0
    assert np.max(np.abs(r2 - 2.0 * r1)) / np.max(r2) < 1e-4


# 7. Boundary behaviour: no energy in, no change; heating never cools anything.
def test_zero_fluence_stays_at_T0():
    T = simulate_stack_heating(ENIG, 0.0, FWHM_REF, TPEAK_REF, DELTA_AU,
                               [10e-9, 1e-6], [0.0, 1e-6, 40.1e-6], T0=295.0)
    assert np.allclose(T, 295.0, atol=1e-9)


def test_never_below_initial_temperature():
    times = np.geomspace(1e-9, 1e-4, 40)
    depths = np.linspace(0.0, 40.1e-6, 81)
    T = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, times, depths)
    assert T.min() >= 300.0 - 1e-6


# 8. Interface: the thin Au film sits on Ni, whose thermal effusivity is lower
#    than gold's, so the ENIG surface ends the pulse hotter than bulk gold.
def test_nickel_underlayer_raises_surface_peak():
    times = np.linspace(20e-9, 40e-9, 41)
    T_enig = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, times, [0.0])
    T_gold = simulate_stack_heating([dict(AU, thickness=40.1e-6)], F_REF, FWHM_REF,
                                    TPEAK_REF, DELTA_AU, times, [0.0])
    assert T_enig.max() > T_gold.max() + 10.0


# 9. Dense output grids (any time may coincide with an internal breakpoint of
#    a solver) must work and agree with a sparse request at shared times.
def test_dense_output_grid_consistent_with_sparse():
    dense = np.linspace(2e-9, 80e-9, 300)
    picks = [0, 75, 150, 299]
    T_dense = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, dense, [0.0, 2e-6])
    T_sparse = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, dense[picks], [0.0, 2e-6])
    assert np.all(np.isfinite(T_dense))
    rise = T_dense[picks] - 300.0
    assert np.max(np.abs(T_sparse - 300.0 - rise)) / np.max(rise) < 1e-4


# 10. Interface contract.
def test_output_shape_and_dtype():
    T = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU,
                               [10e-9, 20e-9, 30e-9], [0.0, 1e-6])
    assert isinstance(T, np.ndarray)
    assert T.shape == (3, 2)
    assert T.dtype == np.float64


@pytest.mark.parametrize("kwargs", [
    dict(layers=[]),
    dict(layers=[dict(AU, thickness=-1e-9)]),
    dict(layers=[{"thickness": 1e-6, "k": 1.0, "rho": 1.0}]),
    dict(absorbed_fluence=-1.0),
    dict(pulse_fwhm=0.0),
    dict(t_peak=1e-9),
    dict(penetration_depth=0.0),
    dict(t_out=[2e-8, 1e-8]),
    dict(t_out=[0.0, 1e-8]),
    dict(probe_depths=[-1e-9]),
    dict(probe_depths=[1.0]),
    dict(T0=0.0),
])
def test_invalid_inputs_raise(kwargs):
    args = dict(layers=ENIG, absorbed_fluence=F_REF, pulse_fwhm=FWHM_REF, t_peak=TPEAK_REF,
                penetration_depth=DELTA_AU, t_out=[1e-8], probe_depths=[0.0], T0=300.0)
    args.update(kwargs)
    with pytest.raises(ValueError):
        simulate_stack_heating(**args)
