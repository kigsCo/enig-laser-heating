# Laser-pulse heating of a layered ENIG stack (Au / Ni / Cu)

## Background

ENIG (electroless nickel, immersion gold) is the surface finish on many PCB contacts: a thin gold film on a few micrometres of nickel on a copper trace. Before a nanosecond laser pulse ablates such a target in laser-induced breakdown spectroscopy (LIBS), it first heats it by conduction. How hot the surface gets, and how fast heat leaves the gold film into the nickel below, sets the fluence at which melting and ablation start.

This problem covers that first stage only: heating below the melting point. Write a function that computes the temperature field in a stack of metal layers heated by one Gaussian laser pulse.

## Geometry and conventions

- One spatial coordinate `x` (m), measured from the irradiated surface (`x = 0`) into the material. `x` increases with depth.
- The stack is a list of layers ordered from the surface down. Layer `j` occupies `[x_j, x_{j+1}]`, with `x_0 = 0` and total thickness `L = x_N`.
- Time `t` (s) starts at `t = 0`. The pulse peaks at `t = t_peak`.
- All quantities in SI units. Temperatures in kelvin.

## Governing equations

Inside each layer (constant properties, Fourier conduction):

```
rho_j * c_j * dT/dt = k_j * d2T/dx2 + S(x, t)
```

Laser source (absorbed power per unit volume, W/m^3):

```
S(x, t) = F * g(t) * p(x)

g(t) = exp(-(t - t_peak)^2 / (2 sigma^2)) / (sigma * sqrt(2 pi)),   sigma = FWHM / (2 sqrt(2 ln 2))

p(x) = exp(-x / delta) / ( delta * (1 - exp(-L / delta)) ),   0 <= x <= L
```

`F` is the absorbed fluence (J/m^2), already net of surface reflection. `g` integrates to 1 over all time and `p` integrates to 1 over the stack, so the energy deposited by time `t` is `F * [Phi((t - t_peak)/sigma) - Phi(-t_peak/sigma)]`, with `Phi` the standard normal CDF. The absorption profile uses one penetration depth `delta` for the whole stack.

Interfaces (perfect thermal contact, no boundary resistance): temperature and heat flux `k dT/dx` are continuous at every `x_j`.

Boundaries: adiabatic at both faces, `dT/dx = 0` at `x = 0` and `x = L`. Radiative and convective losses are neglected.

Initial condition: `T(x, 0) = T0` everywhere.

## Function to implement

```python
def simulate_stack_heating(layers, absorbed_fluence, pulse_fwhm, t_peak,
                           penetration_depth, t_out, probe_depths, T0=300.0):
    ...
```

| Argument | Type | Units | Meaning |
| --- | --- | --- | --- |
| `layers` | list of dict, length >= 1 | | Each dict has keys `thickness` (m), `k` (W m^-1 K^-1), `rho` (kg m^-3), `cp` (J kg^-1 K^-1), all positive and finite. Ordered from the irradiated surface down. |
| `absorbed_fluence` | float >= 0 | J m^-2 | `F` |
| `pulse_fwhm` | float > 0 | s | Full width at half maximum of `g(t)` |
| `t_peak` | float >= 3 * pulse_fwhm | s | Time of the pulse maximum. The lower bound keeps the energy cut off before `t = 0` below 1e-11 of `F`. |
| `penetration_depth` | float > 0 | m | `delta` |
| `t_out` | 1-D array-like, shape (n_t,) | s | Output times, strictly increasing, all > 0 |
| `probe_depths` | 1-D array-like, shape (n_x,) | m | Depths at which to report temperature, each in `[0, L]` |
| `T0` | float > 0 | K | Initial temperature |

Returns a `numpy.ndarray` of dtype float64 and shape `(n_t, n_x)`: element `[i, j]` is `T(probe_depths[j], t_out[i])` in kelvin.

Raises `ValueError` for any input that breaks the constraints in the table.

Dependencies: NumPy and SciPy only.

## Valid regime and approximations

- Peak temperature stays below the lowest melting point in the stack (Au 1337 K, Cu 1358 K, Ni 1728 K). No phase change or ablation.
- Pulse FWHM of about 1 ns or longer. For picosecond pulses, electrons and lattice are out of equilibrium and a two-temperature model is needed instead.
- Constant `k`, `rho` and `cp` per layer.
- Single exponential absorption. In gold, ballistic electron transport spreads the deposited energy deeper than the optical depth (about 13 nm at 1064 nm). That is ignored here.
- No thermal boundary resistance at interfaces.

## Reference stack (used in the example and some tests)

| Layer | Thickness | k (W m^-1 K^-1) | rho (kg m^-3) | cp (J kg^-1 K^-1) |
| --- | --- | --- | --- | --- |
| Au | 100 nm | 317 | 19300 | 129 |
| Ni | 5 um | 90.9 | 8900 | 444 |
| Cu | 35 um | 401 | 8960 | 385 |

Typical pulse: Nd:YAG at 1064 nm, FWHM 8 ns, `t_peak` = 24 ns, `delta` = 13 nm, absorbed fluence 500 J/m^2 (0.05 J/cm^2; with about 97% reflectance from gold, roughly 1.7 J/cm^2 incident).

## Tolerances an accepted solution must meet

Any correct method (finite difference, finite volume, finite element, spectral, any stable time integrator) should pass. The hidden tests compare against values computed independently of the reference solution, with these tolerances:

- Analytic limit comparisons: within 1% of the peak temperature rise in that test.
- Energy balance and late-time equilibrium: relative error below 0.5% and 0.1% respectively.
- Linearity in `F`: relative difference below 1e-4.
