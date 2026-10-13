"""Figure for the work sample: solver curves against independent references."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stack_heating import simulate_stack_heating
from test_stack_heating import (AU, NI, ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU,
                                semi_infinite_rise, film_on_substrate_rise)

t = np.linspace(2e-9, 80e-9, 300)
tm = np.concatenate([np.arange(14e-9, 38e-9, 3e-9), np.arange(44e-9, 81e-9, 9e-9)])

au_bulk = simulate_stack_heating([dict(AU, thickness=100e-6)], F_REF, FWHM_REF, TPEAK_REF,
                                 0.5e-9, t, [0.0])[:, 0] - 300
au_ni = simulate_stack_heating([dict(AU, thickness=100e-9), dict(NI, thickness=200e-6)], F_REF,
                               FWHM_REF, TPEAK_REF, 0.5e-9, t, [0.0])[:, 0] - 300
enig = simulate_stack_heating(ENIG, F_REF, FWHM_REF, TPEAK_REF, DELTA_AU, t, [0.0])[:, 0] - 300

ref_au = [semi_infinite_rise(0.0, x, F_REF, FWHM_REF, TPEAK_REF, AU["k"], AU["rho"], AU["cp"]) for x in tm]
ref_film = [film_on_substrate_rise("surface", x, F_REF, FWHM_REF, TPEAK_REF, 100e-9, AU, NI) for x in tm]

ink, grey, blue, light = "#1f2328", "#8c959f", "#1f6feb", "#d0d7de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
import sys
cover = "--cover" in sys.argv
fig, ax = plt.subplots(figsize=(8.0, 6.0) if cover else (7.0, 3.3), dpi=125 if cover else 200)
ax.plot(t * 1e9, enig, color=blue, lw=2.2, label="ENIG: 100 nm Au / 5 µm Ni / 35 µm Cu (δ = 13 nm)")
ax.plot(t * 1e9, au_ni, color=ink, lw=1.2, ls="--", label="100 nm Au on thick Ni (solver)")
ax.plot(t * 1e9, au_bulk, color=grey, lw=1.2, label="Bulk gold (solver)")
ax.plot(tm * 1e9, ref_film, "o", ms=4.5, mfc="white", mec=ink, mew=1.1,
        label="Au on Ni, exact image series")
ax.plot(tm * 1e9, ref_au, "s", ms=4, mfc="white", mec=grey, mew=1.1,
        label="Bulk gold, exact Duhamel integral")
ax.axvline(TPEAK_REF * 1e9, color=light, lw=1, zorder=0)
ax.text(TPEAK_REF * 1e9 + 0.8, 8, "pulse peak", color=grey, fontsize=8)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Surface temperature rise (K)")
ax.set_xlim(0, 80)
ax.set_ylim(0, 270)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eaeef2", lw=0.8)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title(("Nickel underlayer raises the surface peak\nfrom %.0f K to %.0f K above ambient" if cover else
              "The nickel underlayer raises the surface peak from %.0f K to %.0f K above ambient")
             % (au_bulk.max(), au_ni.max()), fontsize=13 if cover else 10, loc="left", color=ink)
fig.tight_layout()
fig.savefig("upwork_cover.png" if cover else "validation_figure.png")
print("peaks: bulk Au %.1f, Au/Ni %.1f, ENIG %.1f K" % (au_bulk.max(), au_ni.max(), enig.max()))
