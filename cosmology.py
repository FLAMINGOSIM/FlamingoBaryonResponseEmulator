import numpy as np
from scipy import interpolate as inter
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm
import make_training_data as training
import math

h = 0.681
k_min = 10**-1.5
k_max = 10**1.5
num_bins_k = 61
num_bins_k_data = 31
N_sigma_error = 2

k_min_plot = 2e-2
k_max_plot = 50

bins_k = 10 ** (np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k))
bins_k_data = 10 ** (np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k_data))

############################################

# Plot parameters
params = {
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "figure.figsize": (3.3333, 4.0),
    "figure.subplot.left": 0.14,
    "figure.subplot.right": 0.993,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.top": 0.915,
    "figure.subplot.wspace": 0.0,
    "figure.subplot.hspace": 0.0,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "inout",
    "ytick.direction": "inout",
}
rcParams.update(params)
rc("font", **{"family": "sans-serif", "sans-serif": ["Times"]})

############################################

from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Variation of sigma for thermal AGN
redshift = 0.0
models = np.array(
    [
        [-8.0, 0.0, 0],
        [0.0, 0.0, 0.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# Load some test data
import make_training_data as train


# ---------------------------
fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1])

ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
lines = []

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if i == 0:
        data_R=train.PS_ratio(bins_k, redshift, 0, 0, 0, False, True, "LOW_SIGMA8_STRONGEST_AGN")
    else:
        data_R=train.PS_ratio(bins_k, redshift, 0, 0, 0, False, True, "LOW_SIGMA8")    
    l1, = ax.plot(
        bins_k,
        pred_R,
        ls="-",
        color=colors_z[i],
        lw=1,
        label="${\\rm fgas}%+d\\sigma$" % models[i][0],
    )
    l2, = ax.plot(
        bins_k,
        data_R,
        ls="--",
        color=colors_z[i],
        lw=1,
    )
    lines.append(l1)
    lines.append(l2)
    
# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
#ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

# Legend and model
legend = ax.legend(
    handles=lines,
#    loc="center left",
    loc = (0.02, 0.23),
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=1,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%%$\n${\\rm LS8~cosmology}$" % redshift,
    va="top",
    ha="left",
)
ax.add_artist(legend)

l1, = ax.plot([1e5, 1e5], [1, 1], 'k-', lw=1, label="${\\rm Emu.~trained~on~fiducial~cosmology}$")
l2, = ax.plot([1e5, 1e5], [1, 1], 'k--', lw=1, label="${\\rm Sim.~using~LS8~cosmology}$")
legend = ax.legend(
    handles=[l1, l2],
    loc="lower left",
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=1,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")

# Extra axis
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min / h, 2.0 * math.pi / k_max / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])

# ---------------------------------------
ax = axs[1]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
ax.fill_between(
    bins_k,
    np.ones(np.size(bins_k[0])) * 0.99,
    np.ones(np.size(bins_k[0])) * 1.01,
    color="0.9",
)
ax.fill_between(
    bins_k,
    np.ones(np.size(bins_k[0])) * 0.995,
    np.ones(np.size(bins_k[0])) * 1.005,
    color="0.6",
)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if i == 0:
        data_R=train.PS_ratio(bins_k, redshift, 0, 0, 0, False, True, "LOW_SIGMA8_STRONGEST_AGN")
    else:
        data_R=train.PS_ratio(bins_k, redshift, 0, 0, 0, False, True, "LOW_SIGMA8")    

    ax.plot(bins_k, pred_R / data_R, ls="-", color=colors_z[i], lw=1)

        
# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.968, 1.032)
ax.set_ylabel("${\\rm Emu.} / {\\rm Sim.}$", labelpad=1)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=0)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

fig.savefig("cosmology_variations.png", dpi=200)

