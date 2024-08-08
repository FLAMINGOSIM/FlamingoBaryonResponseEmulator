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
num_bins_k = 121
num_bins_k_data = 31
N_sigma_error = 3

k_min_plot = 2e-2
k_max_plot = 50

bins_k = 10 ** (np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k))
bins_k_data = 10 ** (np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k_data))

############################################

# Plot parameters
params = {
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "figure.figsize": (4.5, 4.5),
    "figure.subplot.left": 0.105,
    "figure.subplot.right": 0.993,
    "figure.subplot.bottom": 0.085,
    "figure.subplot.top": 0.93,
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

data_mtng = np.loadtxt("../data_others/mtng.txt")
mtng_k = data_mtng[:, 0]
mtng_R = data_mtng[:, 1]
mtng_R = np.exp(mtng_R)

# Select the range overlapping with the emulator
mask = np.logical_and(mtng_k > k_min, mtng_k < k_max)
mtng_k = mtng_k[mask]
mtng_R = mtng_R[mask]


############################################

# Make some plots
redshift = 0.0
models = np.array(
    [
        [0.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls="-", color="k", lw=0.7)

# Plot the FLAMINGO emulator data
pred_R, pred_var_R = emulator.predict(
    bins_k, redshift, models[0][0], models[0][1], models[0][2]
)
ax.plot(
    bins_k,
    pred_R,
    ls="-",
    color=colors_z[0],
    lw=1,
    label="${\\rm FLAMINGO~fiducial}~{\\rm(emulator)} $",
)

# Plot MTNG data
ax.plot(mtng_k, mtng_R, '-', color='0.5', lw=0.7, label="Millennium-TNG~(Pakmor+2023)")


# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("$k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

# Legend and model
legend = ax.legend(
    loc="lower left",
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
    "$z=%3.2f$" % redshift,
    va="top",
    ha="left",
)

# Extra axis
ax2 = twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min / h, 2.0 * math.pi / k_max / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])

fig.savefig("comparisons.png", dpi=200)
