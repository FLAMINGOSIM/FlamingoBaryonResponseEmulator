import numpy as np
from scipy import interpolate as inter
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm
import make_training_data as training
import math
from astropy.cosmology import FlatLambdaCDM

h = 0.681
Omega_b = 0.0486
Omega_m = 0.306

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
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "figure.figsize": (3.333, 3.333),
    "figure.subplot.left": 0.135,
    "figure.subplot.right": 0.993,
    "figure.subplot.bottom": 0.102,
    "figure.subplot.top": 0.91,
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

# Load FLAMINGO emulator
from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Make some plots
models = np.array(
    [
        [2.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
        [0.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
        [-2.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
        [-4.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
        [-6.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
        [-8.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
    ]
)
colors_m = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the FLAMINGO emulator data
for i in range(len(models)):

    pred_R_0, pred_var_R = emulator.predict(
        bins_k, 0.0, models[i][0], models[i][1], models[i][2]
    )
    pred_R_1, pred_var_R = emulator.predict(
        bins_k, 1.0, models[i][0], models[i][1], models[i][2]
    )
    ax.plot(
        bins_k,
        pred_R_1 / pred_R_0,
        ls="-",
        color=colors_m[i],
        lw=1,
        label="${\\rm fgas}%+d\\sigma$" % models[i][0],
    )

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.87, 1.13)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel(
    "$(P(k) / P_{\\rm DMO}(k))_{z=1} / (P(k) / P_{\\rm DMO}(k))_{z=0}~[-]$", labelpad=2
)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

# Legend and model
legend = ax.legend(
    loc="lower left",
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.12,
    "${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%$",
    va="top",
    ha="left",
)

# Extra axis
ax2 = twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min_plot / h, 2.0 * math.pi / k_max_plot / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])

fig.savefig("evolution.png", dpi=200)
