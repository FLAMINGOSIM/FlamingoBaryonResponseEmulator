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
    "figure.subplot.left": 0.135,
    "figure.subplot.right": 0.993,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.top": 0.92,
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

data_colibre_thermal = np.loadtxt("../data_colibre/ratio_0122.txt")
colibre_thermal_k = data_colibre_thermal[:, 0]
colibre_thermal_R = data_colibre_thermal[:, 1]

# Select the range overlapping with the emulator
mask = colibre_thermal_k < k_max
colibre_thermal_k = colibre_thermal_k[mask]
colibre_thermal_R = colibre_thermal_R[mask]

############################################

from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Variation of sigma for thermal AGN
redshift = 0.0
models = np.array(
    [
        [3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-8.0, 0.0, 0],
        [-8.0, 0.0, 1.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.9, 0.15, len(models)))

# Load some test data
import make_training_data as train


# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
lines = []

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.62, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
#ax.set_yticklabels(["$1$", "$10$", "$100$"])


# Fitting range
ax.vlines(k_min, 0.67, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, 0.66, 100, "k", ls=":", lw=0.7)

ax.text(
    k_max_plot * 0.7,
    0.63,
    "$z=0$",
    va="bottom",
    ha="right",
)

lines = []

# Plot FABLE data
l, = ax.plot(colibre_thermal_k, colibre_thermal_R, "-", color="g", lw=1.0, label="${\\rm COLIBRE\\_m7~thermal}$")
lines.append(l)
    
legend = ax.legend(
    handles=lines,
    fontsize=7.5,
    loc="lower left",
    fancybox=True,
    framealpha=0,
    handlelength=1.5,
    ncol=1,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.add_artist(legend)

# Plot the data
lines = []
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    l, = ax.plot(
        bins_k,
        pred_R,
        'o-',
        ms=2,
        #ls="-",
        color=colors_z[i],
        lw=1,
        label="${\\rm FLAMINGO:~fgas}%+d\\sigma, M_*+%d\\sigma, {\\rm JET}~%d\\%%$" % (models[i][0], models[i][1], models[i][2]*100),
    )
    lines.append(l)
# Legend and model
legend = ax.legend(
    handles=lines,
    loc="upper left",
#    loc="lower left",
#   loc = (0.02, 0.71),
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=1,
    fontsize=7.5,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")


# Extra axis
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min_plot / h, 2.0 * math.pi / k_max / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])

fig.savefig("comparison_colibre.png", dpi=200)
