import numpy as np
from scipy import interpolate as inter

# import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm

# import make_training_data as training
import math

h = 0.681
k_min = 10**-1.5
k_max = 10**1.5
num_bins_k = 61
num_bins_k_data = 31
N_sigma_error = 2

k_min_plot = 2e-2
k_max_plot = 50

bins_k = 10 ** (np.linspace(np.log10(k_min_plot), np.log10(k_max_plot), num_bins_k))
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
    "figure.figsize": (7.111, 3.3333),
    "figure.subplot.left": 0.066,
    "figure.subplot.right": 0.997,
    "figure.subplot.bottom": 0.102,
    "figure.subplot.top": 0.91,
    "figure.subplot.wspace": 0.185,
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


boxsizes = [50, 100, 200, 400, 1000, 2800]
snap = [11, 11, 11, 11, 122, 122]
inverted = [0, 0, 0, 0, 0, 0]
redshift = 0

# boxsizes = [400, 1000, 2800]
# snap = [11, 122, 122]
# inverted = [0, 0, 0]
# redshift = 0

colors_L = cm.plasma(np.linspace(0.0, 0.9, len(boxsizes)))

for i in range(len(inverted)):
    if inverted[i]:
        colors_L = np.insert(colors_L, i, colors_L[i - 1], axis=0)

k_ref = []
R_ref = []

# ---------------------------
fig, axs = plt.subplots(nrows=2, ncols=2, height_ratios=[3.5, 1])

ax = axs[0][0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(boxsizes)):

    if inverted[i]:
        filename = "../data_%04d/HYDRO_FIDUCIAL_INVERTED/ratio_%04d.txt" % (
            boxsizes[i],
            snap[i],
        )
    else:
        filename = "../data_%04d/HYDRO_FIDUCIAL/ratio_%04d.txt" % (boxsizes[i], snap[i])
    data = np.loadtxt(filename)
    k = data[:, 0]
    R = data[:, 1]

    ls = "-"
    label = "$L=%d~{\\rm Mpc}$" % boxsizes[i]
    if inverted[i]:
        ls = "--"
        label += " $({\\rm inverted})$"

    ax.plot(k, R, ls=ls, color=colors_L[i], lw=1, label=label)

    if boxsizes[i] == 2800:
        k_ref = k
        R_ref = R

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.81, 1.03)
# ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$R(k) = P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

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
    k_max_plot * 0.9,
    0.82,
    "$z=%3.2f$" % redshift,
    va="bottom",
    ha="right",
)

ax.text(
    k_max_plot * 0.9,
    0.82,
    "$z=%3.2f$" % redshift,
    va="bottom",
    ha="right",
)

# Extra axis
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min_plot / h, 2.0 * math.pi / k_max_plot / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])


# ---------------------------
def ratio_ref(k):
    interpolator = inter.CubicSpline(np.log10(k_ref), R_ref)
    return interpolator(np.log10(k))


ax = axs[1][0]
ax.set_xscale("log")

# Reference
# ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
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
for i in range(len(boxsizes)):

    if inverted[i]:
        filename = "../data_%04d/HYDRO_FIDUCIAL_INVERTED/ratio_%04d.txt" % (
            boxsizes[i],
            snap[i],
        )
    else:
        filename = "../data_%04d/HYDRO_FIDUCIAL/ratio_%04d.txt" % (boxsizes[i], snap[i])
    data = np.loadtxt(filename)
    k = data[:, 0]
    R = data[:, 1]

    ls = "-"
    label = "$L=%d~{\\rm Mpc}$" % boxsizes[i]
    if inverted[i]:
        ls = "--"
        label += " $({\\rm inverted})$"

    ax.plot(k, R / ratio_ref(k), ls=ls, color=colors_L[i], lw=1, label=label)


# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.968, 1.032)
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$R(k) / R_{\\rm 2800 Mpc}(k)~[-]$", labelpad=2)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])


# ---------------------------

boxsizes = [50, 100, 200, 400, 1000, 2800]
snap = [7, 7, 7, 7, 102, 102]
inverted = [0, 0, 0, 0, 0, 0]
redshift = 1

# boxsizes = [400, 1000, 2800]
# snap = [7, 102, 102]
# inverted = [0, 0, 0]
# redshift = 1


ax = axs[0][1]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(boxsizes)):

    if inverted[i]:
        filename = "../data_%04d/HYDRO_FIDUCIAL_INVERTED/ratio_%04d.txt" % (
            boxsizes[i],
            snap[i],
        )
    else:
        filename = "../data_%04d/HYDRO_FIDUCIAL/ratio_%04d.txt" % (boxsizes[i], snap[i])
    data = np.loadtxt(filename)
    k = data[:, 0]
    R = data[:, 1]
    ls = "-"
    label = "$L=%d~{\\rm Mpc}$" % boxsizes[i]
    if inverted[i]:
        ls = "--"
        label += " $({\\rm inverted})$"

    ax.plot(k, R, ls=ls, color=colors_L[i], lw=1, label=label)

    if boxsizes[i] == 2800:
        k_ref = k
        R_ref = R

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
    k_max_plot * 0.9,
    0.82,
    "$z=%3.2f$" % redshift,
    va="bottom",
    ha="right",
)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.81, 1.03)
# ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$R(k) = P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
# ax.set_yticklabels("")

# Extra axis
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(2.0 * math.pi / k_min_plot / h, 2.0 * math.pi / k_max_plot / h)
ax2.set_xlabel("${\\rm Wavelength}~\\lambda~[{\\rm{Mpc}}]$", labelpad=4)
ax2.tick_params(axis="x", which="major", pad=1)
ax2.set_xticks([1, 10.0, 100.0])
ax2.set_xticklabels(["$1$", "$10$", "$100$"])


# ---------------------------

ax = axs[1][1]
ax.set_xscale("log")

# Reference
# ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
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
for i in range(len(boxsizes)):

    if inverted[i]:
        filename = "../data_%04d/HYDRO_FIDUCIAL_INVERTED/ratio_%04d.txt" % (
            boxsizes[i],
            snap[i],
        )
    else:
        filename = "../data_%04d/HYDRO_FIDUCIAL/ratio_%04d.txt" % (boxsizes[i], snap[i])
    data = np.loadtxt(filename)
    k = data[:, 0]
    R = data[:, 1]

    ls = "-"
    label = "$L=%d~{\\rm Mpc}$" % boxsizes[i]
    if inverted[i]:
        ls = "--"
        label += " $({\\rm inverted})$"

    ax.plot(k, R / ratio_ref(k), ls=ls, color=colors_L[i], lw=1, label=label)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.968, 1.032)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$R(k) / R_{\\rm 2800 Mpc}(k)~[-]$", labelpad=2)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])

fig.savefig("convergence_two_column.png", dpi=200)
