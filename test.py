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
    "figure.figsize": (3.3333, 3.3333),
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

from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Variation of sigma for thermal AGN
redshift = 0.0
models = np.array(
    [
        [-10.0, 0.0, 0],  # [fgas, M*, jet 0/1]
        [-8.0, 0.0, 0],
        [-6, 0.0, 0.0],
        [-4, 0.0, 0.0],
        [-3, 0.0, 0.0],
        [-2, 0.0, 0.0],
        [-1, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)


# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm fgas}%+d\\sigma$" % models[i][0],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm fgas}%+d\\sigma$" % models[i][0],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%%$" % redshift,
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

fig.savefig("thermal_variations.png", dpi=200)

############################################

# Variation of sigma for jet AGN
models = np.array(
    [
        [-8.0, 0.0, 1],  # [fgas, M*, jet 0/1]
        [-6, 0.0, 1.0],
        [-4, 0.0, 1.0],
        [-3, 0.0, 1.0],
        [-2, 0.0, 1.0],
        [-1, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm fgas}%+d\\sigma$" % models[i][0],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm fgas}%+d\\sigma$" % models[i][0],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n${\\rm M*}+0\\sigma$\n${\\rm JET}~100\\%%$" % redshift,
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

fig.savefig("jet_variations.png", dpi=200)

############################################

# Variation of M* sigma for thermal AGN
models = np.array(
    [
        [0.0, -3.0, 0],  # [fgas, M*, jet 0/1]
        [0.0, -2.0, 0],
        [0.0, -1, 0],
        [0.0, -0.5, 0],
        [0.0, 0, 0],
        [0.0, 1, 0],
        [0.0, 2, 0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
ax.plot(bins_k, np.ones(np.size(bins_k)), ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n${\\rm fgas}+0\\sigma$\n${\\rm JET}~100\\%%$" % redshift,
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

fig.savefig("Mstar_variations_0sigma.png", dpi=200)

############################################

# Variation of M* sigma for thermal AGN
models = np.array(
    [
        [-2.0, -3.0, 0],  # [fgas, M*, jet 0/1]
        [-2.0, -2.0, 0],
        [-2.0, -1, 0],
        [-2.0, -0.5, 0],
        [-2.0, 0, 0],
        [-2.0, 1, 0],
        [-2.0, 2, 0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm fgas}-2\\sigma$" % redshift,
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

fig.savefig("Mstar_variations_2sigma.png", dpi=200)


############################################

# Variation of M* sigma for thermal AGN
models = np.array(
    [
        [-4.0, -3.0, 0],  # [fgas, M*, jet 0/1]
        [-4.0, -2.0, 0],
        [-4.0, -1, 0],
        [-4.0, -0.5, 0],
        [-4.0, 0, 0],
        [-4.0, 1, 0],
        [-4.0, 2, 0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm fgas}-4\\sigma$" % redshift,
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

fig.savefig("Mstar_variations_4sigma.png", dpi=200)

############################################

# Variation of jet fraction at fgas=0
models = np.array(
    [
        #                    [0., 0, -1.0],
        #                    [0., 0, -0.5],
        [0.0, 0, 0.0],
        [0.0, 0.0, 0.25],
        [0.0, 0.0, 0.50],
        [0.0, 0.0, 0.75],
        [0.0, 0.0, 1.0],
        #                   [0., 0., 1.5],
        #                   [0., 0., 2.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="$%d\\%%{\\rm JET~fgas}+0\\sigma$" % (models[i][2] * 100),
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="$%d\\%%{\\rm JET~fgas}+0\\sigma$" % (models[i][2] * 100),
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm M*}+0\\sigma$" % redshift,
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

fig.savefig("Jetamount_variations_0sigma.png", dpi=200)

############################################

# Variation of jet fraction at fgas=-2
models = np.array(
    [
        #                    [0., 0, -1.0],
        #                    [0., 0, -0.5],
        [-2.0, 0, 0.0],
        [-2.0, 0.0, 0.25],
        [-2.0, 0.0, 0.50],
        [-2.0, 0.0, 0.75],
        [-2.0, 0.0, 1.0],
        #                   [0., 0., 1.5],
        #                   [0., 0., 2.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="$%d\\%%{\\rm JET~fgas}-2\\sigma$" % (models[i][2] * 100),
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="$%d\\%%{\\rm JET~fgas}-2\\sigma$" % (models[i][2] * 100),
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm M*}+0\\sigma$" % redshift,
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

fig.savefig("Jetamount_variations_2sigma.png", dpi=200)

############################################

# Variation of jet fraction at fgas=4
models = np.array(
    [
        #                    [0., 0, -1.0],
        #                    [0., 0, -0.5],
        [-4.0, 0, 0.0],
        [-4.0, 0.0, 0.25],
        [-4.0, 0.0, 0.50],
        [-4.0, 0.0, 0.75],
        [-4.0, 0.0, 1.0],
        #                   [0., 0., 1.5],
        #                   [0., 0., 2.0],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm JET}~%d\\%%$" % (models[i][2] * 100),
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm JET}~%d\\%%$" % (models[i][2] * 100),
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
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
    "$z=%3.2f$~\n${\\rm fgas}-4\\sigma$\n${\\rm M*}+0\\sigma$" % redshift,
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

fig.savefig("Jetamount_variations_4sigma.png", dpi=200)


############################################

# Variation of M*sigma for JET AGN
models = np.array(
    [
        [-4, -2.0, 1],  # [fgas, M*, jet 0/1]
        [-4.0, -1.0, 1],
        [-4.0, 0.0, 1],
        [-4.0, 1.0, 1],
        [-4.0, 2, 1],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm JET~fgas}-4\\sigma$" % redshift,
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

fig.savefig("Mstar_JET_4sigma_variations.png", dpi=200)

############################################

# Variation of sigma for thermal AGN
models = np.array(
    [
        [0, -2.0, 1],  # [fgas, M*, jet 0/1]
        [0.0, -1.0, 1],
        [0.0, 0.0, 1],
        [0.0, 1.0, 1],
        [0.0, 2, 1],
    ]
)
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(models)))


# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(len(models)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, models[i][0], models[i][1], models[i][2]
    )
    if training.data_exists(models[i]):
        ax.plot(
            bins_k,
            pred_R,
            ls="-",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

        data_R = training.PS_ratio_from_model(bins_k_data, redshift, models[i])
        ax.plot(bins_k_data, data_R, "o", color=colors_z[i], lw=1, ms=2)
    else:
        ax.plot(
            bins_k,
            pred_R,
            ls="--",
            color=colors_z[i],
            lw=1,
            label="${\\rm M*}%+3.1f\\sigma$" % models[i][1],
        )

    ax.fill_between(
        bins_k,
        pred_R - N_sigma_error * np.sqrt(pred_var_R),
        pred_R + N_sigma_error * np.sqrt(pred_var_R),
        color=colors_z[i],
        alpha=0.2,
        ec="face",
        lw=0,
    )

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
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "$z=%3.2f$~\n\n${\\rm JET~fgas}+0\\sigma$" % redshift,
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

fig.savefig("Mstar_JET_0sigma_variations.png", dpi=200)
