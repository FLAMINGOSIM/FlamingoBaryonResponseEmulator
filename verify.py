import numpy as np
from scipy import interpolate as inter
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm
import math

h = 0.681
k_min = 10**-1.5
k_max = 10**1.5
num_bins_k = 61

k_min_plot = 0.02
k_max_plot = 50

# z_predict = [0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
# z_predict = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
z_predict = np.linspace(0.0, 2.0, 21)
model_predict = np.array(
    [
        [-8.0, 0.0, 0],  # [fgas, M*, jet 0/1]
        #                          [-6, 0., 0.],
        [-4, 0.0, 0.0],
        #                          [-3, 0., 0.],
        [-2, 0.0, 0.0],
        #                          [-1, 0., 0.],
        [0.0, 0.0, 0.0],
        #                          [1, 0., 0.],
        [2.0, 0.0, 0.0],
        [-4.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-4.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

redshift_plot = 0
sigma_plot = 0

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
    "figure.figsize": (3.333, 4.0),
    "figure.subplot.left": 0.140,
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
# rc('font',**{'family':'sans-serif','sans-serif':['Times']})


############################################

# Load some test data
import make_training_data as train

bins_k, bins_R, labels, color_m, color_z, sigmas_gas, sigmas_star, jets, redshifts = (
    train.make_training_data(
        z_predict, model_predict, k_min, k_max, num_bins_k, True, False
    )
)

num_runs = len(sigmas_gas)

print("Done loading data! (%d runs)" % num_runs)

############################################

# Emulate at the same places

from flamingo_response_emulator import FlamingoBaryonResponseEmulator
import time

start_time = time.time()
emulator = FlamingoBaryonResponseEmulator()
print("Loading took %s seconds" % (time.time() - start_time))


pred_k = []
pred_R = []

for i in range(num_runs):

    pred_x = np.copy(bins_k[i])

    start_time = time.time()
    pred_y, pred_var = emulator.predict(
        pred_x, redshifts[i], sigmas_gas[i], sigmas_star[i], jets[i]
    )
    # print("Predicting took %s seconds"% (time.time() - start_time))

    pred_k.append(pred_x)
    pred_R.append(pred_y)

print("Done making predictions")

############################################

# Verify all data points
max_error = 0
mean_error = 0
mean_error2 = 0
count_points = 0

for i in range(num_runs):
    mask = bins_k[i][:] < 10.0

    pred = pred_R[i][mask]
    truth = bins_R[i][mask]
    error = np.abs(pred - truth) / truth
    count_points += np.size(pred)

    mean_error += np.sum(error)
    mean_error2 += np.sum(error) ** 2
    if np.max(error) > max_error:
        max_error = np.max(error)

mean_error /= count_points
mean_error2 /= count_points
std_error = np.sqrt(mean_error2 - mean_error**2)

print("Max error: %e" % max_error)
print("Mean error: %e" % mean_error)
print("Std error: %e" % std_error)

############################################

# Visual inspection (at fixed sigma)

fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1])

ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(num_runs):
    if sigmas_gas[i] == 0.0 and sigmas_star[i] == 0.0 and jets[i] == 0:
        ax.plot(
            bins_k[i],
            bins_R[i],
            "o",
            color=color_z[i],
            label="$z=%.1f$" % redshifts[i],
            lw=2,
            ms=1.5,
        )
        ax.plot(pred_k[i], pred_R[i], ls="-", color=color_z[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
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
    ncol=3,
    columnspacing=0.5,
    handletextpad=0.2,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    1.13,
    "${\\rm fgas}+0\\sigma$\n${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%$",
    va="top",
    ha="left",
)

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
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.99,
    np.ones(np.size(bins_k[0])) * 1.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.995,
    np.ones(np.size(bins_k[0])) * 1.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if sigmas_gas[i] == 0.0 and sigmas_star[i] == 0.0 and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] / bins_R[i], ls="-", color=color_z[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.968, 1.032)
ax.set_ylabel("${\\rm Emu.} / {\\rm Sim.}$", labelpad=1)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=0)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

# ---------------------------------------
# ax = axs[2]
# ax.set_xscale("log")

# # Reference
# ax.plot(bins_k[0], np.zeros(np.size(bins_k[0])), ls="-", color="k", lw=1)
# ax.fill_between(
#     bins_k[0],
#     np.ones(np.size(bins_k[0])) * -0.01,
#     np.ones(np.size(bins_k[0])) * 0.01,
#     color="0.9",
# )
# ax.fill_between(
#     bins_k[0],
#     np.ones(np.size(bins_k[0])) * -0.005,
#     np.ones(np.size(bins_k[0])) * 0.005,
#     color="0.6",
# )

# # Plot the data
# for i in range(0,num_runs,2):
#     if sigmas_gas[i] == 0.0 and sigmas_star[i] == 0.0 and jets[i] == 0:
#         ax.plot(pred_k[i], pred_R[i] - bins_R[i], ls="--", color=color_z[i], lw=1)


# Plot range
# ax.set_xlim(k_min_plot, k_max_plot)
# ax.set_ylim(-0.036, 0.036)
# ax.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
# ax.set_ylabel("${\\rm Emu} - {\\rm FLAMINGO}$", labelpad=6)
# ax.set_yticks([-0.02, 0.0, 0.02], ["$-0.02$", "$0.0$", "$0.02$"])

# # Fitting range
# ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
# ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

##########################
fig.savefig("fits_fid.png", dpi=200)


############################################

# Visual inspection (at fixed z)

fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=[3, 1, 1])

ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(num_runs):
    # if redshifts[i] == 0.:# and sigmas_star[i] == 0. and jets[i] == 0:
    if redshifts[i] == 0:
        ax.plot(
            bins_k[i], bins_R[i], "o", color=color_m[i], label=labels[i], lw=2, ms=2
        )
        ax.plot(pred_k[i], pred_R[i], ls="--", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
# ax.set_xlabel("$k~[{\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=1)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$" % redshift_plot, va="top", ha="left")

# ---------------------------------------
ax = axs[1]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.99,
    np.ones(np.size(bins_k[0])) * 1.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.995,
    np.ones(np.size(bins_k[0])) * 1.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 0.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] / bins_R[i], ls="--", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.964, 1.036)
ax.set_ylabel("${\\rm Emu} / {\\rm FLAMINGO}$", labelpad=1)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# ---------------------------------------
ax = axs[2]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * -0.01,
    np.ones(np.size(bins_k[0])) * 0.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * -0.005,
    np.ones(np.size(bins_k[0])) * 0.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 0.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] - bins_R[i], ls="--", color=color_m[i], lw=1)


# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(-0.036, 0.036)
ax.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("${\\rm Emu} - {\\rm FLAMINGO}$", labelpad=6)
ax.set_yticks([-0.02, 0.0, 0.02], ["$-0.02$", "$0.0$", "$0.02$"])

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

##########################
fig.savefig("fits_z0p0.png", dpi=200)


############################################

# Visual inspection (at fixed z)

fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=[4, 1])

ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 0.5:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(
            bins_k[i], bins_R[i], "o", color=color_m[i], label=labels[i], lw=2, ms=1.5
        )
        ax.plot(pred_k[i], pred_R[i], ls="-", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# Legend and model
legend = ax.legend(
    loc="lower left",
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=1,
    columnspacing=0.5,
    handletextpad=0.2,
)
legend.get_frame().set_edgecolor("white")
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$" % 0.5, va="top", ha="left")

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
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.99,
    np.ones(np.size(bins_k[0])) * 1.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.995,
    np.ones(np.size(bins_k[0])) * 1.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 0.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] / bins_R[i], ls=":", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.968, 1.032)
ax.set_ylabel("${\\rm Emu.} / {\\rm Sim.}$", labelpad=1)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=0)


# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# ---------------------------------------
# ax = axs[2]
# ax.set_xscale("log")

# # Reference
# ax.plot(bins_k[0], np.zeros(np.size(bins_k[0])), ls="-", color="k", lw=1)
# ax.fill_between(
#     bins_k[0],
#     np.ones(np.size(bins_k[0])) * -0.01,
#     np.ones(np.size(bins_k[0])) * 0.01,
#     color="0.9",
# )
# ax.fill_between(
#     bins_k[0],
#     np.ones(np.size(bins_k[0])) * -0.005,
#     np.ones(np.size(bins_k[0])) * 0.005,
#     color="0.6",
# )

# # Plot the data
# for i in range(num_runs):
#     if redshifts[i] == 0.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
#         ax.plot(pred_k[i], pred_R[i] - bins_R[i], ls=":", color=color_m[i], lw=1)


# # Plot range
# ax.set_xlim(k_min_plot, k_max_plot)
# ax.set_ylim(-0.036, 0.036)
# ax.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
# ax.set_ylabel("${\\rm Emu} - {\\rm FLAMINGO}$", labelpad=6)
# ax.set_yticks([-0.02, 0.0, 0.02], ["$-0.02$", "$0.0$", "$0.02$"])

# # Fitting range
# ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
# ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

##########################
fig.savefig("fits_z0p5.png", dpi=200)


############################################

# Visual inspection (at fixed z)

fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=[3, 1, 1])

ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 1.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(
            bins_k[i], bins_R[i], "o", color=color_m[i], label=labels[i], lw=2, ms=2
        )
        ax.plot(pred_k[i], pred_R[i], ls="--", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
# ax.set_xlabel("$k~[{\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=1)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$" % 1, va="top", ha="left")

# ---------------------------------------
ax = axs[1]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.99,
    np.ones(np.size(bins_k[0])) * 1.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * 0.995,
    np.ones(np.size(bins_k[0])) * 1.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 1.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] / bins_R[i], ls="--", color=color_m[i], lw=1)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.964, 1.036)
ax.set_ylabel("${\\rm Emu} / {\\rm FLAMINGO}$", labelpad=1)
ax.set_yticks([0.98, 1.0, 1.02], ["$0.98$", "$1.0$", "$1.02$"])

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

# ---------------------------------------
ax = axs[2]
ax.set_xscale("log")

# Reference
ax.plot(bins_k[0], np.zeros(np.size(bins_k[0])), ls="-", color="k", lw=1)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * -0.01,
    np.ones(np.size(bins_k[0])) * 0.01,
    color="0.9",
)
ax.fill_between(
    bins_k[0],
    np.ones(np.size(bins_k[0])) * -0.005,
    np.ones(np.size(bins_k[0])) * 0.005,
    color="0.6",
)

# Plot the data
for i in range(num_runs):
    if redshifts[i] == 0.0:  # and sigmas_star[i] == 0. and jets[i] == 0:
        ax.plot(pred_k[i], pred_R[i] - bins_R[i], ls="--", color=color_m[i], lw=1)


# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(-0.036, 0.036)
ax.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("${\\rm Emu} - {\\rm FLAMINGO}$", labelpad=6)
ax.set_yticks([-0.02, 0.0, 0.02], ["$-0.02$", "$0.0$", "$0.02$"])

# Fitting range
ax.vlines(k_min, -100, 100, "k", ls=":", lw=1)
ax.vlines(k_max, -100, 100, "k", ls=":", lw=1)

##########################
fig.savefig("fits_z1p0.png", dpi=200)
