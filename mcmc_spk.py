import numpy as np
from scipy import interpolate as inter
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm
import make_training_data as training
import math
import pyspk.model as spk
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

redshifts = [0.0, 0.5, 1.0, 1.5, 2.0]
colors_z = cm.plasma(np.linspace(0.0, 0.9, len(redshifts)))

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

# Load FLAMINGO Mhalo - fb bins
data = np.loadtxt("SPK/fit_fiducial.txt")
data_z = data[:, 0]
data_M500 = data[:, 1]
data_fb = data[:, 2] / (Omega_b / Omega_m)

fig, ax = plt.subplots(nrows=1, ncols=1)
for i in range(len(redshifts)):

    z = redshifts[i]
    mask = np.logical_and(data_z == z, data_M500 > 10**12.5)
    ax.plot(
        data_M500[mask],
        data_fb[mask],
        "-",
        lw=0.9,
        color=colors_z[i],
        label="$z=%2.1f$" % z,
    )

    ax.set_xscale("log")
    ax.set_xlabel("$M_{500, {\\rm cr}}~[{\\rm M}_\\odot]$", labelpad=0)
    ax.set_ylabel(
        "$f_{\\rm b}(<R_{500, {\\rm cr}}) / (\\Omega_{\\rm b} / \\Omega_{\\rm m})~[-]$",
        labelpad=2,
    )

    legend = ax.legend(
        loc="lower right",
        fancybox=True,
        framealpha=1,
        handlelength=1,
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    legend.get_frame().set_edgecolor("white")


fig.savefig("FLAMINGO_baryon_fraction.png", dpi=200)

# ############################################

# Build SP(k) repsonse
SPK_k = []
SPK_R = []
for i in range(len(redshifts)):

    z = redshifts[i]
    mask = np.logical_and(data_z == z, np.logical_not(np.isnan(data_fb)))

    my_k, my_R = spk.sup_model(SO=500, z=z, M_halo=data_M500[mask], fb=data_fb[mask])
    SPK_k.append(my_k)
    SPK_R.append(my_R)

# # Create smooth function
# def ratio_bahamas(k, i):
#     interpolator = inter.CubicSpline(np.log10(all_bahamas_k[i]), all_bahamas_R[i])
#     return interpolator(np.log10(k))

# ############################################

# Create some generic MCMC properties
# import emcee
# import corner
# from multiprocessing import Pool


# def model(theta, log10_k):
#     sigma_gas, sigma_star, jet = theta
#     return emulator.predict(10**log10_k, 0.0, sigma_gas, sigma_star, jet, False)


# def lnlike(theta, x, y, yerr):
#     return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)


# def lnprior(theta):
#     sigma_gas, sigma_star, jet = theta
#     if (
#         sigma_gas > -9.0
#         and sigma_gas < 4.0
#         and sigma_star > -2.0
#         and sigma_star < 2.0
#         and jet > -0.5
#         and jet < 1.5
#     ):
#         return 0.0
#     else:
#         return -np.inf


# def lnprob(theta, x, y, yerr):
#     lp = lnprior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + lnlike(theta, x, y, yerr)


# def main(p0, nwalkers, niter, ndim, lnprob, data):

#     with Pool() as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
#         print("Running burn-in...")
#         p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
#         sampler.reset()
#         print("Running production...")
#         pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
#         return sampler, pos, prob, state


# # Run MCMC for each model
# for i in range(len(all_bahamas_k)):
#     mask = np.logical_and(bins_k_data >= 0.049, bins_k_data < 20)
#     bins_k_mcmc = bins_k_data[mask]
#     data = (np.log10(bins_k_mcmc), ratio_bahamas(bins_k_mcmc, i), 0.0001 * bins_k_mcmc)
#     nwalkers = 512
#     niter = 8000
#     initial = np.array([2.0, 0.5, 0.7])
#     ndim = len(initial)
#     p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

#     sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

#     samples = sampler.flatchain
#     theta_max = samples[np.argmax(sampler.flatlnprobability)]
#     best_fit_model = model(theta_max, np.log10(bins_k_data))
#     all_bahamas_fit.append(theta_max)

#     flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
#     fig = corner.corner(flat_samples, labels=["fgas", "M*", "jet"])
#     fig.savefig("BAHAMAS_fitting_params_%s.png" % all_bahamas_short_names[i])

############################################

# Make some plots
models = np.array(
    [
        [0.0, 0.0, 0.0],  # [fgas, M*, jet 0/1]
    ]
)

# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)

# Plot the FLAMINGO emulator data
for i in range(len(redshifts)):

    pred_R, pred_var_R = emulator.predict(
        bins_k, redshifts[i], models[0][0], models[0][1], models[0][2]
    )
    ax.plot(
        bins_k,
        pred_R,
        ls="-",
        color=colors_z[i],
        lw=1,
        label="$z=%2.1f$" % redshifts[i],
    )

# Plot SP(k)
for i in range(len(redshifts)):
    ax.plot(SPK_k[i], SPK_R[i], "--", color=colors_z[i], lw=0.9)


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
    "${\\rm fgas}+0\\sigma$~\n${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%$",
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
ax2.plot([0.1, 0.1], [10, 10], "k-", lw=0.9, label="${\\rm FLAMINGO~emul.}$")
ax2.plot([0.1, 0.1], [10, 10], "k--", lw=0.9, label="${\\rm SP}(k)$")
legend = ax2.legend(
    loc="lower right",
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=1,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")


fig.savefig("comparisons_SPK.png", dpi=200)
