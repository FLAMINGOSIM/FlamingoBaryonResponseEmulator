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

# Load FLAMINGO emulator
from flamingo_response_emulator import FlamingoBaryonResponseEmulator
emulator = FlamingoBaryonResponseEmulator()

############################################

data_cowls = np.loadtxt("./VD20/powtable_C-OWLS_AGN_WMAP7.dat")
cowls_z = data_cowls[:,0]
cowls_k = data_cowls[:,1]
cowls_P = data_cowls[:,2]
data_DMO = np.loadtxt("./VD20/powtable_DMONLY_WMAP7_L400N1024.dat")
cowls_DMO_P = data_DMO[:,2]
cowls_R = cowls_P / cowls_DMO_P

mask = (cowls_z == 0.)
cowls_k = cowls_k[mask]
cowls_R = cowls_R[mask]
cowls_z = cowls_z[mask]

# Correct spurious large-scale excess power
cowls_R[cowls_R > 1.] = 1.

# Select the range overlapping with the emulator
mask = np.logical_and(cowls_k > k_min, cowls_k < k_max)
cowls_k = cowls_k[mask]
cowls_R = cowls_R[mask]

# Create smooth function
def ratio_cowls(k):
    interpolator = inter.CubicSpline(np.log10(cowls_k), cowls_R)
    return interpolator(np.log10(k))

############################################

import emcee
from multiprocessing import Pool

def model(theta, log10_k):
    sigma_gas, sigma_star, jet = theta
    return emulator.predict(10**log10_k, 0.0, sigma_gas, sigma_star, jet, False)

def lnlike(theta, x, y, yerr):
    return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)

def lnprior(theta):
    sigma_gas, sigma_star, jet = theta
    if (
        sigma_gas > -8.0
        and sigma_gas < 4.0
        and sigma_star > -2.0
        and sigma_star < 1.0
        and jet > -0.5
        and jet < 1.5
    ):
        return 0.0
    else:
        return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


mask = np.logical_and(bins_k_data >= 0.099, bins_k_data < 20)
bins_k_mcmc = bins_k_data[mask]
data = (np.log10(bins_k_mcmc), ratio_cowls(bins_k_mcmc), 0.0001 * bins_k_mcmc)
nwalkers = 256
niter = 4000
initial = np.array([2.0, 0.5, 0.7])
ndim = len(initial)
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]


def main(p0, nwalkers, niter, ndim, lnprob, data):

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        return sampler, pos, prob, state


sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

samples = sampler.flatchain
theta_max = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max, np.log10(bins_k_data))
cowls_model_fit = theta_max

import corner
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(flat_samples, labels=["fgas", "M*", "jet"])
fig.savefig("COWLS_fitting_params.png")

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

# Plot other sim data
#ax.plot(cowls_k, cowls_R, "-", color="0.5", lw=0.7, label="C-OWLS-AGN")
ax.plot(bins_k, ratio_cowls(bins_k), '-', color='0.5', lw=0.7, label="C-OWLS-AGN")

print(cowls_model_fit)

# Plot COWLS emulator best-fit
pred_R, pred_var_R = emulator.predict(
    bins_k, redshift, cowls_model_fit[0], cowls_model_fit[1], cowls_model_fit[2]
)
ax.plot(
    bins_k,
    pred_R,
    ls="-",
    color="g",
    lw=1,
    label="emulator fit: $%d\\%%{\\rm JET~fgas}%+3.2f\\sigma~{\\rm M*}%+3.2f\\sigma$"
    % (cowls_model_fit[2] * 100, cowls_model_fit[0], cowls_model_fit[1]),
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

fig.savefig("comparisons_COWL.png", dpi=200)
