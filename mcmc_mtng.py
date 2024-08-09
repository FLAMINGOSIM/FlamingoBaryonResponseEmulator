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


def ratio_mtng(k):
    interpolator = inter.CubicSpline(np.log10(mtng_k), mtng_R)
    return interpolator(np.log10(k))


############################################

# from scipy.optimize import minimize

# def f(model, k):
#    return np.sum((emulator.predict(k, 0, model[0], model[1], model[2]) - ratio_mtng(k))**2)

# res = minimize(f, [2, 0, 0], args=bins_k_data[15:])
# print(res.x)
# mtng_model_fit = res.x

############################################

# def f(k, model0, model1, model2):
#     ret,_ = emulator.predict(10**k, 0., model0, model1, model2)
#     return ret

# from scipy.optimize import curve_fit

# mask = np.logical_and(bins_k_data > 0.2, bins_k_data < 50)

# popt, pcov = curve_fit(f, np.log10(bins_k_data[mask]), ratio_mtng(bins_k_data[mask]), p0=[1., 0.5, 1.2], sigma=np.ones(np.size(bins_k_data[mask]))*0.01, absolute_sigma=True)
# mtng_model_fit = popt
# print(popt)

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


mask = np.logical_and(bins_k_data >= 0.099, bins_k_data < 5)
bins_k_mcmc = bins_k_data[mask]
data = (np.log10(bins_k_mcmc), ratio_mtng(bins_k_mcmc), 0.0001 * bins_k_mcmc)
nwalkers = 512
niter = 16000
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
print(theta_max)
mtng_model_fit = theta_max


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
ax.plot(mtng_k, mtng_R, "-", color="0.5", lw=0.7, label="Millennium-TNG~(Pakmor+2023)")
# ax.plot(bins_k, ratio_mtng(bins_k), '-', color='0.5', lw=0.7, label="Millennium-TNG~(Pakmor+2023)")

print(mtng_model_fit)

# Plot MTNG emulator best-fit
pred_R, pred_var_R = emulator.predict(
    bins_k, redshift, mtng_model_fit[0], mtng_model_fit[1], mtng_model_fit[2]
)
ax.plot(
    bins_k,
    pred_R,
    ls="-",
    color="g",
    lw=1,
    label="MTNG (emulator fit: $%d\\%%{\\rm JET~fgas}%+3.2f\\sigma~{\\rm M*}%+3.2f\\sigma$)"
    % (mtng_model_fit[2] * 100, mtng_model_fit[0], mtng_model_fit[1]),
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

fig.savefig("comparisons.png", dpi=200)
