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

all_bahamas_short_names = ["bahamas_agn7p6", "bahamas_agn8p0"]
all_bahamas_labels = [
    "BAHAMAS-$\\Delta$T7.6",
    "BAHAMAS-$\\Delta$T8.0",
]
all_bahamas_fname = [
    "powtable_BAHAMAS_Theat7.6_nu0_WMAP9.dat",
    "powtable_BAHAMAS_Theat8.0_nu0_WMAP9.dat"
]
all_bahamas_colors = cm.plasma(np.linspace(0.0, 0.9, len(all_bahamas_labels)))
all_bahamas_fit = []
all_bahamas_k = []
all_bahamas_R = []

for i in range(len(all_bahamas_fname)):

    data_bahamas = np.loadtxt("./VD20/%s" % all_bahamas_fname[i])
    bahamas_z = data_bahamas[:, 0]
    bahamas_k = data_bahamas[:, 1]
    bahamas_P = data_bahamas[:, 2]

    data_DMO = np.loadtxt("./VD20/powtable_DMONLY_2fluid_nu0_WMAP9_L400N1024.dat")
    bahamas_DMO_P = data_DMO[:, 2]
    bahamas_DMO_z = data_DMO[:, 0]

    # Select z = 0 only
    mask = bahamas_DMO_z == 0.0
    bahamas_DMO_P = bahamas_DMO_P[mask]
    bahamas_DMO_z = bahamas_DMO_z[mask]

    mask = bahamas_z == 0.0
    bahamas_k = bahamas_k[mask]
    bahamas_z = bahamas_z[mask]
    bahamas_P = bahamas_P[mask]

    bahamas_R = bahamas_P / bahamas_DMO_P

    # Correct spurious large-scale excess power
    bahamas_R[bahamas_R > 1.0] = 1.0

    # Select the range overlapping with the emulator
    mask = np.logical_and(bahamas_k > k_min, bahamas_k < k_max)
    all_bahamas_k.append(bahamas_k[mask])
    all_bahamas_R.append(bahamas_R[mask])


# Create smooth function
def ratio_bahamas(k, i):
    interpolator = inter.CubicSpline(np.log10(all_bahamas_k[i]), all_bahamas_R[i])
    return interpolator(np.log10(k))


# ############################################

# Create some generic MCMC properties
import emcee
import corner
from multiprocessing import Pool


def model(theta, log10_k):
    sigma_gas, sigma_star, jet = theta
    return emulator.predict(10**log10_k, 0.0, sigma_gas, sigma_star, jet, False)


def lnlike(theta, x, y, yerr):
    return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)


def lnprior(theta):
    sigma_gas, sigma_star, jet = theta
    if (
        sigma_gas > -9.0
        and sigma_gas < 4.0
        and sigma_star > -2.0
        and sigma_star < 2.0
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


def main(p0, nwalkers, niter, ndim, lnprob, data):

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        return sampler, pos, prob, state


# Run MCMC for each model
for i in range(len(all_bahamas_k)):
    mask = np.logical_and(bins_k_data >= 0.049, bins_k_data < 20)
    bins_k_mcmc = bins_k_data[mask]
    data = (np.log10(bins_k_mcmc), ratio_bahamas(bins_k_mcmc, i), 0.0001 * bins_k_mcmc)
    nwalkers = 512
    niter = 8000
    initial = np.array([2.0, 0.5, 0.7])
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler, pos, prob, state = main(p0, nwalkers, niter, ndim, lnprob, data)

    samples = sampler.flatchain
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    best_fit_model = model(theta_max, np.log10(bins_k_data))
    all_bahamas_fit.append(theta_max)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=["fgas", "M*", "jet"])
    fig.savefig("BAHAMAS_fitting_params_%s.png" % all_bahamas_short_names[i])

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
    color="k",
    lw=1,
    label="${\\rm FLAMINGO~fid.}~{\\rm(emul.)} $",
)

# Plot other sim data
for i in range(len(all_bahamas_k)):
    # ax.plot(bahamas_k, bahamas_R, "-", color="0.5", lw=0.7, label="C-OWLS-AGN")
    ax.plot(
        bins_k,
        ratio_bahamas(bins_k, i),
        "--",
        color=all_bahamas_colors[i],
        lw=0.7,
        label=all_bahamas_labels[i],
    )

# Plot BAHAMAS emulator best-fit
for i in range(len(all_bahamas_k)):
    pred_R, pred_var_R = emulator.predict(
        bins_k, redshift, all_bahamas_fit[i][0], all_bahamas_fit[i][1], all_bahamas_fit[i][2]
    )
    ax.plot(
        bins_k,
        pred_R,
        ls="-",
        color=all_bahamas_colors[i],
        lw=1,
        label="emul. fit: $%d\\%%{\\rm JET~fgas}%+3.2f\\sigma~{\\rm M*}%+3.2f\\sigma$"
        % (all_bahamas_fit[i][2] * 100, all_bahamas_fit[i][0], all_bahamas_fit[i][1]),
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
    loc="upper left",
    fancybox=True,
    framealpha=1,
    handlelength=1,
    ncol=2,
    columnspacing=0.8,
    handletextpad=0.5,
    fontsize=7,
)
legend.get_frame().set_edgecolor("white")
ax.text(
    k_min * 1.2,
    0.73,
    "$z=%3.2f$" % redshift,
    va="bottom",
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

fig.savefig("comparisons_BAHAMAS.png", dpi=200)
