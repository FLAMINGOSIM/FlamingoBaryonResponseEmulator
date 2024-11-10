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
num_bins_k_data = 61
N_sigma_error = 2

k_min_plot = 2e-2
k_max_plot = 12

bins_k = 10 ** (np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k))
bins_k_data = 10 ** (np.linspace(np.log10(k_min), np.log10(10), num_bins_k_data))
bins_k_Arico = bins_k_data[bins_k_data < 4]
bins_k_Arico = bins_k_Arico[bins_k_Arico >0.05]

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
    "figure.figsize": (7.111, 3.5333),
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

data_Preston = np.loadtxt("./data_S8/Preston_mean.csv")
Preston_k = data_Preston[:, 0]
Preston_R = data_Preston[:, 1]

data_Preston_low = np.loadtxt("./data_S8/Preston_low.csv")
Preston_low_k = data_Preston_low[:, 0]
Preston_low_R = data_Preston_low[:, 1]

data_Preston_high = np.loadtxt("./data_S8/Preston_high.csv")
Preston_high_k = data_Preston_high[:, 0]
Preston_high_R = data_Preston_high[:, 1]

def ratio_Preston_high(k):
    interpolator = inter.CubicSpline(np.log10(Preston_high_k), Preston_high_R)
    return interpolator(np.log10(k))
def ratio_Preston_low(k):
    interpolator = inter.CubicSpline(np.log10(Preston_low_k), Preston_low_R)
    return interpolator(np.log10(k))

############################################

data_Amon_mean = np.loadtxt("./data_S8/Amon_mean.csv")
Amon_mean_k = data_Amon_mean[:, 0]
Amon_mean_R = data_Amon_mean[:, 1]

data_Amon_low = np.loadtxt("./data_S8/Amon_low.csv")
Amon_low_k = data_Amon_low[:, 0]
Amon_low_R = data_Amon_low[:, 1]

data_Amon_high = np.loadtxt("./data_S8/Amon_high.csv")
Amon_high_k = data_Amon_high[:, 0]
Amon_high_R = data_Amon_high[:, 1]

def ratio_Amon_high(k):
    interpolator = inter.CubicSpline(np.log10(Amon_high_k), Amon_high_R)
    return interpolator(np.log10(k))
def ratio_Amon_low(k):
    interpolator = inter.CubicSpline(np.log10(Amon_low_k), Amon_low_R)
    return interpolator(np.log10(k))

############################################

#data_Arico_mean = np.loadtxt("./data_S8/Arico_mean.csv")
#Arico_mean_k = data_Arico_mean[:, 0]
#Arico_mean_R = data_Arico_mean[:, 1]

data_Arico_low = np.loadtxt("./data_S8/Arico_low.csv")
Arico_low_k = data_Arico_low[:, 0]
Arico_low_R = data_Arico_low[:, 1]

data_Arico_high = np.loadtxt("./data_S8/Arico_high.csv")
Arico_high_k = data_Arico_high[:, 0]
Arico_high_R = data_Arico_high[:, 1]

def ratio_Arico_high(k):
    interpolator = inter.CubicSpline(np.log10(Arico_high_k), Arico_high_R)
    return interpolator(np.log10(k))
def ratio_Arico_low(k):
    interpolator = inter.CubicSpline(np.log10(Arico_low_k), Arico_low_R)
    return interpolator(np.log10(k))

############################################

data_BigwoodkSZ_mean = np.loadtxt("./data_S8/Bigwood_kSZ_mean.csv")
BigwoodkSZ_mean_k = data_BigwoodkSZ_mean[:, 0]
BigwoodkSZ_mean_R = data_BigwoodkSZ_mean[:, 1]

data_BigwoodkSZ_low = np.loadtxt("./data_S8/Bigwood_kSZ_low.csv")
BigwoodkSZ_low_k = data_BigwoodkSZ_low[:, 0]
BigwoodkSZ_low_R = data_BigwoodkSZ_low[:, 1]

data_BigwoodkSZ_high = np.loadtxt("./data_S8/Bigwood_kSZ_high.csv")
BigwoodkSZ_high_k = data_BigwoodkSZ_high[:, 0]
BigwoodkSZ_high_R = data_BigwoodkSZ_high[:, 1]

def ratio_BigwoodkSZ_high(k):
    interpolator = inter.CubicSpline(np.log10(BigwoodkSZ_high_k), BigwoodkSZ_high_R)
    return interpolator(np.log10(k))
def ratio_BigwoodkSZ_low(k):
    interpolator = inter.CubicSpline(np.log10(BigwoodkSZ_low_k), BigwoodkSZ_low_R)
    return interpolator(np.log10(k))

############################################

data_Bigwood_mean = np.loadtxt("./data_S8/Bigwood_mean.csv")
Bigwood_mean_k = data_Bigwood_mean[:, 0]
Bigwood_mean_R = data_Bigwood_mean[:, 1]

data_Bigwood_low = np.loadtxt("./data_S8/Bigwood_low.csv")
Bigwood_low_k = data_Bigwood_low[:, 0]
Bigwood_low_R = data_Bigwood_low[:, 1]

data_Bigwood_high = np.loadtxt("./data_S8/Bigwood_high.csv")
Bigwood_high_k = data_Bigwood_high[:, 0]
Bigwood_high_R = data_Bigwood_high[:, 1]

def ratio_Bigwood_high(k):
    interpolator = inter.CubicSpline(np.log10(Bigwood_high_k), Bigwood_high_R)
    return interpolator(np.log10(k))
def ratio_Bigwood_low(k):
    interpolator = inter.CubicSpline(np.log10(Bigwood_low_k), Bigwood_low_R)
    return interpolator(np.log10(k))


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
fig, axs = plt.subplots(nrows=1, ncols=2)

# ---------------------------
ax = axs[0]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
lines = []

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.67, 1.12)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
#ax.set_yticklabels(["$1$", "$10$", "$100$"])


# Fitting range
ax.vlines(k_min, 0.73, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, 0.66, 100, "k", ls=":", lw=0.7)

ax.text(
    k_max_plot * 0.77,
    1.10,
    "$z=0$",
    va="top",
    ha="right",
)

lines = []

# Plot Arico+23 data
Arico_high_plot = ratio_Arico_high(bins_k_Arico)
Arico_low_plot = ratio_Arico_low(bins_k_Arico)
Arico_high_plot[Arico_high_plot > 1] = 1
Arico_low_plot[Arico_low_plot > 1] = 1
ax.fill_between(bins_k_Arico, Arico_low_plot, Arico_high_plot, color="c", alpha=0.3)
l, = ax.plot(bins_k_Arico, (Arico_high_plot + Arico_low_plot) / 2., "-", color="c", lw=0.7, label="${\\rm Arico+23~(DES~WL)}$")
lines.append(l)

# Plot Amon+22 data
# Amon_high_plot = ratio_Amon_high(bins_k_data)
# Amon_low_plot = ratio_Amon_low(bins_k_data)
# Amon_high_plot[Amon_high_plot > 1] = 1
# Amon_low_plot[Amon_low_plot > 1] = 1
# ax.fill_between(bins_k_data, Amon_low_plot, Amon_high_plot, color="y", alpha=0.2)
# l, = ax.plot(Amon_mean_k, Amon_mean_R, "-", color="y", lw=0.7, label="${\\rm Amon+22~(KiDS~WL+CMB)}$")
# #Amon_mean_plot = 0.5 * (Amon_high_plot + Amon_low_plot)
# #l = ax.errorbar(bins_k_data,  Amon_mean_plot, yerr=[Amon_high_plot-Amon_mean_plot, Amon_mean_plot - Amon_low_plot], color="y", alpha=0.8, ls='', lw=1)
# lines.append(l)

# Plot Preston+23 data
# Preston_high_plot = ratio_Preston_high(bins_k_data)
# Preston_low_plot = ratio_Preston_low(bins_k_data)
# Preston_high_plot[Preston_high_plot > 1] = 1
# Preston_low_plot[Preston_low_plot > 1] = 1
# ax.fill_between(bins_k_data, Preston_low_plot, Preston_high_plot, color="m", alpha=0.2)
# l, = ax.plot(Preston_k, Preston_R, "-", color="m", lw=0.7, label="${\\rm Preston+23~(DES~WL+CMB)}$")
# lines.append(l)

# # Plot Bigwood+23 WL + kSZ data
Bigwood_high_plot = ratio_Bigwood_high(bins_k_Arico)
Bigwood_low_plot = ratio_Bigwood_low(bins_k_Arico)
Bigwood_high_plot[Bigwood_high_plot > 1] = 1
Bigwood_low_plot[Bigwood_low_plot > 1] = 1
ax.fill_between(bins_k_Arico, Bigwood_low_plot, Bigwood_high_plot, color="y", alpha=0.2)
l, = ax.plot(Bigwood_mean_k, Bigwood_mean_R, "-", color="y", lw=0.7, label="${\\rm Bigwood+24~(DES~WL)}$")
lines.append(l)

# # Plot Bigwood+23 WL + kSZ data
# BigwoodkSZ_high_plot = ratio_BigwoodkSZ_high(bins_k_Arico)
# BigwoodkSZ_low_plot = ratio_BigwoodkSZ_low(bins_k_Arico)
# BigwoodkSZ_high_plot[BigwoodkSZ_high_plot > 1] = 1
# BigwoodkSZ_low_plot[BigwoodkSZ_low_plot > 1] = 1
# ax.fill_between(bins_k_Arico, BigwoodkSZ_low_plot, BigwoodkSZ_high_plot, color="g", alpha=0.2)
# l, = ax.plot(BigwoodkSZ_mean_k, BigwoodkSZ_mean_R, "-", color="g", lw=0.7, label="${\\rm Bigwood+24~(DES~WL+ACT~kSZ)}$")
# lines.append(l)

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



# ---------------------------
ax = axs[1]
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
lines = []

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.67, 1.12)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
#ax.set_yticklabels(["$1$", "$10$", "$100$"])


# Fitting range
ax.vlines(k_min, 0.76, 100, "k", ls=":", lw=0.7)
ax.vlines(k_max, 0.66, 100, "k", ls=":", lw=0.7)

ax.text(
    k_max_plot * 0.77,
    1.10,
    "$z=0$",
    va="top",
    ha="right",
)

lines = []

lines = []

# Plot Arico+23 data
# Arico_high_plot = ratio_Arico_high(bins_k_Arico)
# Arico_low_plot = ratio_Arico_low(bins_k_Arico)
# Arico_high_plot[Arico_high_plot > 1] = 1
# Arico_low_plot[Arico_low_plot > 1] = 1
# ax.fill_between(bins_k_Arico, Arico_low_plot, Arico_high_plot, color="c", alpha=0.3)
# l, = ax.plot(bins_k_Arico, (Arico_high_plot + Arico_low_plot) / 2., "-", color="c", lw=0.7, label="${\\rm Arico+23~(DES~WL)}$")
# lines.append(l)

# Plot Amon+22 data
Amon_high_plot = ratio_Amon_high(bins_k_data)
Amon_low_plot = ratio_Amon_low(bins_k_data)
Amon_high_plot[Amon_high_plot > 1] = 1
Amon_low_plot[Amon_low_plot > 1] = 1
ax.fill_between(bins_k_data, Amon_low_plot, Amon_high_plot, color="b", alpha=0.2)
l, = ax.plot(Amon_mean_k, Amon_mean_R, "-", color="b", lw=0.7, label="${\\rm Amon+22~(KiDS~WL+CMB)}$")
#Amon_mean_plot = 0.5 * (Amon_high_plot + Amon_low_plot)
#l = ax.errorbar(bins_k_data,  Amon_mean_plot, yerr=[Amon_high_plot-Amon_mean_plot, Amon_mean_plot - Amon_low_plot], color="y", alpha=0.8, ls='', lw=1)
lines.append(l)

# Plot Preston+23 data
Preston_high_plot = ratio_Preston_high(bins_k_data)
Preston_low_plot = ratio_Preston_low(bins_k_data)
Preston_high_plot[Preston_high_plot > 1] = 1
Preston_low_plot[Preston_low_plot > 1] = 1
ax.fill_between(bins_k_data, Preston_low_plot, Preston_high_plot, color="m", alpha=0.2)
l, = ax.plot(Preston_k, Preston_R, "-", color="m", lw=0.7, label="${\\rm Preston+23~(DES~WL+CMB)}$")
lines.append(l)

# # Plot Bigwood+23 WL
# Bigwood_high_plot = ratio_Bigwood_high(bins_k_Arico)
# Bigwood_low_plot = ratio_Bigwood_low(bins_k_Arico)
# Bigwood_high_plot[Bigwood_high_plot > 1] = 1
# Bigwood_low_plot[Bigwood_low_plot > 1] = 1
# ax.fill_between(bins_k_Arico, Bigwood_low_plot, Bigwood_high_plot, color="b", alpha=0.2)
# l, = ax.plot(Bigwood_mean_k, Bigwood_mean_R, "-", color="b", lw=0.7, label="${\\rm Bigwood+24~(DES~WL)}$")
# lines.append(l)

# # Plot Bigwood+23 WL + kSZ data
BigwoodkSZ_high_plot = ratio_BigwoodkSZ_high(bins_k_Arico)
BigwoodkSZ_low_plot = ratio_BigwoodkSZ_low(bins_k_Arico)
BigwoodkSZ_high_plot[BigwoodkSZ_high_plot > 1] = 1
BigwoodkSZ_low_plot[BigwoodkSZ_low_plot > 1] = 1
ax.fill_between(bins_k_Arico, BigwoodkSZ_low_plot, BigwoodkSZ_high_plot, color="g", alpha=0.4)
l, = ax.plot(BigwoodkSZ_mean_k, BigwoodkSZ_mean_R, "-", color="g", lw=0.7, label="${\\rm Bigwood+24~(DES~WL+ACT~kSZ)}$")
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


fig.savefig("comparisons_S8.png", dpi=200)
