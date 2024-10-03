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

data_mtng = np.loadtxt("../data_others/mtng.txt")
mtng_k = data_mtng[:, 0]
mtng_R = data_mtng[:, 1]
mtng_R = np.exp(mtng_R)

# Select the range overlapping with the emulator
mask = mtng_k < k_max
mtng_k = mtng_k[mask]
mtng_R = mtng_R[mask]

# def ratio_mtng(k):
#     interpolator = inter.CubicSpline(np.log10(mtng_k), mtng_R)
#     return interpolator(np.log10(k))

############################################

data_eagle = np.loadtxt("./VD20/powtable_EAGLE_REF.dat")
eagle_z = data_eagle[:, 0]
eagle_k = data_eagle[:, 1]
eagle_P = data_eagle[:, 2]

data_eagle_dmo = np.loadtxt("./VD20/powtable_EAGLE_DMONLY_L100N1504.dat")
eagle_dmo_z = data_eagle_dmo[:, 0]
eagle_dmo_k = data_eagle_dmo[:, 1]
eagle_dmo_P = data_eagle_dmo[:, 2]

# Select z = 0 only
mask = eagle_z == 0.0
eagle_k = eagle_k[mask]
eagle_P = eagle_P[mask]

mask = eagle_dmo_z == 0.0
eagle_dmo_k = eagle_dmo_k[mask]
eagle_dmo_P = eagle_dmo_P[mask]

eagle_R = eagle_P / eagle_dmo_P

def ratio_eagle(k):
    interpolator = inter.CubicSpline(np.log10(eagle_k), eagle_R)
    return interpolator(np.log10(k))

############################################

data_simba = np.loadtxt("./VD20/powtable_SIMBA.dat")
simba_z = data_simba[:, 0]
simba_k = data_simba[:, 1]
simba_P = data_simba[:, 2]

data_simba_dmo = np.loadtxt("./VD20/powtable_SIMBA_DM_L100N1024.dat")
simba_dmo_z = data_simba_dmo[:, 0]
simba_dmo_k = data_simba_dmo[:, 1]
simba_dmo_P = data_simba_dmo[:, 2]

# Select z = 0 only
mask = simba_z == 0.0
simba_k = simba_k[mask]
simba_P = simba_P[mask]

mask = simba_dmo_z == 0.0
simba_dmo_k = simba_dmo_k[mask]
simba_dmo_P = simba_dmo_P[mask]

simba_R = simba_P / simba_dmo_P

def ratio_simba(k):
    interpolator = inter.CubicSpline(np.log10(simba_k), simba_R)
    return interpolator(np.log10(k))

############################################

data_illustris = np.loadtxt("./VD20/powtable_Illustris-1.dat")
illustris_z = data_illustris[:, 0]
illustris_k = data_illustris[:, 1]
illustris_P = data_illustris[:, 2]

data_illustris_dmo = np.loadtxt("./VD20/powtable_Illustris-1-DM.dat")
illustris_dmo_z = data_illustris_dmo[:, 0]
illustris_dmo_k = data_illustris_dmo[:, 1]
illustris_dmo_P = data_illustris_dmo[:, 2]

# Select z = 0 only
mask = illustris_z == 0.0
illustris_k = illustris_k[mask]
illustris_P = illustris_P[mask]

mask = illustris_dmo_z == 0.0
illustris_dmo_k = illustris_dmo_k[mask]
illustris_dmo_P = illustris_dmo_P[mask]

illustris_R = illustris_P / illustris_dmo_P

illustris_R[illustris_k < 1e-1] = 1.

def ratio_illustris(k):
    interpolator = inter.CubicSpline(np.log10(illustris_k), illustris_R)
    return interpolator(np.log10(k))

############################################

all_bahamas_short_names = ["bahamas_agn7p6", "bahamas_agn8p0"]
all_bahamas_labels = [
    "BAHAMAS-$\\Delta$T7.6",
    "BAHAMAS-$\\Delta$T8.0",
]
all_bahamas_fname = [
    "powtable_BAHAMAS_Theat7.6_nu0_WMAP9.dat",
    "powtable_BAHAMAS_Theat8.0_nu0_WMAP9.dat",
]
all_bahamas_colors = ['g', 'g']
all_bahamas_ls = ['--', '-.']
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
    mask = bahamas_k < k_max
    all_bahamas_k.append(bahamas_k[mask])
    all_bahamas_R.append(bahamas_R[mask])


# Create smooth function
def ratio_bahamas(k, i):
    interpolator = inter.CubicSpline(np.log10(all_bahamas_k[i]), all_bahamas_R[i])
    return interpolator(np.log10(k))

############################################

all_cowls_short_names = [
#    "cowl_agn",
    "cowl_agn8p3",
    "cowl_agn8p5",
    "cowl_agn8p7"
]
all_cowls_labels = [
#    "C-OWLS-AGN",
    "C-OWLS-AGN-$\\Delta$T8.3",
    "C-OWLS-AGN-$\\Delta$T8.5",
    "C-OWLS-AGN-$\\Delta$T8.7",
]
all_cowls_fname = [
#    "powtable_C-OWLS_AGN_WMAP7.dat",
    "powtable_C-OWLS_AGN_Theat8.3_WMAP7.dat",
    "powtable_C-OWLS_AGN_Theat8.5_WMAP7.dat",
    "powtable_C-OWLS_AGN_Theat8.7_WMAP7.dat",
]
all_cowls_colors = ['r', 'r', 'r']
all_cowls_ls = ['-', '--', '-.']
all_cowls_fit = []
all_cowls_k = []
all_cowls_R = []

for i in range(len(all_cowls_fname)):

    data_cowls = np.loadtxt("./VD20/%s" % all_cowls_fname[i])
    cowls_z = data_cowls[:, 0]
    cowls_k = data_cowls[:, 1]
    cowls_P = data_cowls[:, 2]

    data_DMO = np.loadtxt("./VD20/powtable_DMONLY_WMAP7_L400N1024.dat")
    cowls_DMO_P = data_DMO[:, 2]
    cowls_DMO_z = data_DMO[:, 0]

    # Select z = 0 only
    mask = cowls_DMO_z == 0.0
    cowls_DMO_P = cowls_DMO_P[mask]
    cowls_DMO_z = cowls_DMO_z[mask]

    mask = cowls_z == 0.0
    cowls_k = cowls_k[mask]
    cowls_z = cowls_z[mask]
    cowls_P = cowls_P[mask]

    cowls_R = cowls_P / cowls_DMO_P

    # Correct spurious large-scale excess power
    cowls_R[cowls_R > 1.0] = 1.0

    # Select the range overlapping with the emulator
    mask = cowls_k < k_max
    all_cowls_k.append(cowls_k[mask])
    all_cowls_R.append(cowls_R[mask])


# Create smooth function
def ratio_cowls(k, i):
    interpolator = inter.CubicSpline(np.log10(all_cowls_k[i]), all_cowls_R[i])
    return interpolator(np.log10(k))


############################################

from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Variation of sigma for thermal AGN
redshift = 0.0
models = np.array(
    [
        [0.0, 0.0, 0.0],
        [-8.0, 0.0, 0],
    ]
)
colors_z = cm.plasma(np.linspace(0.9, 0.2, len(models)))

# Load some test data
import make_training_data as train


# ---------------------------
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.hlines(1, 1e-4, 1e4, ls="-", color="k", lw=0.7)
lines = []

# Plot the data
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
        label="${\\rm FLAMINGO~fgas}%+d\\sigma, M_*+0\\sigma, {\\rm JET}~0\\%%$" % models[i][0],
    )
    lines.append(l)

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.14)
ax.set_xlabel("${\\rm Mode}~k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=0)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
#ax.vlines(k_min, -100, 100, "k", ls=":", lw=0.7)
#ax.vlines(k_max, -100, 100, "k", ls=":", lw=0.7)

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
    fontsize=7,
    columnspacing=0.8,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")
ax.add_artist(legend)
#ax.text(
#    k_min_plot * 1.2,
#    1.13,
#    "$z=%3.2f$~\n${\\rm M*}+0\\sigma$\n${\\rm JET}~0\\%%$"
#    % redshift,
#    va="top",
#    ha="left",
#)

lines = []

# Plot EAGLE data
l, = ax.plot(bins_k, ratio_eagle(bins_k), "-", color='m', lw=0.7, label="${\\rm EAGLE}$")
lines.append(l)

# Plot SIMBA data
l, = ax.plot(bins_k, ratio_simba(bins_k), "-", color='b', lw=0.7, label="${\\rm Simba}$")
lines.append(l)

# Plot SIMBA data
l, = ax.plot(bins_k, ratio_illustris(bins_k), "-", color='y', lw=0.7, label="${\\rm Illustris}$")
lines.append(l)

# Plot MTNG data
l, = ax.plot(mtng_k, mtng_R, "-", color="0.5", lw=0.7, label="${\\rm Millennium-TNG}$")
lines.append(l)

# Plot BAHAMAS data
for i in range(len(all_bahamas_k)):
    l,= ax.plot(
        bins_k,
        ratio_bahamas(bins_k, i),
        ls = all_bahamas_ls[i],
        color=all_bahamas_colors[i],
        lw=0.7,
        label=all_bahamas_labels[i],
    )
    lines.append(l)

# Plot C-OWLS data
for i in range(len(all_cowls_k)):
    l, = ax.plot(
        bins_k,
        ratio_cowls(bins_k, i),
        ls = all_cowls_ls[i],
        color=all_cowls_colors[i],
        lw=0.7,
        label=all_cowls_labels[i],
    )    
    lines.append(l)
    
legend = ax.legend(
    handles=lines,
    fontsize=7,
    loc="lower left",
    fancybox=True,
    framealpha=0,
    handlelength=1.5,
    ncol=1,
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

fig.savefig("comparison.png", dpi=200)
