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
    "figure.figsize": (3.333, 4.00),
    "figure.subplot.left": 0.095,
    "figure.subplot.right": 0.993,
    "figure.subplot.bottom": 0.092,
    "figure.subplot.top": 0.905,
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

# Load FLAMINGO Mhalo - fb bins
data = np.loadtxt("fit_fgas_models.txt")
data_z = data[:, 0]
data_sigma = data[:, 1]
data_jet = data[:, 2]
data_M500 = data[:, 3]
data_fb_mean = data[:, 4] / (Omega_b / Omega_m)
data_fb_median = data[:, 5] / (Omega_b / Omega_m)

bins = [13, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0]
colors_bins = cm.plasma(np.linspace(0.0, 0.9, len(bins)))
sm = cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=13, vmax=15))

fig, ax = plt.subplots(nrows=1, ncols=1)
for i in range(len(bins)):

    mask = np.logical_and(
        data_M500 > 10 ** (bins[i] - 0.01), data_M500 < 10 ** (bins[i] + 0.01)
    )

    mask_jet = np.logical_and(mask, data_jet == 1)
    mask_fid = np.logical_and(mask, data_jet == 0)

    # print(np.sum(mask_fid), np.sum(mask_jet))

    ax.plot(
        data_fb_mean[mask_fid],
        data_sigma[mask_fid],
        "o-",
        lw=0.9,
        ms=2,
        color=colors_bins[i],
        label="$10^{%4.2f}$" % bins[i],
    )
    ax.plot(
        data_fb_mean[mask_jet], data_sigma[mask_jet], "o--", lw=0.9, ms=2, color=colors_bins[i]
    )

ax.text(-0.01, 2.95,
        "$M_{\\rm 500c} / {\\rm M}_\\odot=$",
        fontsize=7.5, rasterized=False)
    
legend = ax.legend(
    loc="lower left",
    fancybox=True,
    framealpha=1,
    handlelength=0.8,
    ncol=5,
    columnspacing=0.8,
    handletextpad=0.5,
    bbox_to_anchor=(0.1, 0.99),
)
legend.get_frame().set_edgecolor("white")
legend.get_title().set_fontsize("7")
ax.add_artist(legend)

(l1,) = ax.plot(
    [1e5, 1e5], [1, 1], "k-", lw=0.9, label="${\\rm Thermal~feedback}$"
)
(l2,) = ax.plot(
    [1e5, 1e5], [1, 1], "k--", lw=0.9, label="${\\rm Jet~feedback}$"
)
legend = ax.legend(
    handles=[l1, l2],
    loc="lower center",
    fancybox=True,
    framealpha=1,
    handlelength=1.15,
    ncol=2,
    columnspacing=2.0,
    handletextpad=0.5,
)
legend.get_frame().set_edgecolor("white")


ax.grid(lw=0.3, color="0.9")
ax.set_xlim(0.06, 0.82)
ax.set_ylim(-9.1, 2.3)
ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_xlabel(
    "$f_{\\rm gas}(<R_{500{\\rm c}}) / (\\Omega_{\\rm b} / \\Omega_{\\rm m})~[-]$",
    labelpad=1,
)
ax.set_ylabel("${\\rm fgas}~N\\sigma$", labelpad=-4)
# plt.colorbar(sm, ticks=[-1, 0, 1], )
ax.yaxis.set_label_coords(-0.06, 0.53)

fig.savefig("gas_fractions.png", dpi=200)
