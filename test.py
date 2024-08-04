import numpy as np
from scipy import interpolate as inter
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm

k_min = 10**-2.0
k_max = 10**1.5
num_bins_k = 71

k_min_plot = 0.006
k_max_plot = 50

bins_k = 10**(np.linspace(np.log10(k_min), np.log10(k_max), num_bins_k))

############################################

# Plot parameters
params = {'axes.labelsize': 10,
'axes.titlesize': 10,
'font.size': 9,
'legend.fontsize': 8,
'xtick.labelsize': 9,
'ytick.labelsize': 9,
'text.usetex': False,
'figure.figsize' : (5.15, 4.75),
'figure.subplot.left'    : 0.125,
'figure.subplot.right'   : 0.995,
'figure.subplot.bottom'  : 0.085,
'figure.subplot.top'     : 0.995,
'figure.subplot.wspace'  : 0.0,
'figure.subplot.hspace'  : 0.0,
'lines.markersize' : 6,
'lines.linewidth' : 1.5,
'xtick.top': True,
'ytick.right': True,
'xtick.direction' : 'inout',
'ytick.direction' : 'inout',
}
rcParams.update(params)
#rc('font',**{'family':'sans-serif','sans-serif':['Times']})

############################################

from flamingo_response_emulator import FlamingoBaryonResponseEmulator

emulator = FlamingoBaryonResponseEmulator()

############################################

# Variation of sigma for thermal AGN
models = np.array([[-10., 0., 0],   # [fgas, M*, jet 0/1]
                   [-8., 0., 0],   
                   [-6, 0., 0.],
                   [-4, 0., 0.],
                   [-3, 0., 0.],
                   [-2, 0., 0.],
                   [-1, 0., 0.],
                   [0., 0., 0.],
                   [1, 0., 0.],
                   [2., 0., 0,],
                   [3., 0., 0,],
                   [4., 0., 0.]])
colors_z = cm.plasma(np.linspace(0., 0.9, len(models)))

# ---------------------------                  
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls='-', color='k', lw=1)

# Plot the data
for i in range(len(models)):
    pred_R,_ = emulator.predict(bins_k, 0., models[i][0], models[i][1], models[i][2])
    ax.plot(bins_k, pred_R, ls='--', color=colors_z[i], lw=1, label="fgas%+dsigma"%models[i][0])        

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
ax.set_xlabel("$k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, 'k', ls='--', lw=1)
ax.vlines(k_max, -100, 100, 'k', ls='--', lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=2)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$"%0., va="top", ha="left")

fig.savefig("thermal_variations.png", dpi=200)

############################################

# Variation of sigma for thermal AGN
models = np.array([[-8., 0., 1],      # [fgas, M*, jet 0/1]
                   [-6, 0., 1.],
                   [-4, 0., 1.],
                   [-3, 0., 1.],
                   [-2, 0., 1.],
                   [-1, 0., 1.],
                   [0., 0., 1.],
                   [1, 0., 1.],
                   [2., 0., 1,],
                   [3., 0., 1,]])
colors_z = cm.plasma(np.linspace(0., 0.9, len(models)))

# ---------------------------                  
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls='-', color='k', lw=1)

# Plot the data
for i in range(len(models)):
    pred_R,_ = emulator.predict(bins_k, 0., models[i][0], models[i][1], models[i][2])
    ax.plot(bins_k, pred_R, ls='--', color=colors_z[i], lw=1, label="JET fgas%+dsigma"%models[i][0])        

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
ax.set_xlabel("$k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, 'k', ls='--', lw=1)
ax.vlines(k_max, -100, 100, 'k', ls='--', lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=2)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$"%0., va="top", ha="left")

fig.savefig("jet_variations.png", dpi=200)


############################################

# Variation of sigma for thermal AGN
models = np.array([[0., -3., 0],      # [fgas, M*, jet 0/1]
                   [0., -2., 0],
                   [0., -1, 0],
                   [0., 0, 0],
                   [0., 1, 0],
                   [0., 2, 0]])
colors_z = cm.plasma(np.linspace(0., 0.9, len(models)))

# ---------------------------                  
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls='-', color='k', lw=1)

# Plot the data
for i in range(len(models)):
    pred_R,_ = emulator.predict(bins_k, 0., models[i][0], models[i][1], models[i][2])
    ax.plot(bins_k, pred_R, ls='--', color=colors_z[i], lw=1, label="M*%+dsigma"%models[i][1])        

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
ax.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, 'k', ls='--', lw=1)
ax.vlines(k_max, -100, 100, 'k', ls='--', lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=2)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$ fgas=0sigma"%0., va="top", ha="left")

fig.savefig("Mstar_variations.png", dpi=200)

############################################

# Variation of sigma for thermal AGN
models = np.array([[0., 0, 0.],
                   [0., 0., 0.25],
                   [0., 0., 0.50],
                   [0., 0., 0.75],
                   [0., 0., 1.0],
])
colors_z = cm.plasma(np.linspace(0., 0.9, len(models)))

# ---------------------------                  
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls='-', color='k', lw=1)

# Plot the data
for i in range(len(models)):
    pred_R,_ = emulator.predict(bins_k, 0., models[i][0], models[i][1], models[i][2])
    ax.plot(bins_k, pred_R, ls='--', color=colors_z[i], lw=1, label="%d%%JET fgas+0sigma"%(models[i][2]*100))

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
ax.set_xlabel("$k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, 'k', ls='--', lw=1)
ax.vlines(k_max, -100, 100, 'k', ls='--', lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=2)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$"%0., va="top", ha="left")

fig.savefig("Jetamount_variations.png", dpi=200)


############################################

# Variation of sigma for thermal AGN
models = np.array([[-4., 0, 0.],
                   [-4., 0., 0.25],
                   [-4., 0., 0.50],
                   [-4., 0., 0.75],
                   [-4., 0., 1.0],
])
colors_z = cm.plasma(np.linspace(0., 0.9, len(models)))

# ---------------------------                  
fig, axs = plt.subplots(nrows=1, ncols=1)

ax = axs
ax.set_xscale("log")

# Reference
ax.plot(bins_k, np.ones(np.size(bins_k)), ls='-', color='k', lw=1)

# Plot the data
for i in range(len(models)):
    pred_R,_ = emulator.predict(bins_k, 0., models[i][0], models[i][1], models[i][2])
    ax.plot(bins_k, pred_R, ls='--', color=colors_z[i], lw=1, label="%d%%JET fgas-4sigma"%(models[i][2]*100))

# Plot range
ax.set_xlim(k_min_plot, k_max_plot)
ax.set_ylim(0.72, 1.15)
ax.set_xlabel("$k~[h\\cdot {\\rm Mpc}^{-1}]$", labelpad=1)
ax.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)

# Fitting range
ax.vlines(k_min, -100, 100, 'k', ls='--', lw=1)
ax.vlines(k_max, -100, 100, 'k', ls='--', lw=1)

# Legend and model
ax.legend(loc="lower left", fancybox=True, framealpha=0, handlelength=1, ncol=2)
ax.text(k_min * 1.2, 1.13, "$z=%3.2f$"%0., va="top", ha="left")

fig.savefig("Jetamount_variations_4sigma.png", dpi=200)
