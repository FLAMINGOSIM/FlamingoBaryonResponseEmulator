import numpy as np
import swiftemulator as se
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.pyplot import cm

k_min = 10**-1.5
k_max = 10**1.5
num_bins_k = 31

k_min_plot = 0.02
k_max_plot = 50

z_train = [0.0, 0.5, 1.0, 1.5, 2.0]
# z_train = [0., 0.5,  1.0,  1.5, 2.0]

model_train = [
    [-8.0, 0.0, 0],  # [fgas, M*, jet 0/1]
    [-4, 0.0, 0.0],
    [-2, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [-4.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [-4.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
]


#########################################
# Load some data and create a simple training set
########################################

import make_training_data as train

bins_k, bins_R, labels, color_m, color_z, sigmas_gas, sigmas_star, jets, redshifts = (
    train.make_training_data(z_train, model_train, k_min, k_max, num_bins_k, True, True)
)

print("Done loading data! (%d runs)" % len(sigmas_gas))

############################################################

# # Add some fake models
# mask = bins_k[0] < 7e-2

# my_k = np.copy(bins_k[0][mask])
# my_R = np.copy(bins_R[0][mask])

# extra_jets = [-6., -2., 2.]

# colors_z = cm.plasma(np.linspace(0., 0.9, len(z_train)))

# for i in range(len(z_train)):
#     for j in range(len(extra_jets)):

#         labels.append("JETS fgas%+dsigma 'fake'"%extra_jets[j])
#         color_m.append('k')
#         color_z.append(colors_z[i])
#         sigmas_gas.append(extra_jets[j])
#         sigmas_star.append(0)
#         jets.append(1)
#         redshifts.append(z_train[i])
#         bins_k.append(my_k)
#         bins_R.append(my_R)

# extra_jets = [-6., -4., -2., 0.,  2.]

# for i in range(len(z_train)):
#     for j in range(len(extra_jets)):

#         labels.append("50%% JETS fgas%+dsigma 'fake'"%extra_jets[j])
#         color_m.append('k')
#         color_z.append(colors_z[i])
#         sigmas_gas.append(extra_jets[j])
#         sigmas_star.append(0)
#         jets.append(0.5)
#         redshifts.append(z_train[i])
#         bins_k.append(my_k)
#         bins_R.append(my_R)


############################################################

num_runs = len(sigmas_gas)

############################################################

# Plot of fiducial model for different z
fig, axs = plt.subplots()
for i in range(num_runs):
    # print("R(-inf)=", bins_R[0][0])

    if sigmas_gas[i] == 0 and sigmas_star[i] == 0 and jets[i] == 0:
        axs.plot(
            bins_k[i],
            bins_R[i],
            "o",
            color=color_z[i],
            label="$z=%.2f$" % redshifts[i],
            ms=2,
        )

axs.plot(bins_k[0], np.ones(np.size(bins_k[0])), ls="-", color="k", lw=1)
axs.vlines(k_min, -100, 100, "k", ls="--", lw=1)
axs.vlines(k_max, -100, 100, "k", ls="--", lw=1)

axs.legend(loc="lower left")
axs.set_xscale("log")
axs.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
axs.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
axs.set_xlim(k_min_plot, k_max_plot)
axs.set_ylim(0.75, 1.15)

plt.savefig("raw_data_fid.png")

# exit()

############################################################

# Plot of z = 0 model for different sigma
fig, axs = plt.subplots()
for i in range(num_runs):
    if redshifts[i] == 0.0 and jets[i] == 1:
        axs.plot(bins_k[i], bins_R[i], "o", color=color_m[i], label=labels[i])

axs.plot(bins_k[0], np.ones(np.size(bins_k[0])), ls="-", color="k", lw=1)
axs.vlines(k_min, -100, 100, "k", ls="--", lw=1)
axs.vlines(k_max, -100, 100, "k", ls="--", lw=1)

axs.legend(loc="lower left")
axs.set_xscale("log")
axs.set_xlabel("$k~[h\\cdot{\\rm Mpc}^{-1}]$", labelpad=1)
axs.set_ylabel("$P(k) / P_{\\rm DMO}(k)~[-]$", labelpad=2)
axs.set_xlim(k_min_plot, k_max_plot)
axs.set_ylim(0.75, 1.15)

plt.savefig("raw_data_z0.png")
# exit()
###########################################

# Create emulator model

print("Setting up emulator range")

model_specification = se.ModelSpecification(
    number_of_parameters=4,
    parameter_names=["z", "sigma_gas", "sigma_star", "jet"],
    parameter_limits=[[0.0, 1.0], [-8.0, 2.0], [-1.0, 0.0], [0.0, 1.0]],
    parameter_printable_names=["Redshift", "fgas sigma", "M* sigma", "Jet model"],
)

emul_params = {}
for i in range(num_runs):
    emul_params[i] = {
        "z": redshifts[i],
        "sigma_gas": sigmas_gas[i],
        "sigma_star": sigmas_star[i],
        "jet": jets[i],
    }
model_parameters = se.ModelParameters(model_parameters=emul_params)

modelvalues = {}
for i in range(num_runs):
    independent = bins_k[i]
    dependent = bins_R[i]
    dependent_error = 0.0001 * bins_k[i]
    # dependent_error[dependent_error > 0.0001] = 0.0001
    modelvalues[i] = {
        "independent": independent,
        "dependent": dependent,
        "dependent_error": dependent_error,
    }
model_values = se.ModelValues(model_values=modelvalues)

############################################

# Train the emulator

print("Start training")

import time

start_time = time.time()

from swiftemulator.emulators import gaussian_process
from swiftemulator.mean_models.polynomial import PolynomialMeanModel
from swiftemulator.mean_models.fixed import FixedMeanModel

# `polynomial_model = PolynomialMeanModel(degree=2)
mean_model = FixedMeanModel()
mean_model.model = 1.0

PS_ratio_emulator = gaussian_process.GaussianProcessEmulator(mean_model=mean_model)
PS_ratio_emulator.fit_model(
    model_specification=model_specification,
    model_parameters=model_parameters,
    model_values=model_values,
)

print("Done training (took %s seconds)" % (time.time() - start_time))

############################################

# Serialise the emulator
import pickle
import lzma

with lzma.open("emulator.xz", "wb") as f:
    pickle.dump(PS_ratio_emulator, f)


print("Done saving everything")
