import numpy as np
from scipy import interpolate as inter

# from numba import jit
import swiftemulator as se
import pickle
import lzma
import time


class FlamingoBaryonResponseEmulator:

    min_k = -1.5
    max_k = 1.5
    num_bins_k = 31

    # Load the emulator
    def load_emulator(self):
        with lzma.open("data/emulator.xz", "r") as f:
            self.PS_ratio_emulator = pickle.load(f)

    def predict(self, k_, z, sigma_gas, sigma_star, jet, return_variance=True):

        # Construct parameters in emulator space.
        predictparams = {
            "z": z,
            "sigma_gas": sigma_gas,
            "sigma_star": sigma_star,
            "jet": jet,
        }

        # Call the emulator for the k array it was trained on
        if return_variance:
            ratio, variance = self.PS_ratio_emulator.predict_values(
                10**self.k_bins, predictparams
            )
        else:
            ratio = self.PS_ratio_emulator.predict_values_no_error(
                10**self.k_bins, predictparams
            )

        # Build a spline emulator between the points
        ratio_interpolator = inter.CubicSpline(self.k_bins, ratio)
        if return_variance:
            variance_interpolator = inter.CubicSpline(self.k_bins, variance)

        # Return the interpolated ratios
        ret_ratio = ratio_interpolator(np.log10(k_))
        if return_variance:
            ret_variance = variance_interpolator(np.log10(k_))

        # Set the ratio at k-values below min_k to 1
        ret_ratio[k_ < 10**self.min_k] = ratio_interpolator(self.min_k)
        if return_variance:
            ret_variance[k_ < 10**self.min_k] = 0.0

        if return_variance:
            return ret_ratio, ret_variance
        else:
            return ret_ratio

    def __init__(self):
        self.load_emulator()

        # Compute emulator interval
        self.delta_bins_k = (self.max_k - self.min_k) / (self.num_bins_k - 1)

        self.k_bins = np.linspace(self.min_k, self.max_k, self.num_bins_k)
