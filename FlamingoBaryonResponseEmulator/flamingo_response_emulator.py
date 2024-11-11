import numpy as np
from scipy import interpolate as inter
import swiftemulator.emulators.gaussian_process as se
import pickle
import lzma
from attr import define


# @define
class FlamingoBaryonResponseEmulator:
    """
    Emulator for the baryon response of the matter power spectrum in
    the FLAMINGO simulations.

    """

    min_k: float = -1.5
    max_k: float = 1.5
    delta_bins_k: float = None
    k_bins: np.array = None
    num_bins_k: int = 31
    PS_ratio_emulator: se.GaussianProcessEmulator = None

    def load_emulator(self):
        """
        Loads the emulator parameters from the compressed
        pickle file

        """

        with lzma.open("data/emulator.xz", "r") as f:
            self.PS_ratio_emulator = pickle.load(f)

    def predict(
        self, k: np.array, z: float, sigma_gas: float, sigma_star: float, jet: float
    ) -> np.array:
        """
        Returns the predicted baryonic response for a set of comoving modes,
        redshift, and galaxy formation model (three parameters).

        Parameters
        ----------

        k: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of [h / Mpc].

        z: float
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 2.

        sigma_gas: float

        sigma_far: float

        jet: float

        Returns
        -------

        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        Raises
        ------

        ValueError
            When the input redshift is not in the range [0, 2].

        """

        # Verify the validity of the redshift
        if z < 0.0 or z > 2.0:
            raise ValueError(
                "The emulator has only been trained for redshifts between 0 and 2."
            )

        # Construct parameters in emulator space.
        predictparams = {
            "z": z,
            "sigma_gas": sigma_gas,
            "sigma_star": sigma_star,
            "jet": jet,
        }

        # Call the emulator for the k array it was trained on
        ratio = self.PS_ratio_emulator.predict_values_no_error(
            10**self.k_bins, predictparams
        )

        # Build a spline interpolator between the points
        ratio_interpolator = inter.CubicSpline(self.k_bins, ratio)

        # Return the interpolated ratios
        baryon_ratio = ratio_interpolator(np.log10(k))

        # Set the ratio at k-values below min_k to 1
        baryon_ratio[k < 10**self.min_k] = ratio_interpolator(self.min_k)

        return baryon_ratio

    def predict_with_variance(
        self, k: np.array, z: float, sigma_gas: float, sigma_star: float, jet: float
    ):

        # Verify the validity of the redshift
        if z < 0.0 or z > 2.0:
            raise ValueError(
                "The emulator has only been trained for redshifts between 0 and 2."
            )

        # Construct parameters in emulator space.
        predictparams = {
            "z": z,
            "sigma_gas": sigma_gas,
            "sigma_star": sigma_star,
            "jet": jet,
        }

        # Call the emulator for the k array it was trained on
        ratio, variance = self.PS_ratio_emulator.predict_values(
            10**self.k_bins, predictparams
        )

        # Build a spline interpolator between the points
        ratio_interpolator = inter.CubicSpline(self.k_bins, ratio)
        variance_interpolator = inter.CubicSpline(self.k_bins, variance)

        # Return the interpolated ratios
        ret_ratio = ratio_interpolator(np.log10(k))
        ret_variance = variance_interpolator(np.log10(k))

        # Set the ratio at k-values below min_k to 1
        ret_ratio[k < 10**self.min_k] = ratio_interpolator(self.min_k)
        ret_variance[k < 10**self.min_k] = 0.0

        return ret_ratio, ret_variance

    def __init__(self):

        # Compute emulator interval
        self.delta_bins_k = (self.max_k - self.min_k) / (self.num_bins_k - 1)

        # Prepare the k_bins we used
        self.k_bins = np.linspace(self.min_k, self.max_k, self.num_bins_k)

        # Load the Gaussian process data
        self.load_emulator()
