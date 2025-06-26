import numpy as np
from scipy import interpolate as inter
import swiftemulator.emulators.gaussian_process as se
import pickle
import lzma
import os
import inspect
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

        data_path = os.path.join(
            os.path.dirname(os.path.abspath(inspect.stack()[0][1])), "data"
        )
        filename = os.path.join(data_path, "emulator.xz")

        with lzma.open(filename, "r") as f:
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
            The value has to be between 0 and 3.

        sigma_gas: float
            The offset in numbers of sigma (of the X-ray data) the gas fraction
            in groups and clusters should be from the data used in the calibration
            of the FLAMINGO model. The emulator was trained between -8 and +2
            for a jet fraction of 0% and between -4 and 0 for a jet fraction of 100%.

        sigma_star: float
            The offset in numbers of sigma (of the data) the stellar mass function
            should be from the data used in the calibration of the FLAMINGO model.
            The emulator was trained between -1 and 0.

        jet: float
            The fraction of the AGN energy released in the form of collimated jets
            (between 0 and 1). The original simulations exist only as purely thermal
            AGN (i.e. with jet = 0) and with purely collimated jets (i.e. with jet = 1).

        Returns
        -------

        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        Raises
        ------

        ValueError
            When the input redshift is not in the range [0, 3].

        """

        # Verify the validity of the redshift
        if z < 0.0 or z > 3.0:
            raise ValueError(
                "The emulator has only been trained for redshifts between 0 and 3."
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
    ) -> tuple[np.array, np.array]:
        """
        Returns the predicted baryonic response as well as the variance around the
        prediction for a set of comoving modes, redshift, and galaxy formation
        model (three parameters).

        Parameters
        ----------

        k: np.array
            The Fourier modes at which the baryonic response has to be evaluated
            expressed in units of [h / Mpc].

        z: float
            The redshift at which the baryonic response has to be evaluated.
            The value has to be between 0 and 3.

        sigma_gas: float
            The offset in numbers of sigma (of the X-ray data) the gas fraction
            in groups and clusters should be from the data used in the calibration
            of the FLAMINGO model. The emulator was trained between -8 and +2
            for a jet fraction of 0% and between -4 and 0 for a jet fraction of 100%.

        sigma_star: float
            The offset in numbers of sigma (of the data) the stellar mass function
            should be from the data used in the calibration of the FLAMINGO model.
            The emulator was trained between -1 and 0.

        jet: float
            The fraction of the AGN energy released in the form of collimated jets
            (between 0 and 1). The original simulations exist only as purely thermal
            AGN (i.e. with jet = 0) and with purely collimated jets (i.e. with jet = 1).

        Returns
        -------

        baryon_ratio: np.array
            The baryonic response at the modes k specified in the input.

        baryon_ratio_variance: np.array
            The estimated variance of the baryonic response from the emulator
            at the modes k specified in the input.

        Raises
        ------

        ValueError
            When the input redshift is not in the range [0, 3].

        """

        # Verify the validity of the redshift
        if z < 0.0 or z > 3.0:
            raise ValueError(
                "The emulator has only been trained for redshifts between 0 and 3."
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
        baryon_ratio = ratio_interpolator(np.log10(k))
        baryon_ratio_variance = variance_interpolator(np.log10(k))

        # Set the ratio at k-values below min_k to 1
        baryon_ratio[k < 10**self.min_k] = ratio_interpolator(self.min_k)
        baryon_ratio_variance[k < 10**self.min_k] = 0.0

        return baryon_ratio, baryon_ratio_variance

    def __init__(self):

        # Compute emulator interval
        self.delta_bins_k = (self.max_k - self.min_k) / (self.num_bins_k - 1)

        # Prepare the k_bins we used
        self.k_bins = np.linspace(self.min_k, self.max_k, self.num_bins_k)

        # Load the Gaussian process data
        self.load_emulator()
