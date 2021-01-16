import numpy as np
from numba.experimental import jitclass
from numba import int32, float32


@jitclass(
    [
        ("n_bins", int32),
        ("weight", float32[:]),
        ("theta", float32[:]),
    ]
)
class EscapedPhotonsHistogram:
    """Histogram object. Used to track the number of photons which have escaped
    along the various escaping angles."""

    def __init__(self, n_bins):
        """
        Create a new histogram.
        Parameters
        ----------
        n_bins: int
            The number of bins to make up the histogram.
        """

        self.n_bins = n_bins
        self.weight = np.zeros(n_bins, dtype=np.float32)
        self.theta = np.zeros(n_bins, dtype=np.float32)
        d_theta = 1 / n_bins
        half_width = 0.5 * d_theta
        for i in range(n_bins):
            self.theta[i] = np.arccos(i * d_theta + half_width)

    def bin_photon(self, cos_theta):
        """
        Bin a photon to the weight histogram.
        Parameters
        ----------
        cos_theta: float
            The escaping cos(theta) direction of the photon.
        """

        self.weight[abs(int(cos_theta * self.n_bins))] += 1

    def calculate_intensity(self, n_photons):
        """
        Calculate the mean intensity of the binned escape angles.
        Parameters
        ----------
        n_photons: int.
            The number of photons in the simulation.
        Returns
        -------
        intensity: np.ndarray
            The intensity for each binned angle.
        """

        return self.weight * self.n_bins / (2 * n_photons * np.cos(self.theta))
