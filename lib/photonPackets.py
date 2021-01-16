import numpy as np
from numba.experimental import jitclass
from numba import float32, boolean


@jitclass(
    [
        ("coords", float32[:]),
        ("costheta", float32),
        ("sintheta", float32),
        ("cosphi", float32),
        ("sinphi", float32),
        ("escaped", boolean)
    ]
)
class PhotonPacket:
    """Photon packet object. Contains all the functions to create and scatter
    photons."""

    def __init__(self):
        """
        Creates a new photon with a random isotropic direction.
        """

        self.coords = np.zeros(3, np.float32)
        self.costheta = np.sqrt(np.random.rand())
        self.sintheta = np.sqrt(1 - self.costheta ** 2)
        self.cosphi = np.cos(2 * np.pi * np.random.rand())
        self.sinphi = np.sqrt(1 - self.cosphi ** 2)
        self.escaped = False

    def move_photon_length_ds(self, ds):
        """
        Move a photon a length ds in its current direction.

        Parameters
        ----------
        ds: float
            The length to move the photon.
        """

        self.coords[0] += ds * self.sintheta * self.cosphi
        self.coords[1] += ds * self.sintheta * self.sinphi
        self.coords[2] += ds * self.costheta

    def isotropic_scatter(self):
        """
        Scatter a photon isotropically.
        """

        self.costheta = 2 * np.random.rand() - 1
        self.sintheta = np.sqrt(1 - self.costheta ** 2)
        self.cosphi = np.cos(2 * np.pi * np.random.rand())
        self.sinphi = np.sqrt(1 - self.cosphi ** 2)
