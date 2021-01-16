"""
A simple script to transport a generation of photons through a 1D homogenous
infinite slab of material. Various variance reduction techniques to reduce the
run time of MCRT are show cased.
"""

import timeit
import numpy as np
from numba import jit

from lib.photonPackets import PhotonPacket
from lib.histogram import EscapedPhotonsHistogram


@jit
def transport_photon_packet(photon, tau_max, albedo):
    """
    Transport a photon packet through the atmosphere.
    Parameters
    ----------
    photon: PhotonPacket
        The photon to transport.
    tau_max: float
        The transverse optical depth of the atmosphere.
    albedo: float
        The scattering albedo in the atmosphere.
    """

    while 0 <= photon.coords[2] <= 1:

        # sample a random optical depth and update the position of the
        # packet

        ds_to_scatter = -np.log(np.random.rand()) / tau_max
        photon.move_photon_length_ds(ds_to_scatter)

        # if z < 0, the photon has traveled deeper into the atmosphere
        # and is lost, hence restart the photon
        # if z > 1, the photon has escaped the atmosphere
        # else it's still in the atmosphere and scatter or absorb it

        if photon.coords[2] < 0:
            photon = PhotonPacket()
        elif photon.coords[2] > 1:
            photon.escaped = True
        else:
            xi = np.random.rand()
            if xi < albedo:
                photon.isotropic_scatter()
            else:
                break

    return


def main():
    """Main function of the script"""

    n_photons = int(1e6)
    n_bins = 20
    tau_max = 7
    albedo = 1.0
    output_freq = n_photons // 10

    print('Beginning Simulation...\n')

    start = timeit.default_timer()
    hist = EscapedPhotonsHistogram(n_bins)
    theta_bins = hist.theta

    for i in range(n_photons):
        pp = PhotonPacket()
        transport_photon_packet(pp, tau_max, albedo)

        if pp.escaped:
            hist.bin_photon(pp.costheta)

        if (i + 1) % output_freq == 0:
            percent_complete = 100 * (i + 1) / n_photons
            print('{} photons ({:3.1f}%) transported.'.format(i + 1, percent_complete))

    intensity = hist.calculate_intensity(n_photons)
    stop = timeit.default_timer()

    print('\nTransport of {} packets completed in {:3.2f} seconds.'.format(n_photons, stop - start))

    return theta_bins, intensity


# =============================================================================
# Simulation
# =============================================================================

if __name__ == "__main__":
    t, i = main()
