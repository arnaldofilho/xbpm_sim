"""Classes to calculate beam position."""

import numpy as np


class BeamPosition():
    """Beam's position by different methods, given the flux on each blade."""
    def __init__(self, pindiodes, prm):
        # def __init__(self, intervals, windowsize=(10, 10), theta=0):
        """Initialize general parameters.

        Args:
            pindiodes (Pinmask object): pin diodes' coordinates;
            prm (dict): general parameters of the simulation.
        """
        # self.intervals = intervals
        # self.coordinates = pincoordinates
        self.coordinates_array = pindiodes.coordinates_array
        self.coordinates       = pindiodes.coordinates
        self.pinsize           = prm["pinsize"] / prm["pixelsize"]
        # self.windowsize = gprm['windowsize']
        # self.pixelsize = gprm['pixelsize']
        # self.nbins = gprm['nbins']

    def center_of_mass(self, hist_img):
        """Calculate the flux on each pin diode.

        Args:
            hist_img (numpy array): histogram array which defines the
                'photons' distribution image.

        Returns:
            beam position and the fluxes on the diodes.
        """
        self.flux = list()
        # Incidence angle correction.
        bsum = 0.
        bpos = np.array([0., 0.])
        fluxes = list()
        halfps = int(self.pinsize / 2)

        # Calculate the flux on every blade.
        for pin_ac, pin_cc in zip(self.coordinates_array,
                                  self.coordinates, strict=True):
            pa0, pb0 = pin_ac[0] - halfps, pin_ac[0] + halfps + 1
            pa1, pb1 = pin_ac[1] - halfps, pin_ac[1] + halfps + 1
            # fluxval = hist_img[pd[0], pd[1]]
            fluxval = np.sum(hist_img[pa0:pb0, pa1:pb1])
            bpos += fluxval * pin_cc
            bsum += fluxval
            fluxes.append([pin_cc, fluxval])
        return bpos / bsum, fluxes
