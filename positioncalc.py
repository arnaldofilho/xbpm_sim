"""Classes to calculate beam position."""

import numpy as np


class BeamPosition():
    """Beam's position by different methods, given the flux on each blade."""
    def __init__(self, pincoordinates_array, gprm):
        # def __init__(self, intervals, windowsize=(10, 10), theta=0):
        """Initialize general parameters.

        Args:
            hist_img (numpy array): histogram array which defines the 'photons'
                    distribution image
            pincoordinates_array (list): pin diodes' coordinates;
            # gprm (dict): general parameters to define the blades geometry.
        """
        # self.intervals = intervals
        # self.coordinates = pincoordinates
        self.coordinates_array = pincoordinates_array
        # self.windowsize = gprm['windowsize']
        # self.pixelsize = gprm['pixelsize']
        # self.nbins = gprm['nbins']

    def center_of_mass(self, hist_img):
        """Calculate the flux on each pin diode."""
        self.flux = list()
        # Incidence angle correction.
        bsum = 0.
        bpos = np.array([0., 0.])
        fluxes = list()

        # Calculate the flux on every blade.
        for pd in self.coordinates_array:
            fluxval = hist_img[pd[0], pd[1]]
            bpos += fluxval * pd
            bsum += fluxval
            fluxes.append([pd, fluxval]) 
        return bpos / bsum, fluxval
