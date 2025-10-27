"""Class to create a mask of "blades" in a XBPM.

A square numpy array is created with the dimensions nbins_h x nbins_v,
corresponding to the number of pixels in horizontal and
vertical directions respectively.
"""

import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy


class PinMask:
    """Create a mask of "blades" in a XBPM.

    A square numpy array is created with the dimensions nbins[0] (horiz) x
    nbins[1] (vert), corresponding to the number of pixels in horizontal and
    vertical directions.

    The standard values are based on real dimensions of the structures, in mm.
    """
    def __init__(self, gprm):
        """Define the main parameters of the mask."""
        self.nbins         = gprm["nbins"]
        self.ndots         = gprm["ndots"]
        self.windowsize    = gprm["windowsize"]
        self.pixelsize     = gprm["pixelsize"]
        self.phi           = self._degtorad(gprm["phi"])
        self.bladelength   = gprm["bladelength"]
        self.pinthickness  = gprm["pinthickness"]
        self.corneroffset  = gprm["corneroffset"]
        self.coordinates   = self.pin_coordinates_calc()
        self.mask_array()

    def mask_array(self):
        """Create a mask to weight the intersection of pin diodes and radiation distribution.

        The mask values are 1 where the pin diodes are defined (dots), 0 elsewhere.
        
        Warning: the usual indexing of an array is from top to bottom, but
        the coodrinates were chosen accordingly to the Cartesian systems,
        so the current indexing is reversed.
        """
        # Create mask array with zeros.
        self.mask = np.zeros((self.nbins[1], self.nbins[0]))
        coordinates_array = list()

        # Scale to array units.
        pxnorm = 1. / self.pixelsize

        # DEBUG
        print(f"\n\n>>> (MASK ARRAY) nbins      = {self.nbins}\n")
        print(f">>> (MASK ARRAY) windowsize = {self.windowsize}\n")
        # DEBUG

        # Run through dot sets to assign weights to pixels.
        for idx, pinset in enumerate(self.coordinates):
                
            # DEBUG
            print(f">>> (MASK ARRAY) pinset {idx} = \n{pinset}\n")
            # DEBUG

            for idd, sdot in enumerate(pinset):
                lin = round(self.nbins[1] * (1 - sdot[1] / self.windowsize[1]))
                col = round(sdot[0] / self.windowsize[0] * self.nbins[0])
                coordinates_array.append([lin, col])

                # DEBUG
                print(f">>> (MASK ARRAY) \nlin ({idd}): {sdot[1]} -> {lin}\n")
                print(f">>> (MASK ARRAY) \ncol ({idd}): {sdot[0]} -> {col}\n")
                # DEBUG

                for c in range():
                    self.mask[lin, col] = 1

        self.coordinates_array = np.array(coordinates_array)

    def pin_coordinates_calc(self):
        """Create a list of the coordinates of the dots in each blade.

        Each element of the list is an array with the coordinates of the
        dots in a set, as many as defined by ndots.

        Each blade is initially created at the bottom left corner, then it is 
        mirrored to other corners, taking into account the corner offset.

        At this stage, the coordinates and measures are set in mm. They are
        transformed onto array indices afterwards.

        phi is the azimuthal angle (in rad) of the blades; not to be mistaken
        with the theta angle, relative to the cut edge of the blades,
        upon which the x-ray is incident.

        Returns:
            blades (list): each blades' corners' coordinates.
        """
        # Versor in phi direction.       
        phivec = np.array([np.cos(self.phi), np.sin(self.phi)])

        # Pin coordinates in bottom left set.
        # The set is displaced from bottom left corner according to corner offset.
        setvec = phivec * self.bladelength
        set_bottom_left = [
            (1.0 - ii / self.ndots) * setvec + self.corneroffset
            for ii in range(self.ndots)
            ]

        # Mirror first set to other corners of the box.
        set_bottom_right = [
            [self.windowsize[0] - xx , yy]
            for xx, yy in set_bottom_left
            ]
        
        set_top_left = [
            [xx, self.windowsize[1] - yy]
            for xx, yy in set_bottom_left
            ]

        set_top_right = [
            [xx , self.windowsize[1] - yy]
            for xx, yy in set_bottom_right
            ]
        
        # DEBUG
        pinset = np.array([set_top_left, set_top_right, set_bottom_right, set_bottom_left])
        print(f"\n\n ##### PIN COORDINATES #####\n {pinset}")
        # DEBUG

        # From top left, clockwise.
        return np.array([set_top_left, set_top_right, set_bottom_right, set_bottom_left])


    def _degtorad(self, phi):
        """Convert angle from degrees to radians."""
        return phi * np.pi / 180.0
