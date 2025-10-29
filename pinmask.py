"""Class to create a mask of "blades" in a XBPM.

A square numpy array is created with the dimensions nbins_h x nbins_v,
corresponding to the number of pixels in horizontal and
vertical directions respectively.
"""

import numpy as np
# import matplotlib.pyplot as plt


class PinMask:
    """Create a mask of "blades" in a XBPM.

    A square numpy array is created with the dimensions nbins[0] (horiz) x
    nbins[1] (vert), corresponding to the number of pixels in horizontal and
    vertical directions.

    The standard values are based on real dimensions of the structures, in mm.
    """
    def __init__(self, prm):
        """Define the main parameters of the mask."""
        self.nbins        = prm["nbins"]
        self.ndots        = prm["ndots"]
        self.windowsize   = prm["windowsize"]
        self.pixelsize    = prm["pixelsize"]
        self.bladelength  = prm["bladelength"]
        self.pinsize      = prm["pinsize"]
        self.centerpins   = prm["centerpins"]
        self.wcenter      = self.windowsize / 2
        self.phi          = self._degtorad(prm["phi"])
        self.corneroffset = np.array([prm["corneroffset"], 0.0])
        self.coordinates  = self.pin_coordinates_calc()
        self.mask_array()

    def mask_array(self):
        """Mask to weight the intersection of pin diodes and  distribution.

        The mask values are 1 where the pin diodes are defined (dots),
        0 elsewhere.

        Warning: the usual indexing of an array is from top to bottom, but
        the coodrinates were chosen accordingly to the Cartesian systems,
        so the current indexing is reversed.
        """
        # Create mask array with zeros.
        self.mask = np.zeros((self.nbins[1], self.nbins[0]))
        coordinates_array = list()

        # Scale to array units.
        # pxnorm = 1. / self.pixelsize
        nsize = int(self.pinsize / self.pixelsize)
        # Pin size in number of pixels is odd, for symmetry.
        if nsize % 2 == 0:
            nsize += 1
        halfnsize = int(nsize / 2)

        # Run through dot sets to assign weights to pixels.
        for pinc in self.coordinates:
            # Array indices from window coordinates.
            lin = round(self.nbins[1] *
                        (pinc[1] + self.wcenter[1])/ self.windowsize[1])
            col = round(self.nbins[0] *
                        (pinc[0] + self.wcenter[0])/ self.windowsize[0])
            coordinates_array.append([lin, col])
            # Set mask.
            for ll in range(lin - halfnsize, lin + halfnsize + 1):
                for cc in range(col - halfnsize, col + halfnsize + 1):
                    self.mask[ll, cc] = 1
        self.coordinates_array = np.array(coordinates_array)

    def pin_coordinates_calc(self):
        """Create a list of the coordinates of the dots in each set.

        Each element of the list is an array with the (x, y) coordinates
        of the dots in a set, as many as defined by ndots * 4 (sets at the
        corners) or ndots * 6 (corners + horizontal center).

        Each set is initially created at the bottom left corner, then it is
        mirrored to other corners, taking into account the corner offset.

        At this stage, the coordinates and measures are set in mm. They are
        transformed onto array indices afterwards.

        phi is the azimuthal angle (in rad) of the radial set.

        Args:
            self

        Returns:
            pin_set (numpy_array): each pin's coordinates.
        """
        # Versor in phi direction.
        phivec = np.array([np.cos(self.phi), np.sin(self.phi)])

        # Pin coordinates in bottom left set.
        # The set is displaced from bottom left corner according to
        # corner offset.
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

        pin_set = (set_top_left + set_top_right +
                   set_bottom_right + set_bottom_left)

        # Include pins at central vertical line.
        if self.centerpins:
            bladecenterlength = self.bladelength * np.sin(self.phi)
            set_top_center = [
                [self.wcenter[0],
                 (1.0 - ii / self.ndots) * bladecenterlength]
                for ii in range(self.ndots)
                ]

            set_bottom_center = [
                [xx, self.windowsize[1] - yy]
                for xx, yy in set_top_center
                ]
            pin_set = (set_top_left + set_top_center + set_top_right +
                       set_bottom_right + set_bottom_center + set_bottom_left)

        # Shift center to the window's center.
        pin_set = np.array(pin_set)
        for ps in pin_set:
            ps -= self.wcenter
        return np.array(pin_set)

    def _degtorad(self, phi):
        """Convert angle from degrees to radians."""
        return phi * np.pi / 180.0
