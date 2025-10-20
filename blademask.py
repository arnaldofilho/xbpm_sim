"""Class to create a mask of "blades" in a XBPM.

A square numpy array is created with the dimensions nbins_h x nbins_v,
corresponding to the number of pixels in horizontal and
vertical directions respectively.
"""

import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy


class BladeMask:
    """Create a mask of "blades" in a XBPM.

    A square numpy array is created with the dimensions nbins[0] (horiz) x
    nbins[1] (vert), corresponding to the number of pixels in horizontal and
    vertical directions.

    The standard values are based on real dimensions of the structures, in mm.

    Version: 2025-07-22.
    """
    def __init__(self, gprm):
        """Define the main parameters of the mask."""
        self.boxsize = gprm['boxsize']
        self.pixelsize = gprm['pixelsize']
        self.nbins = gprm['nbins']
        self.corneroffset = gprm['corneroffset']
        self.bladelength = gprm['bladelength']
        self.bladethickness = gprm['bladethickness']
        self.phi = self._degtorad(gprm['phi'])
        self.bladescoordinates = self._blades_coordinates()
        self.maskarray = self.mask_array()

    def mask_array(self):
        """Create a mask to weight the intersection of pixels and blades.

        The mask values are the percentage of intersection of the blade with
        the pixel array (screen image). If a pixel is fully inside the blade
        area, the pixel value is 1, otherwise it is proportional to the
        intersected area.

        Notice that the calculations are made in real coordinates and then
        projected into the pixel array.

        Warning: the usual indexing of an array is from top to bottom, but
        the coodrinates were chosen accordingly to the Cartesian systems,
        so the current indexing is reversed.
        """
        # Create mask array with zeros.
        self.mask = np.zeros((self.nbins[0], self.nbins[1]))

        # Run through blades to assign weights to pixels.
        for ib, blade in enumerate(self.bladescoordinates):
            # Equations of lines joining the corners of the blade.
            bladeequations = [
                self._edge_line(blade[0], blade[1]),
                self._edge_line(blade[1], blade[2]),
                self._edge_line(blade[2], blade[3]),
                self._edge_line(blade[3], blade[0]),
            ]

            # Set horizontal (x) interval, and top and bottom line
            # equations for each interval.
            intervals, top_eq, bot_eq = list(), list(), list()

            # Even and odd-number blades are defined within different
            # intervals by their respective corners.
            if ib % 2 == 0:
                intervals.append([blade[0][0], blade[1][0]])
                top_eq.append(bladeequations[3])
                bot_eq.append(bladeequations[0])
                intervals.append([blade[1][0], blade[3][0]])
                top_eq.append(bladeequations[3])
                bot_eq.append(bladeequations[1])
                intervals.append([blade[3][0], blade[2][0]])
                top_eq.append(bladeequations[2])
                bot_eq.append(bladeequations[1])
            else:
                intervals.append([blade[3][0], blade[2][0]])
                top_eq.append(bladeequations[2])
                bot_eq.append(bladeequations[3])
                intervals.append([blade[2][0], blade[0][0]])
                top_eq.append(bladeequations[1])
                bot_eq.append(bladeequations[3])
                intervals.append([blade[0][0], blade[1][0]])
                top_eq.append(bladeequations[1])
                bot_eq.append(bladeequations[0])

            # Treat corners of the blade: check whether bottom and top lines
            # cross each other inside the pixel.
            self._pixel_corner_weight(blade, bladeequations)

            # Run vertically in each interval to assign weight to pixels.
            # Treat bulk and borders of the blades, except for corners.
            for ii, interval in enumerate(intervals):
                # Define the interval corresponding to current blade. Take the
                # limits of the box, [0, boxsize[0/1]], into consideration.
                # xA, xB = max(interval[0], 0), min(interval[1], self.boxsize)
                # # xA, xB = interval[0], interval[1]

                # Horizontal range to be scanned, in pixels coordinates, not
                # physical, to guarantee each pixel will be analysed only once.
                # Nx = int((xB - xA) / pixelsize)  # Number of intervals
                # (round up).
                ncolmin = max(0, int(interval[0] / self.pixelsize))
                ncolmax = min(int(interval[1] / self.pixelsize) + 1,
                              self.nbins[1] - 1)

                # Run over all pixels.
                # Horizontal range to be scanned.
                for ncol in range(ncolmin, ncolmax):
                    # Vertical limits in physical (real) coordinates.
                    rx = ncol * self.pixelsize
                    ymin = min(
                        self._linear(rx, *bot_eq[ii]),
                        self._linear(rx + self.pixelsize, *bot_eq[ii]),
                    )
                    ymax = max(
                        self._linear(rx, *top_eq[ii]),
                        self._linear(rx + self.pixelsize, *top_eq[ii]),
                    )
                    # Limits in array coordinates.
                    nlinmin, nlinmax = (
                        round(ymin / self.pixelsize),
                        round(ymax / self.pixelsize + 1),
                    )
                    # int(ymin / self.pixelsize),
                    # int(ymax / self.pixelsize + 1),

                    # Vertical range to be scanned.
                    nlinrange = range(max(nlinmin - 1, 0),
                                      min(nlinmax + 1, self.nbins[0] - 1))
                    for nlin in nlinrange:
                        self.mask[nlin, ncol] = self._pixel_weight(
                            nlin, ncol, bot_eq[ii], top_eq[ii]
                        )
        return self.mask

    def _blades_coordinates(self):
        """Create a list of the coordinates of the four blades.

        Each element of the list is an array with the coordinates of the
        corners of the blade. Each blade is initially created at the center
        of the coordinates system, then it is rotated and shifted accodingly.
        At this stage, the coordinates and measures are set in mm. They are
        transformed onto array indices afterwards.

        phi is the azimuthal angle (in rad) of the blades; not to be mistaken
        with the theta angle, relative to the cut edge of the blades,
        upon which the x-ray is incident.

        Returns:
            blades (list): each blades' corners' coordinates.
        """
        # Define the blade corners. The order is counterclockwise,
        # starting from the bottom left of the blade. The initial position
        # of the blade is: length in vertical, thickness in horziontal,
        # geometric center at zero. The blade is thereafter rotated and
        # shifted to its final location.
        halfthickb = 0.5 * self.bladethickness
        halfheightb = 0.5 * self.bladelength
        bladecoordinates = [[-halfthickb, -halfheightb],
                            [halfthickb, -halfheightb],
                            [halfthickb, halfheightb],
                            [-halfthickb, halfheightb]]

        # List of all four blades.
        blades = list(range(4))

        # Rotate two blades counterclockwise.
        rotmat = self._matrix_rotation(self.phi)
        #
        rotbladecoord = list()
        for bc in bladecoordinates:
            rotbladecoord.append(np.matmul(rotmat, bc))
        # Create copies of the rotated blades.
        blades[1] = deepcopy(np.array(rotbladecoord))
        blades[3] = deepcopy(blades[1])

        # Rotate two blades clockwise.
        rotmat = self._matrix_rotation(-self.phi)
        #
        rotbladecoord = list()
        for bc in bladecoordinates:
            rotbladecoord.append(np.matmul(rotmat, bc))
        # Create copies of the rotated blades.
        blades[0] = np.array(rotbladecoord)
        blades[2] = deepcopy(blades[0])

        # Find the coordinates of the appropriate blade's corner to be set on
        # the box boundaries and shift the blade accordingly.
        # Map: Blades[a][b][c] corresponds to
        # a = # blade (0... 3);
        # b = # corner (counterclockwise);
        # c = coordinate x (0) or y (1).
        #
        # An offset relative to the corner is added as a horizontal shift,
        # since blades are not necessarily located at the corners of the box.

        # Bottom, left blade.
        dx = -blades[0][0][0] + self.corneroffset
        dy = -blades[0][0][1]
        blades[0] += np.array((dx, dy))

        # Bottom, right blade.
        dx = self.boxsize[0] - blades[1][1][0] - self.corneroffset
        dy = -blades[1][1][1]
        blades[1] += np.array((dx, dy))

        # Top, right blade.
        dx = self.boxsize[0] - blades[2][2][0] - self.corneroffset
        dy = self.boxsize[1] - blades[2][2][1]
        blades[2] += np.array((dx, dy))

        # Top, left blade.
        dx = -blades[3][3][0] + self.corneroffset
        dy = self.boxsize[1] - blades[3][3][1]
        blades[3] += np.array((dx, dy))

        return blades

    def _pixel_weight(self, nlin, ncol, bot, top):  # noqa: C901
        """Calculate the intersection of the pixel with the blade.

        Args:
            ncol (int) : horizontal bottom left corner of the pixel
            nlin (int) : vertical bottom left corner of the pixel
            bot (tuple): coefficients of the linear equations which
                        define the blade bottom edges;
            top (tuple): coefficients of the linear equations which
                        define the blade top edges;


        Obs.: top/bottom refers to _blade_ edges;
        '0'/'f' refers to _pixel's_ bottom/top edges, respectively;
        'a' / 'b' refers to left/right pixel's coordinates (the interval),
        respectively.
        """
        # Pixel interval.
        xa, y0 = ncol * self.pixelsize, nlin * self.pixelsize
        xb, yf = xa + self.pixelsize, y0 + self.pixelsize
        pixelarea = self.pixelsize * self.pixelsize

        # Weight is 1 if the pixel is fully inside the blade.
        yatop, ybtop = self._linear(xa, *top), self._linear(xb, *top)
        yabot, ybbot = self._linear(xa, *bot), self._linear(xb, *bot)
        if yatop >= yf and ybtop >= yf and yabot <= y0 and ybbot <= y0:
            return 1.0

        # Analysis of edge cases.

        # Horizontal coordinate of the blade edge lines within the pixel
        # interval, x = (y - b) / a.
        #
        # Blade top line related to pixel bottom/top.
        x0top = (y0 - top[1]) / top[0]
        xftop = (yf - top[1]) / top[0]
        # Blade bottom line related to pixel bottom/top.
        x0bot = (y0 - bot[1]) / bot[0]
        xfbot = (yf - bot[1]) / bot[0]

        # Top line crosses pixel's upper edge
        if xa <= xftop <= xb:
            # and top line crosses left edge.
            if y0 <= yatop <= yf:
                return (1 - ((xftop - xa) * (yf - yatop) * 0.5) / pixelarea)

            # and top line crosses right edge.
            if y0 <= ybtop <= yf:
                return (1 - ((xb - xftop) * (yf - ybtop) * 0.5) / pixelarea)

            # and top line crosses bottom edge.
            if xa <= x0top <= xb:
                # Ascending line.
                if x0top <= xftop:
                    return (((xb - x0top) + (xb - xftop))
                            * 0.5 * self.pixelsize / pixelarea)

                # Descending line.
                return (((x0top - xa) + (xftop - xa))
                        * 0.5 * self.pixelsize / pixelarea)

        # Bottom line crosses pixel's upper edge
        if xa <= xfbot <= xb:
            # and bottom line crosses left edge.
            if y0 <= yabot <= yf:
                return (yf - yabot) * (xfbot - xa) * 0.5 / pixelarea

            # and top line crosses right edge.
            if y0 <= ybbot <= yf:
                return (xb - xfbot) * (yf - ybbot) * 0.5 / pixelarea

            # and bottom line crosses bottom edge.
            if xa <= x0bot <= xb:
                # Ascending line.
                if x0bot <= xfbot:
                    return (((xfbot - xa) + (x0bot - xb))
                            * 0.5 * self.pixelsize / pixelarea)

                # Descending line.
                return (((xb - xfbot) + (xb - x0bot))
                        * 0.5 * self.pixelsize / pixelarea)

        return 0.0

    #
    def _pixel_corner_weight(self, corners, blade_eqs):
        """Assign weights to the pixels at blade's corners."""
        for ic, corner in enumerate(corners):
            xc, yc = corner

            # Skip if corner lies outside the box array.
            if (xc < 0 or xc > self.boxsize[0] or
                yc < 0 or yc > self.boxsize[1]):
                continue

            # Identify pixel (in array) coordinates. It must be inbounds.
            nx = min(int(xc / self.pixelsize), self.nbins[0] - 1)
            ny = max(int(yc / self.pixelsize), self.nbins[1] - 1)
            if nx < 0:
                nx = 0
            if ny < 0:
                ny = 0
            xa, y0 = nx * self.pixelsize, ny * self.pixelsize
            xb, yf = xa + self.pixelsize, y0 + self.pixelsize

            # Calculate crossing point.
            topeq, boteq = blade_eqs[(ic - 1) % 4], blade_eqs[ic]
            xcross = (topeq[1] - boteq[1]) / (boteq[0] - topeq[0])
            ycross = self._linear(xcross, *topeq)

            # Next step, TO BE DONE: find the intersection area.
            # Idea: besides the crossing point coordinates (above), find the
            # side of the blade which intersects the pixel, calculate the
            # height of the triangle formed by the this blade corner
            # (crossing point) and the corresponding pixel edge intercepted
            # by the blade, calculate the area of the triangle and discount
            # the other two minor triangles outside the pixel.

            # Temporary dummy line to avoid warnings from CodeVS.
            ycross += xb + yf

            # WARNING: Temporary!
            self.mask[nx, self.nbins[1] - 1 - ny] = 0.25

        return

    """Mathematical methods: linear equation, integral of a line,
    matrix rotation in 2D."""

    def _linear(self, x, a, b):
        """Straight line equation."""
        return a * x + b

    def _edge_line(self, pp, qq):
        """Calculate the line equation joining two points, P and Q.

        Args:
            pp (tuple): first point;
            qq (tuple): second point.

        Returns:
            [a, b] (list): coefficients of the line ax + b.
        """
        a = (qq[1] - pp[1]) / (qq[0] - pp[0])
        b = pp[1] - a * pp[0]
        return [a, b]

    def _integral_linear(self, coefficients, interval):
        """Integral of a linear equation.

        Args:
            coefficients (tuple): coefficients of the line ax + b;
            interval (tuple): the interval of integration.

        Returns:
            the value of the integral in the interval.

        """
        a, b = coefficients
        xa, xb = interval
        return (xb - xa) * (a * (xa + xb) / 2.0 + b)

    def _matrix_rotation(self, phi):
        """Rotation matrix.

        Args:
            phi (float): angle (in rad).

        Returns:
            (numpy array) 2x2 rotation matrix.
        """
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        return np.array([[cphi, -sphi], [sphi, cphi]])

    def _degtorad(self, phi):
        """Convert from degree to radian."""
        return np.pi / 180 * phi
