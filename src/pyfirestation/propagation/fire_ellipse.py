"""
This module defines the fire ellipse shape that is used for the propagation simulation.

"""

from enum import Enum, auto
from typing import Optional

import numpy as np

from ..utils.cellindex import CellIndex


class FireEllipseType(Enum):
    """Defines the two fire ellipse varieties.

    In the fire propagation simulation, an ellipse shape is considered in modelling the fire spread. This ellipse can be
    either a simple ellipse, for equivalent wind speeds less than 0.2 meters per second, or a double ellipse -
    effectively two half ellipses - for equivalent wind speeds greater than 0.2 meters per second. This Enum defines
    these two cases.
    """
    SINGLE = auto()
    DOUBLE = auto()


class FireEllipse:
    def __init__(self, wind_speed: float, theta: float, *, d: float = 1.0) -> None:
        """Defines the ellipse considered in the fire spread simulation."""
        self._mode: FireEllipseType

        self._wind_speed: float
        self.wind_speed = wind_speed

        self.theta = theta
        self.d = d

    @property
    def mode(self) -> FireEllipseType:
        """The fire ellipse type.

        Check FireEllipseType for more information.

        :return: The fire ellipse type (FireEllipseType).
        """
        return self._mode

    def update_mode(self) -> None:
        if self.wind_speed > 0.2:
            self._mode = FireEllipseType.DOUBLE
        else:
            self._mode = FireEllipseType.SINGLE

    @property
    def wind_speed(self) -> float:
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value: float) -> None:
        self._wind_speed = float(value)
        self.update_mode()

    @property
    def c(self) -> float:
        if self.mode is FireEllipseType.DOUBLE:
            return 0.492 * np.exp(-0.413 * self.wind_speed) * self.d
        else:
            return 2 * self.a2 - self.d

    @property
    def p(self) -> float:
        """Semilatus Rectum"""
        if self.mode is FireEllipseType.DOUBLE:
            return 0.542 * np.exp(-0.3317 * self.wind_speed) * self.d
        else:
            return self.b ** 2 / self.a2

    @property
    def a1(self) -> float:
        """Semi-major axis of back ellipse (or main ellipse in single ellipse mode)"""
        if self.mode is FireEllipseType.DOUBLE:
            return (2.502 * (196.86 * self.wind_speed) ** (-0.3)) * self.d
        else:
            return self.a2

    @property
    def a2(self) -> float:
        """Semi-major axis of front ellipse (or None in single ellipse mode)"""
        if self.mode is FireEllipseType.DOUBLE:
            return self.d + self.c - self.a1  # Different from paper
        else:
            l = 0.5 / self.b
            return 1 - np.sqrt(1 - 1 / l ** 2) / (1 + np.sqrt(1 - 1 / l ** 2))

    @property
    def b(self) -> float:
        """Semi-minor axis"""
        if self.mode is FireEllipseType.DOUBLE:
            return 0.534 * np.exp(-0.2566 * self.wind_speed) * self.d
        else:
            # WARNING! This is different from Lopes (2002)
            # - Using semi-minor axis here (div by 2)
            return 0.5 * self.d * (1 + 0.0012 * (2.237 * self.wind_speed) ** 2.154) ** -1

    @property
    def new_b(self) -> float:
        l = 0.5 / self.b
        return self.a1 / l

    @property
    def f(self) -> float:
        """Distance between focus and centre"""
        return self.a1 - self.c

    def radius(self, target: CellIndex, *, origin: Optional[CellIndex] = None) -> float:
        """Return the radius of the ellipse in the direction of a target cell, given its raster coordinates.

        :param target: The raster coordinates (CellIndex) of the target cell which coincides with the fire spread
                       direction.
        :param origin: The raster coordinates (CellIndex) of the origin cell (focus of the ellipse). If omitted,
                       (i, j) = (0, 0).
        :return: The radius of the spread ellipse in the direction defined by the target cell.
        """
        if not origin:
            origin = CellIndex(i=0, j=0)

        i, j = target - origin

        angle_dir = np.arctan2(j, i) % (2 * np.pi)
        ang_wind = self.theta
        cos_t = np.cos(angle_dir - ang_wind)
        sin_t = np.sin(angle_dir - ang_wind)

        if self.mode is FireEllipseType.SINGLE:
            a = self.a1
            l = 0.5/self.b
            d = np.sqrt(1 - 1 / l ** 2) / (1 + np.sqrt(1 - 1 / l ** 2))
            b = self.new_b
        else:
            d = -self.f
            b = self.b
            if cos_t >= np.cos(np.arctan2(self.b, (self.a1-self.c))):
                a = self.a2
            else:
                a = self.a1

        ca = 1/((a ** 2 * b ** 2) / ((b * cos_t) ** 2 + (a * sin_t) ** 2))
        cb = 2*d*cos_t/a**2
        cc = d**2/a**2 - 1
        return (-cb + np.sqrt(cb**2-4*ca*cc))/(2*ca)
