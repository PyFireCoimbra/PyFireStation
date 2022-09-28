"""
Grid structure for terrain elevation (height) data.

"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr

from .base_grid import BaseGrid
from ..utils.custom_types import TerrainCellLike


@dataclass
class TerrainCell(TerrainCellLike):
    """Terrain height and slope data structure for a grid cell."""
    height: float
    height_center: float
    slope: float
    slope_dir: float


class TerrainGrid(BaseGrid):
    def __init__(self,
                 height_array: npt.ArrayLike,
                 cx: npt.ArrayLike,
                 cy: npt.ArrayLike) -> None:

        cx = np.array(cx)
        cy = np.array(cy)
        cellsize = cx[1] - cx[0]

        height = np.array(height_array)
        height_center = self.calc_height_center(height)
        slope = self.calc_slope_ang(height, cellsize=cellsize)
        slope_dir = self.calc_slope_max_dir(height, cellsize=cellsize)

        self._dataset = xr.Dataset(
            data_vars=dict(
                height=(["x", "y"], height),
                height_center=(["x", "y"], height_center),
                slope=(["x", "y"], slope),
                slope_dir=(["x", "y"], slope_dir)
            ),
            coords=dict(
                x=cx,
                y=cy,
            ),
            attrs=dict(
                description="Terrain grid.",
                units="m, rad"
            )
        )

    @staticmethod
    def calc_height_center(height: npt.NDArray) -> npt.NDArray:
        """Computes the height at the center of grid cells, given the heights at their corners.

        :param height: A Numpy array containing the height of the terrain at the corners of each cell, for a regular
        grid of square cells.
        :return: A Numpy array containing the height o the terrain the center of each cell.
        """
        height_center = np.empty_like(height)
        height_center[0:-1, 0:-1] = (height[:-1, :-1] + height[1:, :-1] + height[:-1, 1:] + height[1:, 1:])/4
        # TODO: Check fill of limits
        height_center[:, -1] = height_center[-1, :] = 0
        return height_center

    @staticmethod
    def calc_slope_ang(height: npt.NDArray, cellsize: float) -> npt.NDArray:
        """Computes the slope angle at the center of grid cells, given the heights at their corners.

        :param height: A Numpy array containing the height of the terrain at the corners of each cell, for a regular
        grid of square cells.
        :param cellsize: The size of the side of each cell, considering square cells.
        :return: A Numpy array containing the slope angle, i.e. the inclination of the terrain, at the center of each
        cell.
        """
        height_grad = np.gradient(height, cellsize)
        angles = 1/2 * np.pi - np.arccos(np.sqrt((height_grad[1] ** 2 + height_grad[0]**2) /
                                                 (height_grad[1] ** 2 + height_grad[0]**2 + 1)))
        return angles

    @staticmethod
    def calc_slope_max_dir(height: npt.NDArray, cellsize: float) -> npt.NDArray:
        """Computes the direction of the slope in each cell, given the heights at their corners.

        :param height: A Numpy array containing the height of the terrain at the corners of each cell, for a regular
        grid of square cells.
        :param cellsize: The size of the side of each cell, considering square cells.
        :return: A Numpy array containing the direction of the slope, i.e. the direction of maximum descent.
        """
        grad = np.gradient(height, cellsize)
        angles = np.arctan2(grad[1], grad[0])
        angles = np.mod(angles, 2 * np.pi)
        return angles

    def __getitem__(self, cell: Iterable) -> TerrainCell:
        cell_data = super().__getitem__(cell)
        return TerrainCell(**cell_data)
