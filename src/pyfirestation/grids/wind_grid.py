"""
Grid structure for wind speed and height data.

"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr

from .base_grid import BaseGrid
from ..utils.custom_types import WindCellLike


@dataclass
class WindCell(WindCellLike):
    """Wind speed and height data structure for a grid cell."""
    vel_x: float
    vel_y: float
    vel_z: float
    height: float


class WindGrid(BaseGrid):
    __slots__ = ["_dataset"]

    def __init__(self,
                 speed_array: npt.ArrayLike,
                 height_array: npt.ArrayLike,
                 cx: npt.ArrayLike,
                 cy: npt.ArrayLike,
                 *,
                 depth_wise_stack: bool = False):

        # Convert 3xMxN array to MxNx3
        if depth_wise_stack:
            speed_array = np.dstack([speed_array])

        # print(height_array.shape)
        self._dataset = xr.Dataset(data_vars=dict(
                                    velocity=(["cx", "cy", "components"], speed_array),
                                    height=(["cx", "cy"], height_array),
                                  ),
                                  coords=dict(
                                      x=cx,
                                      y=cy,
                                      components=["vel_x", "vel_y", "vel_z"]
                                  ),
                                  attrs=dict(
                                      description="Wind grid.",
                                      units="m/s, m"
                                  ))

    def __getitem__(self, cell: Iterable) -> WindCell:
        cell_data = super().__getitem__(cell)
        x, y, z = cell_data["velocity"]
        h = cell_data["height"]
        return WindCell(vel_x=x, vel_y=y, vel_z=z, height=h)
