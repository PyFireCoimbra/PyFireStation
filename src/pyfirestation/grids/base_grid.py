"""
Abstract base grid structure for simulation data that can be used for inheritance.

"""


from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Any, Union

import numpy as np
import numpy.typing as npt
import xarray as xr

from ..utils.cellindex import CellIndex


class InvalidCellDescriptorError(Exception):
    """Exception class for invalid cell descriptor object."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class BaseGrid(ABC):
    """Base implementation of grid structures to be inherited by other custom grid classes."""
    __slots__ = ["_dataset"]

    _dataset: xr.Dataset

    @property
    def cx(self) -> npt.NDArray:
        """The 'x' coordinates of the grid."""
        return self.dataset.coords["x"].values

    @property
    def cy(self) -> npt.NDArray:
        """The 'y' coordinates of the grid."""
        return self.dataset.coords["y"].values

    @property
    def nx(self) -> int:
        """The number of cells on the 'x' axis."""
        return len(self.cx)

    @property
    def ny(self) -> int:
        """The number of cells on the 'y' axis."""
        return len(self.cy)

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the grid."""
        # xr.Dataset doesn't provide .shape property (only xr.DataArray), so extract from cx and cy to make agnostic
        # of implementation
        return len(self.cx), len(self.cy)

    @property
    def cellsize(self) -> float:
        """The size of each grid cell (side, considering square cells)."""
        return float(np.mean(np.ediff1d(self.cx)))

    @property
    def dataset(self) -> xr.Dataset:
        """The underlying raw XArray dataset."""
        return self._dataset

    @staticmethod
    def _check_cell_descriptor_len(descriptor: Iterable) -> None:
        if len(tuple(descriptor)) != 2:
            raise InvalidCellDescriptorError(
                "Cell descriptor should be an iterable with \'x\' and \'y\' raster coordinates.")

    @classmethod
    def _check_cell_coordinates(cls, descriptor: Iterable) -> Tuple[float, float]:
        cls._check_cell_descriptor_len(descriptor)
        i, j = descriptor
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError(f"Cell indices must be of type int or float. Got {type(i)=} and {type(j)=}.")
        return i, j

    @classmethod
    def _check_cell_indices(cls, descriptor: Iterable) -> Tuple[int, int]:
        cls._check_cell_descriptor_len(descriptor)
        i, j = descriptor
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError(f"Cell indices must be of type int. Got {type(i)=} and {type(j)=}.")
        return i, j

    @abstractmethod
    def __getitem__(self, cell: Iterable) -> Any:  # TODO: Fix this type
        i, j = self._check_cell_indices(cell)
        item = {var: self.dataset[var].values[i, j] for var in self.dataset.data_vars}
        return item

    def _get_cell_indices(self, cell: Iterable) -> Tuple[int, int]:
        x, y = cell
        # Note: this seems to be faster than using: round((origin-x) / cellsize)
        # Note: If coordinate arrays (cx and cy) are in respect fo llcorner, adding half the cell size is required
        # before computing np.argmin
        i = np.argmin(np.abs(self.cx + self.cellsize/2 - x))
        j = np.argmin(np.abs(self.cy + self.cellsize/2 - y))
        return int(i), int(j)

    def get_cell_indices(self, cells: Iterable) -> Union[CellIndex, Tuple[CellIndex, ...]]:
        """"""
        cells = list(cells)
        cells_indices = []

        for cell in cells:
            self._check_cell_descriptor_len(cell)
            x, y = self._get_cell_indices(cell)
            cells_indices.append(CellIndex(x, y))

        if len(cells_indices) == 1:
            return cells_indices[0]
        else:
            return tuple(cells_indices)

    # @classmethod
    # @abstractmethod
    # def from_netcdf(cls, path: PathLike):
    #     pass
