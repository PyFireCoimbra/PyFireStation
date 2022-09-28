"""
Adjacency modes between cells used in the fire propagation simulation.

In the fire simulation module, the propagation of wildfire is modelled by considering a set of ignition cells and
then proceeding to calculate the time required for the fire to spread to adjacent non-burning cells. The definition of
the adjacent (i.e. 'neighbouring') cells, as well as other methods for computing the distance and coordinates of said
cells, is provided in this module.

"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Tuple, Union
from typing import NamedTuple

import numpy as np

from ..utils.cellindex import CellIndex


class GridLimits(NamedTuple):
    """Defines the limits of a raster grid."""

    i_min: Union[int, float]
    i_max: Union[int, float]
    j_min: Union[int, float]
    j_max: Union[int, float]

    def is_inside(self, cell: CellIndex) -> bool:
        """Checks if a cell is within the raster limits."""
        return self.i_min <= cell.i < self.i_max and self.j_min <= cell.j < self.j_max

    def __contains__(self, item: object) -> bool:
        if isinstance(item, CellIndex):
            return self.is_inside(item)
        else:
            raise NotImplementedError("Membership (\'__contains__\') only defined for CellIndex objects.")


class CellAdjacencyMode(IntEnum):
    """Defines the available cell adjacency modes.

    For the fire propagation simulation, we may consider two sets of neighbouring cells for which a burning cell may
    spread to. For the first mode - ADJ08 -, it is considered the eight closest cells to the origin cell. For the second
    mode - ADJ16 -, we consider the eight closest cells, as well as eight additional cells, namely those with relative
    displacements (i.e. relative raster coordinates to the origin cell): (i, j) = (+-1, +-2) and (i, j) =  (+-2, +-1).
    """
    ADJ08 = 8
    ADJ16 = 16

    @classmethod
    def get_mode(cls, val: Union[str, int]) -> CellAdjacencyMode:
        """Reverse look-up of mode by name/number of adjacent cells.

        :param val: The number of adjacency cells of the respective mode.
        :return: CellAdjacencyMode object corresponding to the provided number of cells.
        """
        if isinstance(val, int):
            val = str(val)

        if isinstance(val, str):
            val = val.strip()
            if val == "8":
                return cls.ADJ08
            elif val == "16":
                return cls.ADJ16

        raise InvalidCellAdjacencyModeError("No matching cell adjacency mode.")


class InvalidCellAdjacencyModeError(Exception):
    """Exception class for invalid cell adjacency mode."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CellAdjacency:
    """Provides the indices and the distances of cells contained in the adjacency region."""

    def __init__(self, mode: Union[CellAdjacencyMode, str, int], *, cell_size: float = 1) -> None:
        if isinstance(mode, CellAdjacencyMode):
            if mode not in CellAdjacencyMode:
                raise InvalidCellAdjacencyModeError(f"Invalid adjacency mode type.")
        else:
            mode = CellAdjacencyMode.get_mode(mode)

        self._distances: Dict[CellIndex, float]
        self._mode = mode
        self.cell_size = cell_size

    @property
    def mode(self) -> CellAdjacencyMode:
        """Cell adjacency mode (i.e. number and position of neighbouring cells)."

        See CellAdjacencyMode for more information on the available modes.

        :return: CellAdjacencyMode object corresponding to the adjacency mode selected.
        """
        return self._mode

    @property
    def indices(self) -> Tuple[CellIndex, ...]:
        """Relative raster displacement (i.e. relative to (i, j) = (0, 0)) of adjacent cells for the selected mode.

        Depending on the adjacency mode selected, a tuple containing the relative (displacement) raster coordinates is
        provided. See CellAdjacencyMode for more information on the available modes.

        :return: Tuple of CellIndex objects containing the relative indices of adjacent cells.
        """
        adjacent_cells_close = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]
        adjacent_cells_far = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        if self.mode is CellAdjacencyMode.ADJ08:
            adjacent_cells = adjacent_cells_close
        elif self.mode is CellAdjacencyMode.ADJ16:
            adjacent_cells = adjacent_cells_close + adjacent_cells_far
        else:
            raise InvalidCellAdjacencyModeError("Invalid adjacency mode type.")
        return tuple([CellIndex(i=i, j=j) for i, j in adjacent_cells])

    @property
    def cell_size(self) -> float:
        """The size of the side of each grid cell, used for calculating the real distance between adjacent cells.

        This property returns the size of each grid cell (considering square cells) provided by the user. By setting
        this value, the distances between adjacent cells are automatically re-calculated.

        :return: The size of each grid cell (side; square cells).
        """
        return self._cell_size

    @cell_size.setter
    def cell_size(self, size: float) -> None:
        self._cell_size = float(size)
        self._distances = {cell: size * np.sqrt(cell.i ** 2 + cell.j ** 2) for cell in self.indices}

    @property
    def distances(self) -> Dict[CellIndex, float]:
        """The relative distances (i.e. relative to (i, j) = (0, 0)) between adjacent cells for the selected mode.

        :return: A dictionary with the relative cell displacement in raster coordinates and the corresponding distance.
        """
        return self._distances

    def adjacent_cells(self,
                       cell: CellIndex,
                       *,
                       i_min: Union[int, float] = 0,
                       i_max: Union[int, float] = float("inf"),
                       j_min: Union[int, float] = 0,
                       j_max: Union[int, float] = float("inf")) -> Dict[CellIndex, float]:
        """Provides a dictionary of coordinates and distances of adjacent cells with respect to a given raster cell.

        Given an origin cell - defined by its raster coordinates by a CellIndex object - this method returns a
        dictionary where the keys are CellIndex objects with the raster coordinates of the adjacent cells - as
        defined by the adjacency mode - and the values are the real distance between said cells and the origin cell.
        Additionally, one can provide grid limits which effectively filter out adjacent cells that may lie outside the
        area of interest.

        :param cell: CellIndex object with the raster coordinates of the origin cell.
        :param i_min: The lower limit for the 'i' ('x') coordinate.
        :param i_max: The upper limit for the 'i' ('x') coordinate.
        :param j_min: The lower limit for the 'j' ('y') coordinate.
        :param j_max: The upper limit for the 'j' ('y') coordinate.
        :return: A dictionary with the raster coordinates of adjacent cells and their respective distance to the origin
                cell.
        """
        limits = GridLimits(i_min=i_min, i_max=i_max, j_min=j_min, j_max=j_max)
        adjacent_cells = {cell + target: distance for target, distance in self.distances.items()
                          if cell + target in limits}

        return adjacent_cells
