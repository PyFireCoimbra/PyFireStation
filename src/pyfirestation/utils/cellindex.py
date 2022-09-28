"""
Defines the ClassIndex object, which is used throughout this codebase to define a cell by its raster coordinates.

"""

from __future__ import annotations

from typing import NamedTuple


class CellIndex(NamedTuple):
    """Defines the raster coordinates of a cell."""
    i: int
    j: int

    def __add__(self, other: object) -> CellIndex:
        if not isinstance(other, CellIndex):
            return NotImplemented
        return CellIndex(self.i + other.i, self.j + other.j)

    def __sub__(self, other: object) -> CellIndex:
        if not isinstance(other, CellIndex):
            return NotImplemented
        return CellIndex(self.i - other.i, self.j - other.j)
