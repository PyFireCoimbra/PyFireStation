"""
Custom data types for helping with type hints.

"""

import pathlib
from abc import ABC
from typing import TypeVar


PathLike = TypeVar("PathLike", pathlib.Path, str)


class TerrainCellLike(ABC):
    slope: float
    slope_dir: float


class WindCellLike(ABC):
    vel_x: float
    vel_y: float
    vel_z: float
    height: float
