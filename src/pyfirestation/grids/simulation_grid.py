"""
Grid structure containing all the spatial data required for the fire spread simulation.

"""

from itertools import combinations
from typing import Tuple, Iterable, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, root_validator

from .fuel_grid import FuelGrid
from .terrain_grid import TerrainGrid
from .wind_grid import WindGrid
from ..utils.cellindex import CellIndex


class GridShapeMismatchError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GridCoordinatesMismatchError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class SimulationGrid(BaseModel):
    fuel_grid: FuelGrid
    terrain_grid: TerrainGrid
    wind_grid: WindGrid

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        allow_mutation = False

    @root_validator(pre=True)
    def valid_grids_types(cls, values):
        grids_types = {"fuel_grid":  FuelGrid,
                       "terrain_grid": TerrainGrid,
                       "wind_grid": WindGrid}

        for grid_obj_name, expected_type in grids_types.items():
            grid_obj = values.get(grid_obj_name)
            if not isinstance(grid_obj, expected_type):
                raise TypeError(f"\'{grid_obj_name}\' should be of type {expected_type}. "
                                f"\'{type(grid_obj)}\' received.")

        return values

    # TODO: Validate this
    @root_validator(skip_on_failure=True)
    def grid_shapes_match(cls, values):
        grids = ("fuel_grid", "terrain_grid", "wind_grid")

        shapes = [values.get(grid).shape for grid in grids]
        shapes_pairs = list(combinations(shapes, 2))
        for shape1, shape2 in shapes_pairs:
            if not shape1 == shape2:
                raise GridShapeMismatchError("The shapes of the grids provided don't match.")
        return values

    # TODO: Validate this
    @root_validator(skip_on_failure=True)
    def grid_coordinates_match(cls, values):
        grids = (values.get("fuel_grid"),
                 values.get("terrain_grid"),
                 values.get("wind_grid"))

        x_coors = [grid.cx for grid in grids]
        y_coors = [grid.cy for grid in grids]
        for coors in [x_coors, y_coors]:
            # Check for all pair combinations as to avoid error accumulation if done sequentially
            coors_pairs = combinations(coors, 2)
            for coor1, coor2 in coors_pairs:
                if not np.allclose(coor1, coor2, atol=0, rtol=1E-8):
                    raise GridCoordinatesMismatchError("The coordinates of the grids provided don't match.")
        return values

    # # Grid component getters
    # def get_fuel_grid(self) -> FuelGrid:
    #     return self._fuel_grid
    #
    # def get_terrain_grid(self) -> TerrainGrid:
    #     return self._terrain_grid
    #
    # def get_wind_grid(self) -> WindGrid:
    #     return self._wind_grid

    # # Grid component aliases
    # fuel_grid = property(get_fuel_grid)
    # fuel = property(get_fuel_grid)
    #
    # terrain_grid = property(get_terrain_grid)
    # terrain = property(get_terrain_grid)
    #
    # wind_grid = property(get_wind_grid)
    # wind = property(get_wind_grid)

    # Grid properties proxy
    # @property
    # def shape(self) -> Tuple:
    #     return self.get_terrain_grid().shape
    #
    # @property
    # def cx(self) -> npt.NDArray:
    #     return self.get_terrain_grid().cx
    #
    # @property
    # def cy(self) -> Tuple:
    #     return self.get_terrain_grid().shape
    #
    # @property
    # def cellsize(self) -> float:
    #     return self.terrain_grid.cellsize
    #
    @property
    def shape(self) -> Tuple:
        return self.terrain_grid.shape

    @property
    def cx(self) -> npt.NDArray:
        return self.terrain_grid.cx

    @property
    def cy(self) -> npt.NDArray:
        return self.terrain_grid.cy

    @property
    def cellsize(self) -> float:
        return self.terrain_grid.cellsize

    @property
    def nx(self) -> int:
        return len(self.cx)

    @property
    def ny(self) -> int:
        return len(self.cy)

    def get_cell_indices(self, cells: Iterable) -> Union[CellIndex, Tuple[CellIndex, ...]]:
        return self.terrain_grid.get_cell_indices(cells)

