"""
Main classes for the fire propagation simulation.

"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Iterable, Union
from typing import List

import numpy as np

from .adjacency import CellAdjacency, CellAdjacencyMode
from .fire_ellipse import FireEllipse
from .utils import PropagationLog, PropagationPath
from .writer import CLIWriter
from ..grids.simulation_grid import SimulationGrid
from ..rothermel.fuel_model import Spread
from ..utils.cellindex import CellIndex


class SimulationStopMode(Enum):
    """Defines the simulation stop modes available.

    The fire propagation simulation is an iterative process that is stopped when one of the following conditions,
    specified by the user:

    1. The area burnt (AREA, 0) in T;
    2. The number of cells burnt (CELLS, 1);
    3. The time of the fire propagation (TIME, 2);

    reaches the stop value.
    """
    AREA = 0
    CELLS = 1
    TIME = 2


class Propagation:
    def __init__(self,
                 grid: SimulationGrid,
                 cell_adjacency_mode: Union[int, str, CellAdjacencyMode],
                 stop_mode: Union[int, SimulationStopMode],
                 stop_value: Union[int, float],
                 *,
                 propagation_log: PropagationLog = PropagationLog(),
                 writer=CLIWriter()) -> None:
        """

        :param grid: SimulationGrid object including all data grids (wind, fuel, terrain).
        :param cell_adjacency_mode:
        :param stop_mode:
        :param stop_value:
        """

        # Note: Setters with different types then their properties is not currently supported with mypy
        # (see: https://github.com/python/mypy/issues/3004). Type checking for cell_adjacency mode and stop_mode is
        # bypassed below.

        # Simulation data and parameters
        self.grid = grid

        self._cell_adjacency_mode: CellAdjacencyMode
        self.cell_adjacency_mode = cell_adjacency_mode  # type: ignore

        self._cell_adjacency: CellAdjacency

        self._stop_mode: SimulationStopMode
        self.stop_mode = stop_mode  # type: ignore

        self._stop_value: Union[int, float]
        self.stop_value = stop_value

        self._force_stop: bool = False  # Flag to force the simulation to stop on the next iteration

        # Dictionaries to hold grid cells
        self.raw_cells: Dict[CellIndex, float]  # Not burnt yet
        self.lit_cells: Dict[CellIndex, float]  # On fire
        self.brt_cells: Dict[CellIndex, float]  # Burnt

        # Time of propagation
        self.t: float

        # Utils
        self.propagation_log = propagation_log
        self.writer = writer

    @property
    def cell_adjacency_mode(self) -> CellAdjacencyMode:
        """Cell adjacency mode that defines the number and location of neighbours of a cell."""
        return self._cell_adjacency_mode

    @cell_adjacency_mode.setter
    def cell_adjacency_mode(self, mode: Union[int, str, CellAdjacencyMode]) -> None:
        self._cell_adjacency_mode = CellAdjacencyMode.get_mode(mode)
        self._cell_adjacency = CellAdjacency(mode=mode, cell_size=self.grid.cellsize)

    @property
    def cell_adjacency(self) -> CellAdjacency:
        """CellAdjacency object that provides the distribution and distances from a cell to neighbour cells."""
        return self._cell_adjacency

    @property
    def stop_mode(self) -> SimulationStopMode:
        """Simulation stop mode. See SimulationStopMode for more information."""
        return self._stop_mode

    @stop_mode.setter
    def stop_mode(self, mode: Union[int, float, str, SimulationStopMode]) -> None:
        if isinstance(mode, SimulationStopMode):
            self._stop_mode = mode
        elif isinstance(mode, (int, float, str)):
            self._stop_mode = SimulationStopMode(int(mode))
        else:
            raise TypeError(f"Stop mode should be of type SimulationStopMode or int (\'{mode}\' provided).")

    @property
    def stop_value(self) -> Union[int, float]:
        """Simulation stop value. See SimulationStopMode for more information."""
        return self._stop_value

    @stop_value.setter
    def stop_value(self, value: Union[int, float]) -> None:
        if isinstance(value, (float, int)):
            self._stop_value = value
        else:
            raise TypeError(f"Stop value should be of type [int, float] (\'{value}\' provided).")

    @property
    def stop(self) -> bool:
        """Defines if the simulation should be stopped.

        :return: Bool value for whether the simulation should be stopped (True) or not (False).
        """
        # Not very pretty, but should be somewhat fast
        if len(self.raw_cells) == 0 or self._force_stop:
            self._force_stop = False
            return True
        elif self.stop_mode is SimulationStopMode.AREA:  # Burnt area (ha)
            if (len(self.brt_cells) + len(self.lit_cells)) * self.grid.cellsize ** 2 / 10000 >= self.stop_value:
                return True
        elif self.stop_mode is SimulationStopMode.CELLS:  # Number of cells
            if len(self.brt_cells) + len(self.lit_cells) >= self.stop_value:
                return True
        elif self.stop_mode is SimulationStopMode.TIME:  # Propagation time
            if self.t > self.stop_value:
                return True

        return False

    def start_burning(self, cell: CellIndex) -> float:
        """Removes cell from raw cells dictionary and places it in the burning cells' dictionary.

        :param cell: CellIndex defining the cell to be moved.
        :return: The expected time for the ignition of the cell.
        """
        ig_time = self.raw_cells.pop(cell)
        self.lit_cells.update({cell: ig_time})
        return ig_time

    def stop_burning(self, cell: CellIndex) -> float:
        """Removes cell from burning cells dictionary and places it in the burnt cells' dictionary.

        :param cell: CellIndex defining the cell to be moved.
        :return: The expected time for the ignition of the cell.
        """
        params = self.lit_cells.pop(cell)
        self.brt_cells.update({cell: params})
        return params

    def find_next(self) -> CellIndex:
        """Finds next cell to ignite based on the simulation ignition time prediction (minimum time to ignition).

        :return: The next cell that will start burning.
        """
        return min(self.raw_cells, key=self.raw_cells.__getitem__)

    def get_available_adjacent_cells(self, cell: CellIndex) -> Dict[CellIndex, float]:
        """Retrieves the available adjacent cells of a burning cell that are candidates to start burning.

        :param cell: The center cell for which the adjacent cells are obtained.
        :return: A dictionary containing the adjacent cells within the grid limits and their distance.
        """
        # Get adjacent cells within the grid limits
        adjacent_cells = self.cell_adjacency.adjacent_cells(cell,
                                                            i_max=self.grid.nx,
                                                            j_max=self.grid.ny)

        # If not enough cells were returned (i.e. we have reached the limits of the grid), stop in next iteration
        if len(adjacent_cells) != self.cell_adjacency.mode:
            self._force_stop = True

        # Filter out cells that are already burning/burnt
        adjacent_cells = {k: v for k, v in adjacent_cells.items()
                          if k not in self.lit_cells and k not in self.brt_cells}
        return adjacent_cells

    def calc_cell_propagation(self, source_cell: CellIndex) -> List[PropagationPath]:
        """Calculates the propagation time from one burning cell to non-burning adjacent cells.

        :param source_cell: The burning cell from which the fire propagates.
        :return: A list of PropagationPath objects defining a source and target cell and the respective time of fire
        propagation between them.
        """
        source_fuel_model = self.grid.fuel_grid[source_cell]
        spread = Spread(
            fuel=source_fuel_model,
            terrain=self.grid.terrain_grid[source_cell],
            wind=self.grid.wind_grid[source_cell],
        )

        rc = source_fuel_model.rate_of_spread_o

        height_source = self.grid.terrain_grid[source_cell].height_center
        flux = np.linalg.norm(spread.total_flux)
        wind_speed = spread.equivalent_wind
        flux_x, flux_y = spread.total_flux
        flux_direction = np.arctan2(flux_y, flux_x) % (2 * np.pi)

        ellipse = FireEllipse(wind_speed=wind_speed, theta=flux_direction)

        propagations = []

        adjacent_cells = self.get_available_adjacent_cells(source_cell)
        for target_cell, plane_dist in adjacent_cells.items():
            target_fuel_model = self.grid.fuel_grid[target_cell]
            rn = target_fuel_model.rate_of_spread_o
            if rc == 0 and rn == 0:
                r_harm = 0.0
            else:
                r_harm = 2 * rc * rn / (rc + rn)

            rate_of_spread = r_harm * (1 + flux)
            ellipse_r = ellipse.radius(origin=source_cell, target=target_cell)

            if rate_of_spread != 0 and ellipse_r != 0:
                height_target = self.grid.terrain_grid[target_cell].height_center
                cell_dist = np.sqrt(plane_dist**2 + (height_target - height_source)**2)
                time_delta = cell_dist / (rate_of_spread * ellipse_r)
            else:
                time_delta = float("inf")

            propagations.append(PropagationPath(source=source_cell,
                                                target=target_cell,
                                                time=time_delta)
                                )

        return propagations

    def update_neighbours(self, source_cell: CellIndex) -> None:
        """Calculates the propagation times between a burning cell and adjacent non-burning cells, updating the
        minimum time of arrival for each cell.

        :param source_cell: The source cell from which the time of propagation to adjacent cells is calculated.
        """
        time_ign_source_cell = self.lit_cells[source_cell]

        propagations = self.calc_cell_propagation(source_cell)

        for propagation in propagations:
            time_delta = propagation.time
            if time_delta != float("inf"):
                target_cell = propagation.target
                time_old = self.raw_cells.get(target_cell, float("inf"))
                time_new = time_delta + time_ign_source_cell

                self.raw_cells[target_cell] = min(time_old, time_new)

    def reset_run(self) -> None:
        """Resets the simulation data structures for a new run, namely the cell dictionaries and the simulation time."""
        # Dictionaries to hold grid cells
        self.raw_cells = {}
        self.lit_cells = {}
        self.brt_cells = {}

        # Time of propagation
        self.t = 0

        # Reset log
        self.propagation_log.reset()

    def run(self,
            ignition: Union[CellIndex, Iterable[CellIndex]],
            *,
            start_time: float = 0) -> None:  # TODO: Fix type hint
        """Runs the propagation simulation.

        :param ignition: List or Tuple of ignition cells from which the fire starts to spread.
        :param start_time: The starting time of the simulation at which the provided ignition cells start burning.
        """

        if isinstance(ignition, CellIndex):
            ignition = (ignition,)

        self.reset_run()
        self.writer.write_pre_hook(log=self.propagation_log, grid=self.grid)
        self.t = start_time

        # TODO: Check if cell indices are within grid. Maybe add method to SimulationGrid?

        # Add ignition nodes
        for cell in ignition:
            if not isinstance(cell, CellIndex):
                cell = CellIndex(i=cell[0], j=cell[1])
            cell = CellIndex(i=cell.i - 1, j=cell.j - 1)  # TODO: TEMP, REMOVE. EDIT: Error when this is removed...
            self.raw_cells.update({cell: self.t})

        while not self.stop:
            # Set new node on fire and update time
            active = self.find_next()
            self.t = self.start_burning(active)
            self.update_neighbours(active)
            self.propagation_log.add_entry(cell=active, time=self.t)

            self.writer.write_step(log=self.propagation_log, grid=self.grid)
        self.writer.write_post_hook(log=self.propagation_log, grid=self.grid)
