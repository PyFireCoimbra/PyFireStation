"""
Various classes for reading input data files in legacy format (compatible with FireStation).

"""

from __future__ import annotations

import csv
from typing import Union, Optional, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, PositiveInt, NonNegativeFloat, conlist

from ..utils.custom_types import PathLike
from ..utils.esri_ascii import EsriASCII, base_esri_ascii_parser
from ..utils.io import origin_handle, DataInputMode


class GridData(BaseModel):
    """Validates and structures grid data."""

    data: npt.NDArray
    """Array containing the grid data."""
    cx: List[float]
    """List containing the cell coordinates for the X axis."""
    cy: List[float]
    """List containing the cell coordinates for the Y axis."""

    class Config:
        arbitrary_types_allowed = True

    @property
    def nx(self) -> int:
        """Number of cells in the X axis."""
        return len(self.cx)

    @property
    def ny(self) -> int:
        """Number of cells in the Y axis."""
        return len(self.cy)

    @property
    def shape(self) -> Tuple:
        """The shape of grid (nx x ny)."""
        return self.data.shape

    @classmethod
    def from_EsriASCII(cls, ascii_grid: EsriASCII) -> GridData:
        """Constructs a GridData object from a EsriASCII object."""
        return cls(data=ascii_grid.data, cx=ascii_grid.cx, cy=ascii_grid.cy)


class Control(BaseModel):
    """Validates control file data (stop mode, stop value, and adjacency mode)."""
    mode: int
    value: float
    adj: str


class Ignition(BaseModel):
    """Validates ignition file data (ignition time, and number and raster coordinates of ignition cells)."""
    time: NonNegativeFloat
    ncells: PositiveInt
    cells: conlist(item_type=Tuple[int, int], unique_items=True)    # type: ignore
    # See: https://github.com/pydantic/pydantic/issues/156


def control_parser(origin: Union[str, PathLike],
                   *,
                   mode: Optional[DataInputMode] = None) -> Control:
    """Reads simulation control data from a multi-line string or file.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: Control object with the simulation stop mode, stop value, and adjacency mode.
    """
    with origin_handle(origin, mode=mode) as f:
        reader = csv.reader(f, delimiter=" ")
        data = [v[0] for v in reader if v]

    control_mode, control_value, control_adj = data[:3]

    return Control(mode=control_mode,
                   value=control_value,
                   adj=control_adj)


def ignition_parser(origin: Union[str, PathLike],
                    *,
                    mode: Optional[DataInputMode] = None) -> Ignition:
    """Reads fire ignition data from a multi-line string or file.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: Ignition object with the simulation start time, and the number and location of the ignition cells.
    """

    with origin_handle(origin, mode=mode) as f:
        reader = csv.reader(f, delimiter=" ")
        values = [v for v in reader if v]

    ignition_start = float(values[0][0])
    ignition_ncells = int(values[1][0])
    ignition_cells = [list(filter(len, cell)) for cell in values[2:2 + ignition_ncells]]

    return Ignition(time=ignition_start,
                    ncells=ignition_ncells,
                    cells=ignition_cells)


def fuel_models_parser(origin: Union[str, PathLike],
                       *,
                       mode: Optional[DataInputMode] = None) -> Dict[int, Dict[str, Union[str, float]]]:
    """Reads fuel model data from a multi-line string or file.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: An indexed dictionary of dictionaries each containing the parameters of a fuel model.
    """
    with origin_handle(origin, mode=mode, encoding="iso-8859-1") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        fuels_data = list(reader)

    # Create dict with numbered fuel models
    fuel_param_names = ["name",
                        "oven_dry_fuel_load_dead_1h",
                        "oven_dry_fuel_load_dead_10h",
                        "oven_dry_fuel_load_dead_100h",
                        "oven_dry_fuel_load_alive",
                        "fuel_depth_dead",
                        "fuel_depth_alive",
                        "surface_area_to_volume_ratio_dead_1h",
                        "surface_area_to_volume_ratio_alive",
                        "heat_content_dead",
                        "heat_content_alive",
                        "fuel_moisture_dead_1h",
                        "fuel_moisture_dead_10h",
                        "fuel_moisture_dead_100h",
                        "fuel_moisture_alive",
                        "fuel_moisture_dead_extinction",
                        "fuel_colour",
                        "flame_length",
                        "decaytime",
                        "initial_rhr_factor"]

    # Apply conversions to SI units and relative values to [0, 1]
    conversions = {"surface_area_to_volume_ratio_dead_1h": 100,
                   "surface_area_to_volume_ratio_alive": 100,
                   "fuel_moisture_dead_extinction": 0.01,
                   "fuel_moisture_dead_1h": 0.01,
                   "fuel_moisture_dead_10h": 0.01,
                   "fuel_moisture_dead_100h": 0.01,
                   "fuel_moisture_alive": 0.01}

    # Extract data
    fuels = {}
    n_params = len(fuel_param_names)
    for i, start_line in enumerate(range(1, len(fuels_data), n_params)):
        # Get raw data for fuel
        fuel = fuels_data[start_line:start_line + n_params]

        # Create fuel model dict and add data
        fuel_model: Dict[str, Union[str, float]] = {}

        fuel_name: str = " ".join(fuel[0])
        fuel_model[fuel_param_names[0]] = fuel_name

        fuel_params: List[float] = [float(i[0]) for i in fuel[1:]]
        for param, val in zip(fuel_param_names[1:], fuel_params):
            fuel_model[param] = val*conversions.get(param, 1.0)

        # Add entry to fuels dict
        fuels[i] = fuel_model

    return fuels


def wind_parser(origin: Union[str, PathLike],
                *,
                mode: Optional[DataInputMode] = None) -> Tuple[GridData, GridData]:
    """Reads wind grid data from a multi-line string or file in Esri ASCII format.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: GridData objects of wind speed and wind height data.
    """
    with origin_handle(origin, mode=mode) as f:
        # Ignore first line (metadata)
        _ = f.readline()
        # Read remaining lines. Rows are reversed and grid transposed..
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        grid_data = [[v for v in row if v] for row in reader if row]

    # First line of data is the dimension of the 3D array
    nx, ny, nz = [int(i) for i in grid_data[0] if i]

    # Remaining lines include the coordinates and wind speed of each cell interleaved
    coord = grid_data[1:nx * ny * nz * 2 + 1:2]
    wind = grid_data[2:nx * ny * nz * 2 + 1:2]

    # Extract X and Y axis coordinates
    cx = [i[0] for i in coord[::ny * nz]]
    cy = [i[1] for i in coord[:ny * nz:nz]]

    # Extract Z coordinates (cell dependent)
    wind_height_3d = np.array([[[coord[i * ny * nz + j * nz + k][2]
                                 for k in range(nz)]
                                for j in range(ny)]
                               for i in range(nx)],
                              dtype=np.float_)[:, :, 1:3]

    wind_height_data = np.diff(wind_height_3d, axis=2)[:, :, 0]

    # Extract 3D wind speed
    wind_layer = 1
    wind_speed_data = np.array([[[wind[i * ny * nz + j * nz + k]
                                  for k in range(nz)]
                                 for j in range(ny)]
                                for i in range(nx)],
                               dtype=np.float_)[:, :, wind_layer, :]

    wind_speed = GridData(data=wind_speed_data,
                          cx=cx,
                          cy=cy)

    wind_height = GridData(data=wind_height_data,
                           cx=cx,
                           cy=cy)

    return wind_speed, wind_height


def fuel_dist_parser(origin: Union[str, PathLike],
                     *,
                     mode: Optional[DataInputMode] = None) -> GridData:
    """Reads fuel distribution grid from a multi-line string or file.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: GridData object of the fuel model distribution data.
    """
    ascii_grid = base_esri_ascii_parser(origin, mode=mode, dtype=np.int_)
    return GridData.from_EsriASCII(ascii_grid)


def terrain_height_parser(origin: Union[str, PathLike],
                          *,
                          mode: Optional[DataInputMode] = None) -> GridData:
    """Reads terrain height (elevation) grid from a multi-line string or file.

    :param origin: The data source, either a path to a file or a string containing the read data.
    :param mode: The read mode (file or string) to be used with the provided origin.
    :return: GridData object of the terrain elevation (height) data.
    """
    ascii_grid = base_esri_ascii_parser(origin, mode=mode, dtype=np.float_)
    return GridData.from_EsriASCII(ascii_grid)
