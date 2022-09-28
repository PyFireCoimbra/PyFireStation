"""
Custom class for working with Esri ASCII grid and related functions.

"""

import csv
from typing import Union, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, PositiveInt, PositiveFloat, validator

from .custom_types import PathLike
from .io import DataInputMode, origin_handle


class EsriASCII(BaseModel):
    """Validates grid data provided by Esri ASCII files."""
    ncols: PositiveInt
    nrows: PositiveInt
    xllcorner: float
    yllcorner: float
    xllcenter: Optional[float] = None
    yllcenter: Optional[float] = None
    cellsize: PositiveFloat
    nodata_value: Optional[Union[int, str, float]] = None
    data: npt.NDArray

    class Config:
        arbitrary_types_allowed = True

    @validator("xllcenter", "yllcenter")
    def center_origin_provided(cls, v):
        if v:
            raise NotImplementedError(
                "Currently only \'xllcorner\' and \'yllcorner\' origin modes are supported for Esri ASCII grid.")

    @property
    def cx(self):
        """X grid coordinates."""
        start = self.xllcorner
        stop = start + self.cellsize * int(self.ncols)
        return list(np.linspace(start=start, stop=stop, num=self.ncols, endpoint=False))

    @property
    def cy(self):
        """Y grid coordinates."""
        start = self.yllcorner
        stop = start + self.cellsize * int(self.nrows)
        return list(np.linspace(start=start, stop=stop, num=self.nrows, endpoint=False))


def base_esri_ascii_parser(origin: Union[str, PathLike],
                           *,
                           mode: Optional[DataInputMode] = None,
                           dtype: Optional[npt.DTypeLike] = None) -> EsriASCII:
    """Reads grid data from a multi-line string or file in Esri ASCII format."""
    # Extract data
    with origin_handle(origin, mode=mode) as f:
        raw_header = [f.readline().strip().split() for _ in range(6)]
        reader = csv.reader(f, delimiter=" ")
        raw_grid_data = list(reader)

    header = {str(line[0]).strip().lower(): line[1]
              for line in raw_header}

    grid_data = np.array([[v for v in row if v]
                          for row in raw_grid_data],
                         dtype=dtype)

    # Flip up/down and transpose
    grid_data = grid_data[::-1, :].T

    return EsriASCII(
        ncols=header.get("ncols"),
        nrows=header.get("nrows"),
        xllcorner=header.get("xllcorner"),
        xllcenter=header.get("xllcenter"),
        yllcorner=header.get("yllcorner"),
        yllcenter=header.get("yllcenter"),
        cellsize=header.get("cellsize"),
        nodata_value=header.get("nodata_value"),
        data=grid_data
        )
