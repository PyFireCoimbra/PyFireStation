"""
Fuel grid structure for the distribution of types of fuel models.

"""

from typing import Iterable, Mapping

import numpy.typing as npt
import xarray as xr

from .base_grid import BaseGrid
from ..rothermel.fuel_model import FuelModel


class FuelGrid(BaseGrid):
    __slots__ = ["_dataset", "_fuel_models"]

    def __init__(self,
                 index_array: npt.ArrayLike,
                 cx: npt.ArrayLike,
                 cy: npt.ArrayLike,
                 fuel_models: Mapping[int, FuelModel]) -> None:

        self._fuel_models = fuel_models
        self._dataset = xr.Dataset(
            data_vars=dict(
                fuel_id=(["x", "y"], index_array)
            ),
            coords=dict(
                x=cx,
                y=cy,
            ),
            attrs=dict(
                description="Fuel grid.",
                fuel_models=fuel_models
            )
        )

    @property
    def fuel_models(self) -> Mapping[int, FuelModel]:
        """A dictionary (or mapping) containing the available fuel models."""
        return self._fuel_models

    def __getitem__(self, cell: Iterable) -> FuelModel:
        cell_data = super().__getitem__(cell)
        fuel_id = int(cell_data["fuel_id"])
        # print(fuel_id)
        return self.fuel_models[fuel_id]
