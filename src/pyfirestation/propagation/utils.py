"""
Various utility classes used for the propagation simulation.

"""

from collections import UserList
from typing import NamedTuple, Optional, Union

from ..utils.cellindex import CellIndex


class FireCell(NamedTuple):
    """Includes a cell and the respective"""
    cell: CellIndex
    time: float


class PropagationPath(NamedTuple):
    """Defines a fire spread path between a source and a target cells, and the respective time of propagation."""
    source: CellIndex
    target: CellIndex
    time: float


class InvalidPropagationLogEntryError(Exception):
    """Exception class for invalid propagation log entries."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PropagationLog(UserList):
    def reset(self) -> None:    # TODO: Check if this works
        self.data = []

    def add_entry(self,
                  cell: Optional[Union[CellIndex, FireCell]] = None,
                  *,
                  time: Optional[float] = None,
                  i: Optional[int] = None,
                  j: Optional[int] = None) -> None:
        if all(arg is None for arg in [cell, time, i, j]):
            raise ValueError("No values provided for log entry.")
        elif isinstance(cell, FireCell):
            new_entry = cell
        elif isinstance(cell, CellIndex) and time is not None:
            new_entry = FireCell(cell=cell, time=time)
        elif isinstance(i, (int, float)) and isinstance(j, (int, float)) and isinstance(time, (int, float)):
            new_entry = FireCell(cell=CellIndex(int(i), int(j)), time=float(time))
        else:
            raise InvalidPropagationLogEntryError(f"Could not create log entry with provided values.\n")

        self.data.append(new_entry)

    def get_last_entry(self):
        return self.data[-1]
