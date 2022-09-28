"""
ABC and a few interfaces for writing the status and results of Propagation simulations.

"""

import csv
import os.path
from abc import ABC, abstractmethod

from .utils import PropagationLog
from ..grids.simulation_grid import SimulationGrid


class BasePropagationWriter(ABC):
    """ABC for interfacing with writers of Propagation simulation results and status."""

    @abstractmethod
    def write_pre_hook(self,
                       log: PropagationLog,
                       grid: SimulationGrid,
                       *args, **kwargs) -> None:
        ...

    @abstractmethod
    def write_step(self,
                   log: PropagationLog,
                   grid: SimulationGrid,
                   *args, **kwargs) -> None:
        ...

    @abstractmethod
    def write_post_hook(self,
                        log: PropagationLog,
                        grid: SimulationGrid,
                        *args, **kwargs) -> None:
        ...


class CLIWriter(BasePropagationWriter):
    """Interface class for writing Propagation simulation results to .csv file."""

    def __init__(self, step: int = 100) -> None:
        self._step = step    # How many results to hold before printing
        self._counter = 0
        self._last_print = 0

    def write_pre_hook(self,
                       log: PropagationLog,
                       grid: SimulationGrid,
                       *args, **kwargs) -> None:
        # Reset
        self._counter = 0
        self._last_print = 0
        print("\nStarting propagation simulation...")
        print("#"*30)
        print("i, j, time")
        print("__________")

    def write_step(self,
                   log: PropagationLog,
                   grid: SimulationGrid,
                   *args, **kwargs) -> None:
        # If enough lines have been stored (=step), print to console
        if self._counter - self._last_print >= self._step:
            for entry in log[-(self._counter - self._last_print):]:
                print(entry.cell.i, entry.cell.j, entry.time)
            self._last_print = self._counter
        else:
            self._counter += 1

    def write_post_hook(self,
                        log: PropagationLog,
                        grid: SimulationGrid,
                        *args, **kwargs) -> None:
        # Print remaining lines
        if self._last_print != self._counter:
            for entry in log[-(self._counter - self._last_print):]:
                print(entry.cell.i, entry.cell.j, entry.time)
            self._last_print = self._counter
        print("#" * 30)
        print("\nFinished!\n")


class CSVWriter(BasePropagationWriter):
    """Interface class for writing Propagation simulation results to .csv file."""

    def __init__(self, path: str = "out.csv") -> None:
        self.path = os.path.normpath(path)

    @staticmethod
    def write_csv(path: str, log: PropagationLog) -> None:
        """Writes the data in a PropagationLog to a .csv file."""
        with open(path, "w", newline="") as f:
            wr = csv.writer(f, delimiter=",")
            wr.writerow(("i", "j", "time"))
            for entry in log:
                wr.writerow((entry.cell.i, entry.cell.j, entry.time))

    def write_pre_hook(self,
                       log: PropagationLog,
                       grid: SimulationGrid,
                       *args, **kwargs) -> None:
        print("\nStarting propagation simulation...")

    def write_step(self,
                   log: PropagationLog,
                   grid: SimulationGrid,
                   *args, **kwargs) -> None:
        pass

    def write_post_hook(self,
                        log: PropagationLog,
                        grid: SimulationGrid,
                        *args, **kwargs) -> None:
        print("\nFinished! \n\t- Writing output...")
        self.write_csv(path=self.path, log=log)
        print(f"\t- Output written to: {self.path}")
