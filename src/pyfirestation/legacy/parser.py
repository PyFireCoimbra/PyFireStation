"""
Custom ArgumentParser class for working with legacy FireStation input types.

"""

import os
from argparse import ArgumentParser
from typing import Dict


class LegacyArgumentParser(ArgumentParser):
    """Custom ArgumentParser class for working with legacy FireStation input types."""

    def __init__(self, *args, **kwargs):
        # Parent init
        description = """PyFireStation fire propagation simulation.
        """
        super().__init__(description=description, *args, **kwargs)

        # File group
        file_group_description = """
        Path to data files. Can be relative (with respect to `--work-dir`) or absolute. For absolute paths, the 
        `--work-dir` argument is ignored.
        """
        file_group = self.add_argument_group("Files", description=file_group_description)
        file_group.add_argument("-c", "--control", metavar="PATH",
                                default="Control.dat",
                                help="Propagation simulation control parameters.")
        file_group.add_argument("-i", "--ignition", metavar="PATH",
                                default="ignition.dat",
                                help="Ignition cells for propagation simulation.")
        file_group.add_argument("-w", "--wind", metavar="PATH",
                                default="nuatmos.out",
                                help="Nuatmos Input/Output file.")
        file_group.add_argument("-t", "--terrain", metavar="PATH",
                                default="terrain.asc",
                                help="Terrain height ASCII raster.")
        file_group.add_argument("-f", "--fuel-distr", metavar="PATH",
                                default="fueldistr.asc",
                                help="Fuel distribution ASCII raster.")
        file_group.add_argument("-m", "--fuel-models", metavar="PATH",
                                default="fuelmodels.fls",
                                help="Fuel models file.")

        # General group
        general_group_description = """General simulation parameters.
        """
        general_group = self.add_argument_group("General", description=general_group_description)
        general_group.add_argument("-d", "--work-dir", metavar="PATH",
                                   default="FireInput",
                                   help="Working directory.")
        general_group.add_argument("-o", "--output", metavar="PATH",
                                   default="run.csv",
                                   help="Output path of file with results.")

    @staticmethod
    def add_workdir(path: str, workdir: str) -> str:
        """Joins a basename (workdir) to a basename if provided path is not an absolute path."""
        if not os.path.isabs(path):
            base = os.path.basename(path)
            path = os.path.join(workdir, base)
        return path

    @property
    def _args(self) -> Dict:
        """Mapping og the parsed arguments."""
        return vars(self.parse_args())

    @property
    def workdir(self) -> str:
        """The working directory where the files are stored"""
        return str(self._args.get("work_dir"))

    @property
    def files(self) -> Dict[str, str]:
        """Dictionary with file ID and respective path."""
        return {k: self.add_workdir(v, self.workdir) for k, v in self._args.items()}
