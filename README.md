# PyFireStation

## About

Fire propagation simulation based on the Rothermel equations, following closely the implementation of [FireStation -
Lopes et. al (2002)](https://doi.org/10.1016/S1364-8152%2801%2900072-X). The purpose of this package is to provide an 
easy and modern framework for testing and developing fire propagation and fuel parameter models, and to support future scientific research in these fields.

## Requirements

This project requires `Python >= 3.8` and `pip`. The installation process is automated and all the dependencies are
installed if not yet available.

## Installation

### (Recommended) Creating a virtual environment

It is always recommended that you use a virtual environment. First create it:

```
python -m venv venv

# or

python3 -m venv venv

```

And finally activate it:

```
# Unix
source ./venv/bin/activate

# or

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

After the environment is set up and activated, this package can then be easily installed. Anytime you wish to use this
package, you should activate the respective environment.

### Using `pip`

First clone the repository:


```
git clone https://github.com/PyFireCoimbra/PyFireStation.git
```

And then install the package using `pip`:

```
python -m pip install -e PyFireStation

# or 

python3 -m pip install -e PyFireStation
```

## Usage

### Propagation simulation

After installation, the fire propagation simulation can be run by simply calling `pyfirestation`. By default, this
package expects the input files to be in the FireStation-compatible format, placed in a folder named `FireInput` in
current working directory. One can also specify the location of the input files by passing their path as arguments.
Run `pyfirestation -h` for more information.

```
$ pyfirestation -h

usage: pyfirestation [-h] [-c PATH] [-i PATH] [-w PATH] [-t PATH] [-f PATH] [-m PATH] [-d PATH] [-o PATH]

PyFireStation fire propagation simulation.

options:
  -h, --help            show this help message and exit

Files:
  Path to data files. Can be relative (with respect to `--work-dir`) or absolute. 
  For absolute paths, the `--work-dir` argument is ignored.

  -c PATH, --control PATH
                        Propagation simulation control parameters.
  -i PATH, --ignition PATH
                        Ignition cells for propagation simulation.
  -w PATH, --wind PATH  Nuatmos Input/Output file.
  -t PATH, --terrain PATH
                        Terrain height ASCII raster.
  -f PATH, --fuel-distr PATH
                        Fuel distribution ASCII raster.
  -m PATH, --fuel-models PATH
                        Fuel models file.

General:
  General simulation parameters.

  -d PATH, --work-dir PATH
                        Working directory.
  -o PATH, --output PATH
                        Output path of file with results.
```

### Using the package and building custom applications

The provided package and subpackages can be used for building custom applications.
After installation, one can use the usual Python import system:

#### Example #1: Accessing fuel model properties

```python
from pyfirestation.rothermel import FuelModel

fuel_model = FuelModel(heat_content_dead=22700.0,
                       heat_content_alive=22700.0,
                       oven_dry_fuel_load_alive=0.6,
                       oven_dry_fuel_load_dead_1h=0.1,
                       oven_dry_fuel_load_dead_10h=0.1,
                       oven_dry_fuel_load_dead_100h=0.1,
                       fuel_depth_alive=1.4,
                       fuel_depth_dead=1.4,
                       fuel_moisture_alive=0.6,
                       fuel_moisture_dead_extinction=0.4,
                       fuel_moisture_dead_1h=0.1,
                       fuel_moisture_dead_10h=0.1,
                       fuel_moisture_dead_100h=0.1,
                       flame_length=4.0,
                       surface_area_to_volume_ratio_alive=6000.0,
                       surface_area_to_volume_ratio_dead_1h=6000.0)

print(fuel_model.rate_of_spread_o)
```

#### Example #2: Creating a SimulationGrid object from Numpy arrays

```python
import numpy as np

from pyfirestation import grids

# Read the files containing the X and Y coordinates for the grid
cx = np.read("cx.npy")  # Size M
cy = np.read("cy.npy")  # Size N

# Read the arrays containing the grid data 
terrain_array = np.read("terrain.npy")
wind_speed_array = np.read("wind_speed.npy")
wind_height_array = np.read("wind_speed.npy")
fuel_array = np.read("fuel.npy")

# Create the data grids
fuel_grid = grids.FuelGrid(terrain_array, cx=cx, cy=cy)
terrain_grid = grids.TerrainGrid(terrain_array, cx=cx, cy=cy)
wind_grid = grids.WindGrid(wind_speed_array, wind_height_array, cx=cx, cy=cy)

# Create the simulation grid
simulation_grid = grids.SimulationGrid(fuel_grid=fuel_grid,
                                       terrain_grid=terrain_grid,
                                       wind_grid=wind_grid)

```

## Authors

* João Aveiro ([@joao-aveiro](https://github.com/joao-aveiro))

* Daniel Neves ([@danielneves1](https://github.com/danielneves1))

* Jaime Silva (jaime.silva@fis.uc.pt)

## License

**pyfirestation** is available under the **GNU GPLv3** license. 
See the [LICENSE](https://github.com/PyFireCoimbra/PyFireStation/blob/main/LICENSE)  for more info.
