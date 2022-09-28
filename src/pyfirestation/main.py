"""
Propagation simulation entrypoint compatible with legacy FireStation input files.

"""

from .grids import FuelGrid, TerrainGrid, WindGrid, SimulationGrid
from .legacy import (LegacyArgumentParser,
                     terrain_height_parser,
                     fuel_models_parser,
                     fuel_dist_parser,
                     wind_parser,
                     control_parser,
                     ignition_parser)
from .propagation import CLIWriter, Propagation
from .rothermel import FuelModel
from .utils import calc_simple_grid_nearest_interpolation, calc_vector_grid_bilinear_interpolation


def main() -> None:
    arg_parser = LegacyArgumentParser()
    files = arg_parser.files

    # Read data from files
    print("Reading terrain data...")
    terrain_grid_data = terrain_height_parser(files['terrain'])
    print("Reading fuel model data...")
    fuel_models_data = fuel_models_parser(files["fuel_models"])
    print("Reading fuel grid data...")
    fuel_grid_data = fuel_dist_parser(files['fuel_distr'])
    print("Reading wind data...")
    wind_speed_data, wind_height_data = wind_parser(files['wind'])
    print("Reading simulation control data...")
    control_data = control_parser(files["control"])
    print("Reading ignition data...")
    ignition_data = ignition_parser(files["ignition"])

    # Create data objects
    print("Creating fuel grid...")
    fuel_models = {model_idx: FuelModel.from_dict(model_params)
                   for model_idx, model_params in fuel_models_data.items()}
    fuel_grid = FuelGrid(index_array=fuel_grid_data.data,
                         cx=fuel_grid_data.cx,
                         cy=fuel_grid_data.cy,
                         fuel_models=fuel_models)

    print("Creating terrain grid...")
    terrain_grid = TerrainGrid(height_array=terrain_grid_data.data,
                               cx=terrain_grid_data.cx,
                               cy=terrain_grid_data.cy)

    print("Creating wind grid...")
    wind_speed_data_interp = calc_vector_grid_bilinear_interpolation(wind_speed_data.data,
                                                                     source_coor_x=wind_speed_data.cx,
                                                                     source_coor_y=wind_speed_data.cy,
                                                                     target_coor_x=terrain_grid_data.cx,
                                                                     target_coor_y=terrain_grid_data.cy)

    wind_height_data_interp = calc_simple_grid_nearest_interpolation(wind_height_data.data,
                                                                     source_coor_x=wind_height_data.cx,
                                                                     source_coor_y=wind_height_data.cy,
                                                                     target_coor_x=terrain_grid_data.cx,
                                                                     target_coor_y=terrain_grid_data.cy)

    wind_grid = WindGrid(speed_array=wind_speed_data_interp,
                         height_array=wind_height_data_interp,
                         cx=terrain_grid_data.cx,
                         cy=terrain_grid_data.cy)

    print("Preparing simulation...")
    simulation_grid = SimulationGrid(fuel_grid=fuel_grid,
                                     terrain_grid=terrain_grid,
                                     wind_grid=wind_grid)

    ignition_cells = simulation_grid.get_cell_indices(ignition_data.cells)

    # Create propagation object and run simulation
    p = Propagation(grid=simulation_grid,
                    stop_mode=control_data.mode,
                    stop_value=control_data.value,
                    cell_adjacency_mode=control_data.adj,
                    writer=CLIWriter())

    p.run(ignition=ignition_cells)


if __name__ == '__main__':
    main()
