"""
Interpolation functions for raster data grids.

"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import interpolate  # type: ignore


def center_coordinates(coordinate_axis: npt.ArrayLike) -> npt.NDArray:
    """Shifts an array of coordinates of an axis from the corner to the center of the cells.

    :param coordinate_axis: An array containing the one axis coordinates of the corners of the grid cells.
    :return: The array of axis coordinates with respect to the center of the grid cells.
    """
    coordinate_axis = np.array(coordinate_axis, copy=True)
    dx = np.ediff1d(coordinate_axis, to_end=0)
    dx[-1] = dx[-2]
    return coordinate_axis + (dx / 2)


def calc_vector_grid_bilinear_interpolation(array: npt.ArrayLike,
                                            *,
                                            source_coor_x: npt.ArrayLike,
                                            source_coor_y: npt.ArrayLike,
                                            target_coor_x: npt.ArrayLike,
                                            target_coor_y: npt.ArrayLike,
                                            center: bool = True,
                                            channel_first: Optional[bool] = None) -> npt.NDArray:
    """Interpolates (bi-linear) a vector array from the source to the target grid coordinates.

    Given a 2D vector array (i.e. MxNx3 array containing vector components), the array values are projected onto a
    new grid of different dimensions and/or resolution using bi-linear interpolation.

    :param array: Source data array.
    :param source_coor_x: Array containing the X coordinates of the source grid.
    :param source_coor_y: Array containing the Y coordinates of the source grid.
    :param target_coor_x: Array containing the X coordinates of the target (projection) grid.
    :param target_coor_y: Array containing the Y coordinates of the target (projection) grid.
    :param center: Whether to center the coordinates axis or not. See `center_coordinates` for more information.
    :param channel_first: Whether the array is formatted in channel first order (i.e. the first dimension corresponds to
    the vector components.
    :return: The interpolated array in the target grid coordinates.
    """
    # Original array
    array = np.array(array)
    source_coor_x = np.array(source_coor_x)
    source_coor_y = np.array(source_coor_y)

    # Projection axis
    target_coor_x = np.array(target_coor_x)
    target_coor_y = np.array(target_coor_y)

    # Consider center coordinates
    if center:
        source_coor_x = center_coordinates(source_coor_x)
        source_coor_y = center_coordinates(source_coor_y)
        target_coor_x = center_coordinates(target_coor_x)
        target_coor_y = center_coordinates(target_coor_y)

    # Infer order of channels
    if channel_first is None:
        i, j, k = array.shape

        if i == 3 and k == 3:
            # TODO: FALLBACK, print message
            channel_first = True
        elif i == 3:
            channel_first = True
        elif k == 3:
            channel_first = False
        else:
            raise

    if channel_first:
        array_layers = (array[i, :, :] for i in range(3))
    else:
        array_layers = (array[:, :, i] for i in range(3))

    # Interpolation
    interpolated_layers = []
    for layer in array_layers:
        interpolator = interpolate.RectBivariateSpline(x=source_coor_x,
                                                       y=source_coor_y,
                                                       z=layer,
                                                       kx=1, ky=1)
        interpolated_array = interpolator(target_coor_x, target_coor_y)
        interpolated_layers.append(interpolated_array)

    # Stack
    if channel_first:
        stack_axis = 0
    else:
        stack_axis = -1

    interpolated_array = np.stack(interpolated_layers, axis=stack_axis)

    return interpolated_array


def calc_2darray_nearest_interpolation_old(array: npt.ArrayLike,
                                           *,
                                           source_coor_x: npt.ArrayLike,
                                           source_coor_y: npt.ArrayLike,
                                           target_coor_x: npt.ArrayLike,
                                           target_coor_y: npt.ArrayLike,
                                           center: bool = False,
                                           ) -> npt.NDArray:
    # Original array
    array = np.array(array)
    source_coor_x = np.array(source_coor_x)
    source_coor_y = np.array(source_coor_y)

    # Projection axis
    target_coor_x = np.array(target_coor_x)
    target_coor_y = np.array(target_coor_y)

    # Consider center coordinates
    if center:
        source_coor_x = center_coordinates(source_coor_x)
        source_coor_y = center_coordinates(source_coor_y)
        target_coor_x = center_coordinates(target_coor_x)
        target_coor_y = center_coordinates(target_coor_y)

    interpolator = interpolate.RegularGridInterpolator(points=(source_coor_x, source_coor_y),
                                                       values=array,
                                                       method="nearest",
                                                       bounds_error=False)

    target_grid = np.stack(np.meshgrid(target_coor_x, target_coor_y, indexing="ij"), -1)
    interpolated_array = interpolator(target_grid.flatten()).reshape(target_grid.shape[:2])

    return interpolated_array


def calc_simple_grid_nearest_interpolation(array: npt.ArrayLike,
                                           *,
                                           source_coor_x: npt.ArrayLike,
                                           source_coor_y: npt.ArrayLike,
                                           target_coor_x: npt.ArrayLike,
                                           target_coor_y: npt.ArrayLike,
                                           center: bool = False) -> npt.NDArray:
    """Interpolates (nearest neighbour) a simple grid array from the source to the target grid coordinates.

    Given a 2D array (i.e. MxN array containing numeric data), the array values are projected onto a new grid of
    different dimensions and/or resolution using nearest neighbour interpolation.

    :param array: Source data array.
    :param source_coor_x: Array containing the X coordinates of the source grid.
    :param source_coor_y: Array containing the Y coordinates of the source grid.
    :param target_coor_x: Array containing the X coordinates of the target (projection) grid.
    :param target_coor_y: Array containing the Y coordinates of the target (projection) grid.
    :param center: Whether to center the coordinates axis or not. See `center_coordinates` for more information.
    :return: The interpolated array in the target grid coordinates.
    """

    # Original array
    array = np.array(array)
    source_coor_x = np.array(source_coor_x)
    source_coor_y = np.array(source_coor_y)

    # Projection axis
    target_coor_x = np.array(target_coor_x)
    target_coor_y = np.array(target_coor_y)

    # Consider center coordinates
    if center:
        source_coor_x = center_coordinates(source_coor_x)
        source_coor_y = center_coordinates(source_coor_y)
        target_coor_x = center_coordinates(target_coor_x)
        target_coor_y = center_coordinates(target_coor_y)

    #
    xllcorner = target_coor_x[0]
    yllcorner = target_coor_y[0]

    source_cellsize = source_coor_x[1] - source_coor_x[0]
    target_cellsize = target_coor_x[1] - target_coor_x[0]

    idx_x = [int(((xllcorner + target_cellsize * 0.5 - source_coor_x[0]
                   - source_cellsize * 0.5) + n * target_cellsize) / source_cellsize)
             for n in range(len(target_coor_x))]
    idx_x = np.clip(idx_x, 0, len(source_coor_x) - 1)

    idx_y = [int(((yllcorner + target_cellsize * 0.5 - source_coor_y[0]
                   - source_cellsize * 0.5) + n * target_cellsize) / source_cellsize)
             for n in range(len(target_coor_y))]
    idx_y = np.clip(idx_y, 0, len(source_coor_y) - 1)

    return np.array([[array[i][j] for j in idx_y] for i in idx_x])
