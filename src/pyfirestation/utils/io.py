"""
Various assorted functions and classes for reading and working with files or strings containing data retrieved from data
 files.

"""

import io
import pathlib
from contextlib import contextmanager
from enum import Enum, auto
from typing import Union, Optional, TextIO, Generator

from .custom_types import PathLike


class DataInputMode(Enum):
    """Possible types of inputs for the reader functions.

    The 'FILEPATH' mode considers that input data is provided in files (or the respective system paths) that can be
    read using internal functions and modules (e.g. csv.py) or external libraries (e.g. Numpy). The 'DATASTRING' mode
    considers that the input data is provided as a string object. This can be useful, for instance, for parsing data
    from an API call. The 'DATABYTES' mode considers a binary stream of data.
    """
    FILEPATH = auto()
    DATASTRING = auto()
    DATABYTES = auto()


class InvalidDataInputModeError(Exception):
    """Exception class for invalid input mode."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DataInputModeInferenceError(Exception):
    """Exception class for unsuccessful input mode inference."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def infer_input_mode(origin: Union[str, PathLike]) -> DataInputMode:
    """Infers DataInputMode from provided origin.

    Simple function that tries to infer if provided origin (i.e. data) is either a path-like object which leads to a
    file or a str object with the desired data encoded. Since the logic behind the inference is rather simple,
    for robustness it is better to rely on explicitly defining the input mode whenever possible.

    :param origin: The data origin, usually a file path containing the desired data or a string already read.
    :return: The inferred DataInputMode.
    """
    is_path = isinstance(origin, pathlib.Path)
    is_str = isinstance(origin, str)
    is_multiline = is_str and (str(origin).count("\n") + 1) != 1

    is_filepath = is_path or (is_str and not is_multiline)
    is_datastring = is_str and is_multiline

    if is_filepath:
        return DataInputMode.FILEPATH
    elif is_datastring:
        return DataInputMode.DATASTRING
    else:
        raise DataInputModeInferenceError("Could not infer input mode from provided origin.")


def validate_input_mode(mode: Union[str, DataInputMode]) -> DataInputMode:
    """Validates the provided input mode for the origin_handle or similar handles or data processing functions.

    :param mode: The mode used for reading
    :return: If valid, the provided mode.
    """

    def raise_input_mode_error():
        raise InvalidDataInputModeError(f"Provided parser mode does not match any available mode."
                                        f"\n\t- Available modes: {tuple([e.name for e in DataInputMode])}")

    if isinstance(mode, str):
        mode = mode.strip().upper()
        try:
            mode = DataInputMode[mode]
            return mode
        except KeyError:
            raise_input_mode_error()
    elif isinstance(mode, DataInputMode):
        if mode in DataInputMode:
            return mode
        else:
            raise_input_mode_error()

    raise TypeError("Parser mode should be either \'FILEPATH\' (for data files)"
                    "or \'DATASTRING\' (for raw string data).")


@contextmanager
def origin_handle(origin: Union[str, PathLike],
                  *,
                  mode: Optional[DataInputMode] = None,
                  encoding: Optional[str] = None) -> Generator[TextIO, None, None]:
    """Context manager for streamlining reading data from both str objects and files.

    :param origin: A path for a file or a string containing the data to be processed.
    :param mode: If origin is a file or a string.
    :param encoding: Encoding of the data.
    """
    # Infer/Validate mode
    if mode is None:
        mode = infer_input_mode(origin)
    mode = validate_input_mode(mode)

    # Build context manager
    if mode is DataInputMode.FILEPATH:
        with open(origin, "r", encoding=encoding, newline="") as file_handle:
            yield file_handle
    elif mode is DataInputMode.DATASTRING:
        with io.StringIO(initial_value=str(origin)) as string_handle:
            yield string_handle
    else:
        # Possible to implement io.BytesIO buffer here
        raise NotImplementedError("Currently bytes input is not supported.")
