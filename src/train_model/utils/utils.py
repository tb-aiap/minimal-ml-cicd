"""Other utilities function."""

import importlib
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_object(target_object: Any, file_path: str | Path) -> None:
    """Save target object as pickle file."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with open(file_path.with_suffix(".pkl"), "wb") as picklefile:
        pickle.dump(target_object, picklefile)


def load_object(file_path: str | Path) -> Any:
    """Load target pickle filepath as Python object."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.suffix != ".pkl":
        raise TypeError("Expecting .pkl file extension for this function.")

    with open(file_path, "rb") as picklefile:
        f = pickle.load(picklefile)

    return f


def load_func(dotpath: str):
    """Load function in module. Function name is right-most segment.

    Requires full library name.

    Example:
    A string value torch.nn.MSELoss
    module_ = torch.nn
    func_result = getattr(module, MSELoss)

    A string value numpy.sum / Does not work with np.sum
    module_ = numpy
    func_result = getattr(module, sum):q
    """
    module_, func = dotpath.rsplit(".", maxsplit=1)

    try:
        m = importlib.import_module(module_)
        func_result = getattr(m, func)
    except (AttributeError, ModuleNotFoundError) as e:
        logger.error("Check spelling in config '{}' Error - {}".format(dotpath, e))
        raise
    logger.debug("load_func returns result = {}".format(func_result))
    return func_result
