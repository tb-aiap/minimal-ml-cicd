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


def get_output_folder(root_path: str | Path) -> Path:
    """Get date folder - time folder for saving artifacts.

    Args:
        root_path (str | Path): The path to create date folder = time folder

    Raises:
        FileExistsError: If the same time folder exists, it errors out

    Returns:
        Path: The full folder path for ./yyyy-mm-dd/hh-mm-ss-msms
    """
    if isinstance(root_path, str):
        root_path = Path(root_path)

    import datetime

    date_obj = datetime.datetime.now()

    year = date_obj.year
    month = date_obj.month
    day = f"{date_obj.day:02d}"

    hour = date_obj.hour
    minute = date_obj.minute
    second = date_obj.second
    microsecond = date_obj.microsecond

    day_folder = f"{year}-{month}-{day}"
    root_day_folder = Path(root_path, day_folder)

    if not root_day_folder.exists():
        root_day_folder.mkdir(parents=True)

    time_folder = f"{hour}-{minute}-{second}-{microsecond//100:04d}"
    root_time_folder = Path(root_day_folder, time_folder)

    if root_time_folder.exists():
        error_msg = f"{root_time_folder} already exists."
        logger.error(error_msg)
        raise FileExistsError(error_msg)

    root_time_folder.mkdir(parents=True, exist_ok=True)

    return root_time_folder
