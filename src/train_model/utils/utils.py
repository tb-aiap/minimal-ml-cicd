"""Other utilities function."""

import pickle
from pathlib import Path
from typing import Any


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
