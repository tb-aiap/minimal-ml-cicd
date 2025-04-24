"""Test module for data preprocessor."""

import datetime
from pathlib import Path

import omegaconf
import pytest

import train_model as tm


@pytest.fixture
def config(tmp_path):
    """Config yaml mock for preprocessor."""
    config = {
        "save_path": tmp_path,
    }
    return omegaconf.DictConfig(config)


def test_utils_save_and_load_object(config):
    """Test utility function for saving and loading object."""
    sample_object = [1, 2, 3]
    object_path = Path(config.save_path, "sample_list.pkl")
    tm.utils.utils.save_object(sample_object, object_path)

    assert len(list(object_path.parent.iterdir())) == 1

    loaded_object = tm.utils.utils.load_object(object_path)

    assert loaded_object == sample_object


def test_utils_save_and_load_object_wrong_extension(config):
    """Test utility function for saving and loading object."""
    sample_object = [1, 2, 3]
    object_path = Path(config.save_path, "sample_list.pl")
    tm.utils.utils.save_object(sample_object, object_path)

    with pytest.raises(TypeError):
        tm.utils.utils.load_object(object_path)


def test_utils_load_function():
    """Test utility function that loads a module from string input."""
    import datetime

    dotpath = "datetime.datetime"
    assert type(tm.utils.utils.load_func(dotpath)) is type(datetime.datetime)

    import random

    dotpath = "random.random"
    assert type(tm.utils.utils.load_func(dotpath)) is type(random.random)


def test_get_output_folder(config):
    """Test utility function that generates an output folder."""
    root_path = config.save_path
    now_time = datetime.datetime.now()
    day = f"{now_time.day:02d}"
    hour = now_time.hour
    minute = now_time.minute
    r = tm.utils.utils.get_output_folder(root_path)

    assert r.parent.name.split("-")[2] == day
    assert r.name.split("-")[0] == str(hour)
    assert r.name.split("-")[1] == str(minute)
