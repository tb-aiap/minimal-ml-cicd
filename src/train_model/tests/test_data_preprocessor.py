"""Test module for data preprocessor."""

from pathlib import Path

import omegaconf
import pandas as pd
import pytest

import train_model as tm

COL = tm.data_model.ColumnEnum


@pytest.fixture
def config(tmp_path):
    """Config yaml mock for preprocessor."""
    config = {
        "save_path": tmp_path,
        "preprocessor": {
            "standardscaler": {"columns": ["floor_area_sqm", "storey_to"]},
            "onehotencoder": {"columns": ["Column2", "Column3"]},
        },
    }
    return omegaconf.DictConfig(config)


@pytest.fixture
def data():
    """Dataframe fixture to test feature engineering."""
    df = pd.DataFrame(
        {
            "storey_range": ["1 TO 4", "9 TO 12", "11 TO 15", "22 TO 24", "14 TO 24"],
            "Column2": ["A", "B", "C", "D", "E"],
            "Column3": ["F", "F", "M", "M", "F"],
            "storey_to": [4, 12, 9, 9, 12],
            "floor_area_sqm": [60, 70, 80, 90, 60],
            "remaining_lease": [49, 50, 51, 90, 60],
        }
    )
    return df


def test_preprocessor_init(config):
    """Test preprocessor config initialization."""
    preprocessor = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    assert "standardscaler" in preprocessor.params
    assert "onehotencoder" in preprocessor.params


def test_preprocessor_preprocess(config, data):
    """Test preprocess main method."""
    preprocessor = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    preprocessor.fit_preprocessors(data)
    assert [i.__class__.__name__ for i in preprocessor._preprocessors] == [
        "StandardScaler",
        "OneHotEncoder",
    ]
    preprocessor_2 = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    assert preprocessor_2._preprocessors == []


def test_preprocessor_save_scalers_properly(config, data):
    """Test preprocessor saves scalers into provided folder."""
    preprocessor = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    assert list(Path(config.save_path).iterdir()) == []
    preprocessor.fit_preprocessors(data)
    assert list(Path(config.save_path).iterdir()) != []


def test_preprocessor_transform_data(config, data):
    """Test preprocess transforms data into dataframe."""
    preprocessor = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    preprocessor.fit_preprocessors(data)
    assert [i.__class__.__name__ for i in preprocessor._preprocessors] == [
        "StandardScaler",
        "OneHotEncoder",
    ]
    dataframe = preprocessor.transform_data(data)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape[0] == 5
    assert dataframe.shape[1] == 9


def test_preprocessor_transform_data_inference(config, data):
    """Test preprocess main method."""
    preprocessor = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    preprocessor.fit_preprocessors(data)

    preprocessor_2 = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    scaled_df = preprocessor_2.transform_data(data)

    assert isinstance(scaled_df, pd.DataFrame)
    assert scaled_df.shape[0] == 5
    assert scaled_df.shape[1] == 9


def test_feature_engineer(config, data):
    """Test main feature engineer function."""
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    cleaned_data = cleaner.feature_engineer(data)

    assert COL.storey_area_ratio in cleaned_data.columns
    assert COL.lease_less_than_50_yrs in cleaned_data.columns


def test_fe_ratio_storey_to_floor_area(config, data):
    """Test feature engineer storey range."""
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    cleaned_data = cleaner._fe_ratio_storey_to_floor_area(data)

    assert COL.storey_area_ratio in cleaned_data.columns
    assert cleaned_data[COL.storey_area_ratio].values[0] == 15.0
    assert cleaned_data[COL.storey_area_ratio].values[1] == (70 / 12)


def test_fe_lease_less_than_50_yrs(config, data):
    """Test feature engineer for lease less than 50 yrs bool."""
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(
        config.preprocessor, config.save_path
    )
    cleaned_data = cleaner._fe_lease_less_than_50_yrs(data)

    assert COL.lease_less_than_50_yrs in cleaned_data.columns
    assert cleaned_data[COL.lease_less_than_50_yrs].values[0]
    assert not cleaned_data[COL.lease_less_than_50_yrs].values[1]
    assert not cleaned_data[COL.lease_less_than_50_yrs].values[2]
