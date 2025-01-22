"""Test module for data preprocessor."""

import pandas as pd
import pytest

import train_model as tm

COL = tm.data_model.ColumnEnum


@pytest.fixture
def data():
    """Dataframe fixture to test feature engineering."""
    df = pd.DataFrame(
        {
            "storey_range": ["1 TO 4", "9 TO 12", "11 TO 15", "22 TO 24", "14 TO 24"],
            "Column2": ["A", "B", "C", "D", "E"],
            "storey_to": [4, 12, 9, 9, 12],
            "floor_area_sqm": [60, 70, 80, 90, 60],
            "remaining_lease": [49, 50, 51, 90, 60],
        }
    )
    return df


def test_feature_engineer(data):
    """Test main feature engineer function."""
    args = {"sample": 1}
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(args)
    cleaned_data = cleaner.feature_engineer(data)

    assert COL.storey_area_ratio in cleaned_data.columns
    assert COL.lease_less_than_50_yrs in cleaned_data.columns


def test_fe_ratio_storey_to_floor_area(data):
    """Test feature engineer storey range."""
    args = {"sample": 1}
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(args)
    cleaned_data = cleaner._fe_ratio_storey_to_floor_area(data)

    assert COL.storey_area_ratio in cleaned_data.columns
    assert cleaned_data[COL.storey_area_ratio].values[0] == 15.0
    assert cleaned_data[COL.storey_area_ratio].values[1] == (70 / 12)


def test_fe_lease_less_than_50_yrs(data):
    """Test feature engineer for lease less than 50 yrs bool."""
    args = {"sample": 1}
    cleaner = tm.data_preprocessor.HdbDataPreprocessor(args)
    cleaned_data = cleaner._fe_lease_less_than_50_yrs(data)

    assert COL.lease_less_than_50_yrs in cleaned_data.columns
    assert cleaned_data[COL.lease_less_than_50_yrs].values[0]
    assert not cleaned_data[COL.lease_less_than_50_yrs].values[1]
    assert not cleaned_data[COL.lease_less_than_50_yrs].values[2]
