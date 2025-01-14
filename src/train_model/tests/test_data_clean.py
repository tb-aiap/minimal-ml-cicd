"""Test module for data cleaner."""

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
        }
    )
    return df


def test_fe_storey_range(data):
    """Test feature engineer storey range."""
    args = {"sample": 1}
    cleaner = tm.data_cleaner.HdbDataCleaner(args)
    cleaned_data = cleaner.clean_data(data)

    assert COL.storey_from in cleaned_data.columns
    assert COL.storey_to in cleaned_data.columns
    assert cleaned_data[COL.storey_from].tolist() == ["1", "9", "11", "22", "14"]
