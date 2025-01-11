"""Module for data cleaning and feature engineering."""

import logging
from abc import ABC, abstractmethod

import omegaconf
import pandas as pd

from . import data_model

logger = logging.getLogger(__name__)

COL = data_model.ColumnEnum


class DataCleaner(ABC):
    """For cleaning data into required format."""

    @abstractmethod
    def clean_data(self, data: pd.DataFrame):
        """Main abstract method to clean data."""
        ...

    @abstractmethod
    def validate_args(self):
        """Validate method to check args."""
        ...


class HdbDataCleaner(DataCleaner):
    """Data Cleaner Class to clean data before being used by model training pipeline."""

    def __init__(self, args: omegaconf.DictConfig = None):
        """Takes in args if required."""
        self.args = args

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main method to clean the data for saving.

        Args:
            data (pd.DataFrame): Input data as dataframe

        Returns:
            pd.DataFrame: Cleaned data as dataframe
        """
        data = self._normalize_storey_range(data)

        return data

    def validate_args(self):
        """Placeholder method for validate_args method."""
        ...

    def _normalize_storey_range(self, data: pd.DataFrame):
        """Method to normalize storey range.

        Args:
            data (pd.DataFrame): Input data with required storey range.

        Returns:
            _type_: Resultant data with new columns.
        """
        data[COL.storey_from] = data[COL.storey_range].str.split(" TO ", expand=True)[0]
        data[COL.storey_to] = data[COL.storey_range].str.split(" TO ", expand=True)[1]

        return data
