"""Module for data cleaning and feature engineering."""

import logging
from abc import ABC, abstractmethod

import omegaconf
import pandas as pd

from . import data_model

logger = logging.getLogger(__name__)
COL = data_model.ColumnEnum


class DataPreprocessor(ABC):
    """For cleaning data into required format."""

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame):
        """Main abstract method to clean data."""
        ...

    @abstractmethod
    def feature_engineer(self):
        """Validate method to check args."""
        ...


class HdbDataPreprocessor(DataPreprocessor):
    """For feature engineering and additional preprocessing."""

    def __init__(self, params: omegaconf.DictConfig):
        """Initialize data preprocessor."""
        self.params = params

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for preprocessing / scaling ."""
        # one hot encode columns
        return data

    def feature_engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main function to feature engineer data."""
        data = self._fe_ratio_storey_to_floor_area(data)
        data = self._fe_lease_less_than_50_yrs(data)

        return data

    # TODO: for feature engineering, if the return is dataframe, it is hard to trace the
    # result of engineered features. example, if the feature engineered is a bool, return
    # value will still be a pd.DataFrame

    def _fe_ratio_storey_to_floor_area(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineer ratio of floor storey and sq area."""
        data[COL.storey_area_ratio] = data[COL.floor_area_sqm] / data[COL.storey_to]

        return data

    def _fe_lease_less_than_50_yrs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineer for boolean lease if it is less than 50 years remaining."""
        data[COL.lease_less_than_50_yrs] = data[COL.remaining_lease] < 50

        return data
