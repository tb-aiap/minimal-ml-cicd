"""Module for data cleaning and feature engineering."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data_model, utils

logger = logging.getLogger(__name__)
COL = data_model.ColumnEnum


PREPROCESSOR = {"standardscaler": StandardScaler, "onehotencoder": OneHotEncoder}


class DataPreprocessor(ABC):
    """For cleaning data into required format."""

    @abstractmethod
    def fit_preprocessors(self, data: pd.DataFrame):
        """Main abstract method to clean data."""
        ...

    @abstractmethod
    def feature_engineer(self):
        """Validate method to check args."""
        ...


class HdbDataPreprocessor(DataPreprocessor):
    """For feature engineering and additional preprocessing."""

    def __init__(
        self, params: omegaconf.DictConfig, object_filepath: str | Path
    ) -> None:
        """Initialize data preprocessor."""
        self.params = params
        self.object_filepath = object_filepath
        self._preprocessors = []
        self._validate_params()

    def _validate_params(self):
        """Check the following in params.

        - If preprocessors are typed correctly in config file.
        """
        for key in self.params:
            if PREPROCESSOR.get(key) is None:
                raise NotImplementedError(
                    f"{key} is either not implemented or check spelling."
                )

        if isinstance(self.object_filepath, str):
            logger.debug(f"Converting {self.object_filepath} in to Path object.")
            self.object_filepath = Path(self.object_filepath)

    def fit_preprocessors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for preprocessing / scaling .

        Save the preprocessor after fit_transform.
        Load the preprocess for transform.
        """
        for key in self.params:
            preprocessor = PREPROCESSOR.get(key)()
            preprocess_data = data[self.params[key]["columns"]]
            preprocessor.fit(preprocess_data)
            self._preprocessors.append(preprocessor)

        self._save_preprocessors()
        return None

    def _save_preprocessors(self):
        """Save preprocessors after fitting it."""
        for p in self._preprocessors:
            logger.debug(f"Saving preprocessor {p} into {self.object_filepath} folder.")
            p_name = p.__class__.__name__.lower()
            p_file_path = Path(self.object_filepath, p_name)
            utils.utils.save_object(p, p_file_path)

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Uses fitted scaler to preprocessing / scaling .

        Load the preprocess for transform if self._preprocessors is empty.
        Uses exisiting self.preprocessors after fitted for transforming data.
        """
        scaled_data_array = []
        scaled_data_names = []

        if not self._preprocessors:
            logger.info(f"Preprocessor attribute is empty {self._preprocessors=}.")
            logger.info(f"Loading preprocesors from {self.object_filepath}")

            for key in self.params:
                p_file_path = Path(self.object_filepath, key + ".pkl")
                preprocessor = utils.utils.load_object(p_file_path)
                self._preprocessors.append(preprocessor)

        for p in self._preprocessors:
            p_name = p.__class__.__name__.lower()
            logger.info(f"Transforming data with {p_name}")
            preprocess_data = data[self.params[p_name]["columns"]]
            scaled_data = p.transform(preprocess_data)

            if scipy.sparse.issparse(scaled_data):
                logger.debug(f"Converting sparse matrix from {p}")
                scaled_data = scaled_data.toarray()

            scaled_data_array.append(scaled_data)
            scaled_data_names.append(p.get_feature_names_out())

        df = pd.DataFrame(
            np.concatenate(scaled_data_array, axis=1),
            columns=np.concatenate(scaled_data_names),
        )
        return df

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
        data[COL.storey_area_ratio.name] = (
            data[COL.floor_area_sqm] / data[COL.storey_to]
        )

        return data

    def _fe_lease_less_than_50_yrs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineer for boolean lease if it is less than 50 years remaining."""
        data[COL.lease_less_than_50_yrs.name] = data[COL.remaining_lease] < 50

        return data
