"""Models from sklearn library."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import omegaconf
import sklearn

from ..utils import utils

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """Defining methods for training pipeline."""

    @abstractmethod
    def fit(self, x, y) -> None:
        """Abstract method to fit the model."""
        ...

    @abstractmethod
    def predict(self, x) -> np.ndarray:
        """Abstract method to predict with the model."""
        ...

    @abstractmethod
    def save(self, save_path: str | Path) -> None:
        """Method to save the model as an object."""
        ...


class SKLearnPredictor(Predictor):
    """SKLearn related models Predictor implemented here.

    Takes in SKLearn's estimator as the Predictor.
    """

    def __init__(
        self,
        params: omegaconf.DictConfig,
        model: sklearn.base.BaseEstimator,
    ):
        """Initialize to ingest yaml config params."""
        self.params = params
        self.model_obj = model

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model after loading."""
        logger.debug("Initializing model with model and params.")
        self.model = self.model_obj(**self.params)

    def fit(self, x, y):
        """SKLearn's fit method."""
        self.model.fit(x, y)

    def predict(self, x) -> np.ndarray:
        """SKLearn's predict method."""
        ypred = self.model.predict(x)
        return ypred

    def save(self, save_path: str | Path) -> None:
        """Saving the model object as pickle file."""
        if isinstance(save_path, str):
            save_path = Path(save_path)

        model_name = self.model.__class__.__name__.lower()
        logger.debug(f"Saving model {model_name} into {save_path} folder.")

        m_file_path = Path(save_path, model_name)
        utils.save_object(self.model, m_file_path)
