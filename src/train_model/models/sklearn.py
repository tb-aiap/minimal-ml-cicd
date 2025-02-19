"""Models from sklearn library."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import omegaconf

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


class SKLearnPredictor(Predictor):
    """SKLearn related models Predictor implemented here."""

    def __init__(self, params: omegaconf.DictConfig, model):
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
