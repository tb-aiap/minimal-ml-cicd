"""Test modules related to models module."""

import numpy as np
import omegaconf
import pytest
from sklearn.metrics import accuracy_score, f1_score

import train_model as tm


@pytest.fixture
def config(tmp_path):
    """Config yaml mock for linear model params."""
    config = {
        "save_path": tmp_path,
        "evaluate": ["sklearn.metrics.f1_score", "sklearn.metrics.accuracy_score"],
    }
    return omegaconf.DictConfig(config)


@pytest.fixture
def classification_data():
    """Test data for ytrue and ypred classification."""
    ytrue = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    ypred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    return ytrue, ypred


@pytest.fixture
def regression_data():
    """Dataframe fixture to train mode."""
    ytrue = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    ypred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    return ytrue, ypred


def test_evaluator_initialize(
    config,
    classification_data,
):
    """Test sklearn predictor method allows different type of model and respective params."""
    ytrue, ypred = classification_data
    sklearn_model = tm.evaluator.Evaluator(config.evaluate)
    sklearn_model.evaluate(ytrue, ypred)

    assert sklearn_model.metrics["f1_score"] == f1_score(ytrue, ypred)
    assert sklearn_model.metrics["accuracy_score"] == accuracy_score(ytrue, ypred)
