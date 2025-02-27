"""Test modules related to models module."""

import numpy as np
import omegaconf
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

import train_model as tm


@pytest.fixture
def classification_config(tmp_path):
    """Config yaml mock for linear model params."""
    config = {
        "save_path": tmp_path,
        "evaluate": ["sklearn.metrics.f1_score", "sklearn.metrics.accuracy_score"],
    }
    return omegaconf.DictConfig(config)


@pytest.fixture
def regression_config(tmp_path):
    """Config yaml mock for linear model params."""
    config = {
        "save_path": tmp_path,
        "evaluate": [
            "sklearn.metrics.mean_absolute_error",
            "sklearn.metrics.mean_squared_error",
        ],
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
    ytrue = np.array([0.66, 0.76, 0.23, 0.45, 0.80, 0.44, 0.44, 0.75, 0.10, 0.23])
    ypred = np.array([0.56, 0.36, 0.93, 0.55, 0.79, 0.34, 0.64, 0.95, 0.20, 0.23])
    return ytrue, ypred


@pytest.fixture
def metric_hash():
    """Dictionary of cross validation results for CVMetric testing."""
    training_metrics = {
        "f1_score": [0.55, 0.44, 0.89, 0.67],
        "accuracy_score": [0.99, 0.87, 0.75, 0.90],
    }
    return training_metrics


def test_evaluator_classification_metric(
    classification_config,
    classification_data,
):
    """Test sklearn predictor method allows different type of model and respective params."""
    ytrue, ypred = classification_data
    sklearn_model = tm.evaluator.Evaluator(classification_config.evaluate)
    sklearn_model.evaluate(ytrue, ypred)

    assert sklearn_model.metrics["f1_score"] == f1_score(ytrue, ypred)
    assert sklearn_model.metrics["accuracy_score"] == accuracy_score(ytrue, ypred)


def test_evaluator_regression_metric(
    regression_config,
    regression_data,
):
    """Test sklearn predictor method allows different type of model and respective params."""
    ytrue, ypred = regression_data
    sklearn_model = tm.evaluator.Evaluator(regression_config.evaluate)
    sklearn_model.evaluate(ytrue, ypred)

    assert sklearn_model.metrics["mean_absolute_error"] == mean_absolute_error(
        ytrue, ypred
    )
    assert sklearn_model.metrics["mean_squared_error"] == mean_squared_error(
        ytrue, ypred
    )


def test_cv_metric_class_update_metrics(metric_hash):
    """Test CVMetrics class is able to update metrics."""
    single_hash = {k: v[0] for k, v in metric_hash.items()}
    cv_metrics = tm.evaluator.CVMetrics()
    cv_metrics.update_metrics(single_hash)

    assert isinstance(cv_metrics.metrics, dict)
    assert cv_metrics.metrics["f1_score"][0] == 0.55


def test_cv_metric_class_raises_error(metric_hash):
    """Test CVMetrics class is able to raise error when wrong metrics is provided."""
    cv_metrics = tm.evaluator.CVMetrics()
    cv_metrics.metrics = metric_hash

    with pytest.raises(KeyError):
        cv_metrics.get_mean("f2_score")


def test_cv_metric_class_get_mean(metric_hash):
    """Test CVMetrics class is able to raise error wit hthe wrong key."""
    cv_metrics = tm.evaluator.CVMetrics()
    cv_metrics.metrics = metric_hash

    assert cv_metrics.get_mean("f1_score") == 0.6375
