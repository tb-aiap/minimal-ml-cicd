"""Test modules related to models module."""

import omegaconf
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import train_model as tm


@pytest.fixture
def linear_regression_config(tmp_path):
    """Config yaml mock for linear model params."""
    config = {
        "save_path": tmp_path,
        "model": {
            "fit_intercept": True,
        },
    }
    return omegaconf.DictConfig(config)


@pytest.fixture
def randomforest_regressor_config(tmp_path):
    """Config yaml mock for random forest regressor params."""
    config = {
        "save_path": tmp_path,
        "model": {
            "n_estimators": 10,
        },
    }
    return omegaconf.DictConfig(config)


@pytest.fixture
def train_data():
    """Dataframe fixture to train mode."""
    df = pd.DataFrame(
        {
            "floor_area_sqm": [60, 70, 80, 90, 60],
            "remaining_lease": [49, 50, 51, 90, 60],
        }
    )
    return df


@pytest.fixture
def test_data():
    """Dataframe fixture to test model prediction."""
    df = pd.DataFrame(
        {
            "floor_area_sqm": [60, 70, 80, 90, 60],
        }
    )
    return df


@pytest.mark.parametrize(
    "model,config",
    [
        pytest.param(
            LinearRegression,
            "linear_regression_config",
            id="linear_regression",
        ),
        pytest.param(
            RandomForestRegressor,
            "randomforest_regressor_config",
            id="randomforestregressor",
        ),
    ],
)
def test_sklearn_predictor(
    config, train_data, test_data, model, request: pytest.FixtureRequest
):
    """Test sklearn predictor method allows different type of model and respective params."""
    config = request.getfixturevalue(config)
    sklearn_model = tm.models.sklearn.SKLearnPredictor(config.model, model=model)

    sklearn_model.fit(train_data[["floor_area_sqm"]], train_data["remaining_lease"])
    ypred = sklearn_model.predict(test_data[["floor_area_sqm"]])

    assert len(ypred) == test_data.shape[0]
