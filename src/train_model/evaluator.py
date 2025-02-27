"""Module to evaluate the results of the a model prediction."""

import logging
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import omegaconf

import train_model.utils.utils as utils

logger = logging.getLogger(__name__)


class Evaluator:
    """
    For evaluating the performance of the model.

    Retrieve the list of required metrics as params from a list of string.

    Example:
        - sklearn.metrics.f1_score
        - sklearn.metrics.accuracy_score

    `initialize_metrics_fn` will parse the str into python function for evaluation.

    str "sklearn.metrics.f1_score" becomes function `f1_score()`.
    """

    def __init__(self, params: omegaconf.DictConfig) -> None:
        """Initialize evaluator with required params.

        Params required is a list of metrics to measure for model performance.
        List of metric is the dotpath of the Python module/function to calulate metric.

        """
        self.params: list[str] = params

    def initialize_metrics_fn(self) -> list[Callable[[Any, Any], float]]:
        """Reads a list of str and parse it into Python functions."""
        return [utils.load_func(p) for p in self.params]

    def evaluate(self, ypred, ytrue):
        """Evaluates ypred and ytrue with a list of metrics provided in params."""
        metrics_fn = self.initialize_metrics_fn()
        self.metrics: dict[str, float] = dict()

        for fn in metrics_fn:
            fn_name = fn.__name__
            fn_result = fn(ytrue, ypred)
            self.metrics[fn_name] = fn_result


class CVMetrics:
    """Data class to store evaluation metrics from cross validation(cv)."""

    def __init__(self):
        """Initialize class with metrics attribute to store cv results."""
        self.metrics = defaultdict(list)

    def update_metrics(self, current_metrics: dict[str, float]) -> None:
        """Ingest current training metrics and store it across cv."""
        for k, v in current_metrics.items():
            self.metrics[k].append(v)

    def get_mean(self, metric_name: str) -> float:
        """Get the mean metrics of selected metrics across all cv."""
        result = self.metrics.get(metric_name)
        if result is None:
            raise KeyError(
                f"Selected Metric {metric_name} is not a valid name.",
                f"Expected the followings from {list(self.metrics.keys())}",
            )

        return np.mean(result)
