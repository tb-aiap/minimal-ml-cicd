"""Module to evaluate the results of the a model prediction."""

import logging

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

    def __init__(self, params: omegaconf.DictConfig):
        """Initialize evaluator with required params.

        Params required is a list of metrics to measure for model performance.
        List of metric is the dotpath of the Python module/function to calulate metric.

        """
        self.params: list[str] = params
        self.metrics: dict[str, float] = dict()

    def __validate_params(self) -> None: ...

    def initialize_metrics_fn(self):
        """Reads a list of str and parse it into Python functions."""
        return [utils.load_func(p) for p in self.params]

    def evaluate(self, ypred, ytrue):
        """Evaluates ypred and ytrue with a list of metrics provided in params."""
        metrics_fn = self.initialize_metrics_fn()
        for fn in metrics_fn:
            fn_name = fn.__name__
            fn_result = fn(ytrue, ypred)
            self.metrics[fn_name] = fn_result
