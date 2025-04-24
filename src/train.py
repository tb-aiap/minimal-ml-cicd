"""Main module for training model."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from sklearn.model_selection import KFold

import train_model as tm
from train_model.models import Predictor
from train_model.utils.utils import get_output_folder, load_func

logger = logging.getLogger(__name__)


def load_data(file_path: Path | str) -> pd.DataFrame:
    """Read data for preprocessing."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    data = pd.read_csv(file_path)
    return data


@hydra.main(config_path="../conf", config_name="process_data.yaml", version_base=None)
def main(args):
    """Model Training Loop."""
    # get cleaned data
    logger.info(f"Loading train data from {args.data_folder}")
    cleaned_data_path = Path(args.data_folder, args.cleaned_file)
    data = load_data(cleaned_data_path)

    day_time_folder = get_output_folder(args.model_folder)
    # split data into train and val
    preprocesser = tm.data_preprocessor.HdbDataPreprocessor(
        params=args.preprocess,
        object_filepath=day_time_folder,
    )

    fe_data = preprocesser.feature_engineer(data)

    # TODO: remove the hard coded splits and move it to config
    cv_split = KFold(n_splits=3)
    cv_metrics = tm.evaluator.CVMetrics()
    for train_index, val_index in cv_split.split(fe_data):

        logger.info("START OF Cross Validation.")

        train_data = fe_data.iloc[train_index].drop(columns=args.target)
        train_label = fe_data.iloc[train_index][args.target]

        val_data = fe_data.iloc[val_index].drop(columns=args.target)
        val_label = fe_data.iloc[val_index][args.target]

        logger.info("Fitting Scalers")
        preprocesser.fit_preprocessors(train_data)
        scaled_train_data = preprocesser.transform_data(train_data)

        logger.info("Fitting Model")
        predictor = load_func(args.model.predictor_path)
        model_object = load_func(args.model.model_object)

        model: Predictor = predictor(params=args.model.params, model=model_object)
        model.fit(scaled_train_data, train_label)

        # evaluate
        scaled_val_data = preprocesser.transform_data(val_data)
        ypred = model.predict(scaled_val_data)

        evaluator = tm.evaluator.Evaluator(params=args.evaluate.metrics)
        evaluator.evaluate(ypred=ypred, ytrue=val_label)
        cv_metrics.update_metrics(evaluator.metrics)
        # output result for HPO
        # save artifact for model and scaler

    # TODO remove the hard coded metrics into config.
    objective = cv_metrics.get_mean("mean_absolute_error")

    return objective


if __name__ == "__main__":
    main()
