"""Pipeline to retrieve data from a single data source for cleaning."""

import logging
from pathlib import Path

import hydra
import pandas as pd

import train_model as tm

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="process_data.yaml", version_base=None)
def main(args):
    """Main fuction to retrieve and process initial raw data."""
    logger.info("Validating file path")
    if not Path(args.data_folder).is_dir():
        err_msg = f"Data path {args.data_folder} does not exist"
        logger.error(err_msg)
        raise NotADirectoryError(err_msg)

    raw_data_path = Path(args.data_folder, args.raw_file)
    cleaned_data_path = Path(args.data_folder, args.cleaned_file)

    # retrieve data, validate data with pydantic
    # TODO: to make this a potential abstract class for various dataset
    logger.info("Retrieving data")
    hdb_data = tm.retrieve_data.get_multiple_offset_response(args.api_entry_call)

    logger.info(f"Saving data to {raw_data_path}")
    data = pd.DataFrame([d.model_dump() for d in hdb_data])
    data.to_csv(raw_data_path, index=False)

    logger.info("Processing / Cleaning data")
    cleaner = tm.data_cleaner.HdbDataCleaner()
    cleaned_data = cleaner.clean_data(data)

    # TODO: add save_data function into the cleaner
    logger.info(f"Data cleaned and saved to {cleaned_data_path}")
    cleaned_data.to_csv(cleaned_data_path, index=False)


if __name__ == "__main__":
    main()
