"""Module to retrieve the data from API."""

import logging
import os
from typing import Any

import dotenv
import requests

from . import data_model

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL")
API_URL = os.getenv("API_URL")

# if API_URL is None:
#     error_msg = "API_URL is None, check environment variable"
#     logger.error(error_msg)
#     raise ValueError(error_msg)


def get_single_response(api_url: str):
    """Get a single api call response."""
    response = requests.get(api_url)

    return response


def get_multiple_offset_response(entry_number: int = 500) -> list[dict[str, Any]]:
    """To call multiple api calls using offset."""
    response = get_single_response(API_URL)

    resp = response.json()
    resp = resp["result"]

    hdb_results = [data_model.HDBData(**r) for r in resp["records"]]
    total_records = resp["total"] if entry_number == 0 else entry_number
    offset_records = resp.get("offset", 0)

    while offset_records < total_records:

        logger.debug("Retrieving additional records", offset_records, total_records)

        next_api = BASE_URL + resp["_links"]["next"]

        response = get_single_response(next_api)

        resp = response.json()
        resp = resp["result"]
        offset_records = resp.get("offset")

        hdb_results.extend([data_model.HDBData(**r) for r in resp["records"]])

    return hdb_results
