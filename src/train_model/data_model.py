"""Data models to define expected datatypes for each model phase."""

from enum import Enum

from pydantic import BaseModel


class ColumnEnum(str, Enum):
    """Setting enums for column names."""

    # initial columns
    month = "month"
    town = "town"
    resale_price = "resale_price"
    flat_model = "flat_model"
    lease_commence_date = "lease_commence_date"
    storey_range = "storey_range"
    floor_area_sqm = "floor_area_sqm"
    remaining_lease = "remaining_lease"

    # cleaned column
    storey_from = "storey_from"
    storey_to = "storey_to"

    # feature engineered column
    storey_area_ratio = "storey_area_ratio"
    lease_less_than_50_yrs = "lease_less_than_50_yrs"


class HDBData(BaseModel):
    """Expected data types from API call."""

    _id: int
    month: str
    town: str
    flat_type: str
    block: str
    street_name: str
    storey_range: str
    floor_area_sqm: float
    flat_model: str
    lease_commence_date: int
    remaining_lease: int
    resale_price: float
