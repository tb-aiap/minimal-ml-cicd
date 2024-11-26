"""Data models to define expected datatypes for each model phase."""

from pydantic import BaseModel


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
