from typing import Any, TypedDict

import pandas as pd
from django.conf import settings

from api.general.utils.loading import (
    JsonMapping,
    PklMapping,
    load_files,
)

# Private constants for module name and file mappings
_MODULE_NAME = "Distrib"

_PKL_FILES_DATA = PklMapping(multe_data="multe_data.pkl")

_JSON_FILES_DATA = JsonMapping()


# Data structures
class DistribData(TypedDict):
    """
    TypedDict for the distribution data structure.
    """

    multe_data: pd.DataFrame


# Private function to postprocess the data
def _postprocess(data: dict[str, Any]) -> DistribData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return DistribData(**data)


# Public variables and functions
# Data storage
data_store = DistribData(multe_data=pd.DataFrame())


# Load data function
def load_data() -> None:
    """
    Load the distribution data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_files(
        settings.DATA_DIR / "distrib",
        _MODULE_NAME,
        _PKL_FILES_DATA,
        _JSON_FILES_DATA,
    )

    data_store.update(**_postprocess(out_data))
