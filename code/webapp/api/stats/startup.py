from typing import Any, TypedDict

from django.conf import settings

from api.general.utils.loading import (
    JsonMapping,
    PklMapping,
    load_files,
)

# Imports for data structures and preprocessing functions
import pandas as pd  # isort: skip

# Private constants for module name and file mappings
_MODULE_NAME = "Stats"

_PKL_FILES_DATA = PklMapping(
    events_data="event_data.pkl",
    multe_data="multe_data_tab3.pkl",
    poi_data="pois_data.pkl",
    zone_params="zone_params.pkl",
)

_JSON_FILES_DATA = JsonMapping()


# Data structures
ZoneParamsMapping = dict[str, list[str]]


class StatsData(TypedDict):
    """
    TypedDict for the statistics data structure.
    """

    events_data: pd.DataFrame
    multe_data: pd.DataFrame
    poi_data: pd.DataFrame
    zone_params: ZoneParamsMapping


# Private function to postprocess the data
def _postprocess(data: dict[str, Any]) -> StatsData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return StatsData(**data)


# Public variables and functions
# Data storage
stats_data_store = StatsData(
    events_data=pd.DataFrame(),
    multe_data=pd.DataFrame(),
    poi_data=pd.DataFrame(),
    zone_params=ZoneParamsMapping(),
)


# Load data function
def load_data() -> None:
    """
    Load the stats data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_files(
        settings.DATA_DIR / "stats",
        _MODULE_NAME,
        _PKL_FILES_DATA,
        _JSON_FILES_DATA,
    )

    stats_data_store.update(**_postprocess(out_data))
