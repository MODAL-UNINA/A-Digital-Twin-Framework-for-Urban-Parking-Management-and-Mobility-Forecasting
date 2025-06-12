from typing import Any, TypedDict

import geopandas as gpd

# Imports for data structures and preprocessing functions
import pandas as pd
from django.conf import settings

from api.general.startup import ZoneDictZoneDataMapping
from api.general.utils.loading import (
    JsonMapping,
    PklMapping,
    load_files,
)

# Private constants for module name and file mappings
_MODULE_NAME = "Map"

_PKL_FILES_DATA = PklMapping(
    parkingmeter_positions="posizioni_parcometri_new.pkl",
    roads_gdf="roads_gdf.pkl",
    sensors="stalli_selection_new.pkl",
)

_JSON_FILES_DATA = JsonMapping()


# Data structures
class MapData(TypedDict):
    """
    TypedDict for the map data structure.
    """

    parkingmeter_positions: pd.DataFrame
    roads_gdf: gpd.GeoDataFrame
    sensors: pd.DataFrame


# Private function to postprocess the data
def _postprocess_parkingmeter_positions(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data["id_parcometro"] = data["id_parcometro"].astype(int)
    data = data.rename(columns={"id_parcometro": "id", "id_strada": "road_id"})
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x: int(  # type: ignore
            [
                zone_name
                for zone_name in zone_names
                if x in zone_dict[zone_name]["parcometro"]
            ][0].split("_")[1]
        )
        + 1
    )
    return data


def _postprocess_roads_gdf(
    data: gpd.GeoDataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> gpd.GeoDataFrame:
    data.insert(  # type: ignore
        1,
        "zone_id",
        data["road_id"].apply(  # type: ignore
            lambda x: int(  # type: ignore
                [
                    zone_name
                    for zone_name in zone_names
                    if x in zone_dict[zone_name]["strade"]
                ][0].split("_")[1]
            )
            + 1
        ),
    )
    return data


def _postprocess_sensors(
    data: pd.DataFrame, zone_dict: ZoneDictZoneDataMapping, zone_names: list[str]
) -> pd.DataFrame:
    data = data.rename(columns={"id_strada": "road_id"})

    sensors_mapping = {
        zone_name: zone_dict[zone_name]["stalli"] for zone_name in zone_names
    }
    sensors_sets = [set(v) for v in sensors_mapping.values()]
    assert len(set.intersection(*sensors_sets)) == 0, (  # type: ignore
        "There are sensors that are present in multiple zones. "
        "This is not allowed and will cause issues in the application."
    )
    sensors_mapping_inv = {
        x: zone_name for zone_name, v in sensors_mapping.items() for x in v
    }
    data["zone_id"] = data["id"].apply(  # type: ignore
        lambda x:  # type: ignore
        int(sensors_mapping_inv[x].split("_")[1]) + 1
        if x in sensors_mapping_inv
        else -1
    )
    data = data[data["zone_id"] != -1]  # Filter out sensors not in any zone
    return data


def _postprocess(data: dict[str, Any]) -> MapData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    from api.general.views import get_zone_dict

    zone_dict = get_zone_dict()
    zone_names = sorted(
        [zone_name for zone_name in zone_dict.keys() if zone_name != "all_map"],
        key=lambda v: int(v.split("_")[1]),
    )

    data["parkingmeter_positions"] = _postprocess_parkingmeter_positions(
        data["parkingmeter_positions"], zone_dict, zone_names
    )
    data["roads_gdf"] = _postprocess_roads_gdf(data["roads_gdf"], zone_dict, zone_names)
    data["sensors"] = _postprocess_sensors(data["sensors"], zone_dict, zone_names)
    return MapData(**data)


# Public variables and functions
# Data storage
data_store = MapData(
    parkingmeter_positions=pd.DataFrame(),
    roads_gdf=gpd.GeoDataFrame(),
    sensors=pd.DataFrame(),
)


# Load data function
def load_data() -> None:
    """
    Load the map data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_files(
        settings.DATA_DIR / "map",
        _MODULE_NAME,
        _PKL_FILES_DATA,
        _JSON_FILES_DATA,
    )

    data_store.update(**_postprocess(out_data))
