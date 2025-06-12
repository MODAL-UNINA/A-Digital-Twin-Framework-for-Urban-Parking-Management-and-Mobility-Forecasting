from typing import Any, TypedDict

import pandas as pd
from django.conf import settings

from api.general.utils.loading import (
    JsonMapping,
    PklMapping,
    load_files,
)

# imports for data structures and preprocessing functions
import datetime  # isort: skip

import numpy as np  # isort: skip
from numpy.typing import NDArray  # isort: skip

# Private constants for module name and file mappings
_MODULE_NAME = "General"

_PKL_FILES_DATA = PklMapping(
    transactions_parkingmeters="transactions_number_parkimeters.pkl",
    amount_parkingmeters="transactions_amount_parkimeters.pkl",
    all_sensors="sensor_data_new.pkl",
    status_sensors="status_sensor_data_new_2.pkl",
    zone="zone.pkl",
)

_JSON_FILES_DATA = JsonMapping(
    hourslots="hour_slots.json",
    timeslots_macroareas="time_slots_macroareas.json",
    zone_dict="mapping_dict.json",
    macrozone_params="macrozone_params.json",
)

# Data structures
FloatArray = NDArray[np.float64]


class ZoneData(TypedDict):
    """
    TypedDict for the zone data structure.
    """

    code: str
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    grid: list[FloatArray]


ZoneDataMapping = dict[str, ZoneData]


TimeSlotsMacroAreas = dict[str, list[list[list[int]]]]


class ZoneDictZoneData(TypedDict):
    parcometro: list[int]
    stalli: list[int]
    camera_ztl: list[str]
    strade: list[int]
    strade_name: list[str]


ZoneDictZoneDataMapping = dict[str, ZoneDictZoneData]


class CityMapData(TypedDict):
    area_id_zone_map: dict[int, str]
    area_name_map: dict[str, int]
    id_label_map: dict[int, str]
    area_parkingmeter_map: dict[int, list[int]]
    parkingmeter_area_map: dict[int, int]
    area_sensor_map: dict[int, list[int]]
    sensor_area_map: dict[int, list[int]]
    street_name_map: dict[int, str]
    area_street_map: dict[int, list[int]]
    street_area_map: dict[int, list[int]]


class HourSlotsData(TypedDict):
    range: list[int] | None
    label: str


class AvailableDatesData(TypedDict):
    min_date: datetime.date
    max_date: datetime.date


class MacroAreaMapData(TypedDict):
    macrozone_params: dict[str, list[str]]
    timeslots_macroareas: TimeSlotsMacroAreas
    hour_slots: dict[int, HourSlotsData]
    macroarea_id_map: dict[str, int]
    macroarea_name_map: dict[int, str]
    macroarea_label_map: dict[int, str]
    macroarea_timeslot_map: dict[str, dict[tuple[int, int], bool]]
    macroarea_hourslot_map: dict[str, dict[tuple[int, int], bool]]


class GeneralData(TypedDict):
    """
    TypedDict for the general data structure.
    """

    transactions_parkingmeters: pd.DataFrame
    amount_parkingmeters: pd.DataFrame
    all_sensors: pd.DataFrame
    status_sensors: pd.DataFrame
    zone: ZoneDataMapping
    hourslots: list[int]
    timeslots_macroareas: TimeSlotsMacroAreas
    zone_dict: ZoneDictZoneDataMapping
    macrozone_params: dict[str, list[str]]
    citymap: CityMapData
    hour_slots: dict[int, HourSlotsData]
    available_dates: AvailableDatesData
    macroarea_map: MacroAreaMapData


# Private function to postprocess the data
def _postprocess(data: dict[str, Any]) -> GeneralData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    data["citymap"] = build_city_map(data["zone_dict"])
    data["hour_slots"] = build_hour_slots(data["hourslots"])
    data["available_dates"] = build_available_dates(data["transactions_parkingmeters"])
    data["macroarea_map"] = build_macroarea_map(
        data["macrozone_params"],
        data["timeslots_macroareas"],
        data["hour_slots"],
    )
    return GeneralData(**data)


# Public variables and functions
# Data storage
data_store = GeneralData(
    transactions_parkingmeters=pd.DataFrame(),
    amount_parkingmeters=pd.DataFrame(),
    all_sensors=pd.DataFrame(),
    status_sensors=pd.DataFrame(),
    zone={},
    hourslots=[],
    timeslots_macroareas={},
    zone_dict={},
    macrozone_params={},
    citymap=CityMapData(
        area_id_zone_map={},
        area_name_map={},
        id_label_map={},
        area_parkingmeter_map={},
        parkingmeter_area_map={},
        area_sensor_map={},
        sensor_area_map={},
        street_name_map={},
        area_street_map={},
        street_area_map={},
    ),
    hour_slots={},
    available_dates=AvailableDatesData(
        min_date=datetime.date(1970, 1, 1), max_date=datetime.date(1970, 1, 1)
    ),
    macroarea_map=MacroAreaMapData(
        macrozone_params={},
        timeslots_macroareas={},
        hour_slots={},
        macroarea_id_map={},
        macroarea_name_map={},
        macroarea_label_map={},
        macroarea_timeslot_map={},
        macroarea_hourslot_map={},
    ),
)


def build_city_map(data: ZoneDictZoneDataMapping) -> CityMapData:
    """
    Build a city map from the provided JSON data.

    Args:
        data (dict): The JSON data containing the city map information.

    Returns:
        dict: The city map data structure.
    """

    def get_zone_label(zone: str) -> str:
        if zone == "all_map":
            return "All zones"
        return zone.replace("zone_", "Zone ").capitalize()

    def get_zone_id(zone: str) -> int:
        if zone == "all_map":
            return 0
        return int(zone.replace("zone_", "")) + 1

    zones = list(data.keys())

    assert "all_map" in zones

    zones = [zone for zone in zones if zone != "all_map"]

    zones = ["all_map"] + sorted(zones)

    assert all(zone.startswith("zone_") for zone in zones if zone != "all_map")

    id_zone_map = {get_zone_id(zone): zone for zone in zones}

    zone_name_map = {v: k for k, v in id_zone_map.items()}

    id_label_map = {
        zone_id: get_zone_label(zone) for zone_id, zone in id_zone_map.items()
    }

    zone_parkingmeter_map = {
        zone_id: data[zone]["parcometro"] for zone_id, zone in id_zone_map.items()
    }

    parkingmeter_zone_map = {
        parkingmeter: zone_id
        for zone_id, zone in id_zone_map.items()
        for parkingmeter in data[zone]["parcometro"]
    }
    parkingmeter_zone_map = {
        parkingmeter: parkingmeter_zone_map[parkingmeter]
        for parkingmeter in sorted(parkingmeter_zone_map)
    }

    slots = [s for zone in zones for s in data[zone]["stalli"]]

    assert set(slots) == set(data["all_map"]["stalli"])

    zone_slots_map = {
        zone_id: sorted(data[zone]["stalli"]) for zone_id, zone in id_zone_map.items()
    }

    slots_zone_map = {
        slot: [
            zone_id
            for zone_id, zone in id_zone_map.items()
            if slot in data[zone]["stalli"]
        ]
        for slot in slots
    }
    slots_zone_map = {
        slot: sorted(slots_zone_map[slot]) for slot in sorted(slots_zone_map)
    }

    roads = [s for zone in zones for s in data[zone]["strade"]]

    assert set(roads) == set(data["all_map"]["strade"])

    roads_names = [s for zone in zones for s in data[zone]["strade_name"]]

    n_roads = len(roads)
    n_roads_unique = len(set(roads))

    roads_names_pairs = list(zip(roads, roads_names, strict=False))
    assert len(roads_names_pairs) == n_roads

    # Check that the mapping is unique
    assert len(set(roads_names_pairs)) == n_roads_unique

    roads_name_map = dict(roads_names_pairs)

    zone_roads_map = {
        zone_id: sorted(data[zone]["strade"]) for zone_id, zone in id_zone_map.items()
    }

    road_zone_map = {
        road: [
            zone_id
            for zone_id, zone in id_zone_map.items()
            if road in data[zone]["strade"]
        ]
        for road in roads
    }
    road_zone_map = {
        road: sorted(road_zone_map[road]) for road in sorted(road_zone_map)
    }

    return CityMapData(
        area_id_zone_map=id_zone_map,
        area_name_map=zone_name_map,
        id_label_map=id_label_map,
        area_parkingmeter_map=zone_parkingmeter_map,
        parkingmeter_area_map=parkingmeter_zone_map,
        area_sensor_map=zone_slots_map,
        sensor_area_map=slots_zone_map,
        street_name_map=roads_name_map,
        area_street_map=zone_roads_map,
        street_area_map=road_zone_map,
    )


def build_hour_slots(hour_slots: list[int]) -> dict[int, HourSlotsData]:
    hour_slots_ranges = [
        [hour_slots[i], hour_slots[i + 1]] for i in range(len(hour_slots) - 1)
    ]

    hour_slots_s = [
        f"{start:02d}:00 - {end:02d}:00" for start, end in hour_slots_ranges
    ]

    pre_out = {
        i + 1: HourSlotsData(range=range, label=label)
        for i, (range, label) in enumerate(
            zip(hour_slots_ranges, hour_slots_s, strict=False)
        )
    }

    out = {0: HourSlotsData(range=None, label="All day"), **pre_out}

    return out


def build_available_dates(df_data: pd.DataFrame) -> AvailableDatesData:
    from typing import cast

    min_date = cast(
        datetime.date,
        df_data.index.min().date(),  # type: ignore
    )
    max_date = cast(
        datetime.date,
        df_data.index.max().date() - datetime.timedelta(days=6),  # type: ignore
    )

    return AvailableDatesData(min_date=min_date, max_date=max_date)


def build_macroarea_map(
    macrozone_data: dict[str, list[str]],
    timeslots_data: TimeSlotsMacroAreas,
    hour_slots_data: dict[int, HourSlotsData],
) -> MacroAreaMapData:
    macrozones = sorted(list(macrozone_data.keys()))
    assert sorted(list(timeslots_data.keys())) == macrozones
    macrozones_ids = list(range(len(macrozones)))

    macrozone_id_map = dict(zip(macrozones, macrozones_ids, strict=True))
    macrozone_name_map = dict(zip(macrozones_ids, macrozones, strict=True))

    macrozones_labels = ["Macroarea " + s.split("_")[1] for s in macrozones]
    macroarea_label_map = dict(zip(macrozones_ids, macrozones_labels, strict=True))

    assert all(len(timeslots_data[macrozone]) == 7 for macrozone in macrozones)

    timeslot_map: dict[str, dict[tuple[int, int], bool]] = {}
    for macrozone in macrozones:
        timezone_data = timeslots_data[macrozone]
        out_timeslots: dict[tuple[int, int], bool] = {}
        for weekday, timeslots in enumerate(timezone_data):
            payment_hours = [
                q for timeslot in timeslots for q in range(timeslot[0], timeslot[1])
            ]
            for hour in range(24):
                out_timeslots[weekday, hour] = hour in payment_hours
        timeslot_map[macrozone] = out_timeslots

    hourslot_map: dict[str, dict[tuple[int, int], bool]] = {}
    for macrozone, out_timeslots in timeslot_map.items():
        out_hourslots = {}
        for weekday in range(7):
            for i_hourslot, hourslot_data in hour_slots_data.items():
                hourslot_range = hourslot_data["range"]
                out_hourslots[weekday, i_hourslot] = hourslot_range is None or any(
                    out_timeslots[(weekday, hour)]
                    for hour in range(hourslot_range[0], hourslot_range[1])
                )
        hourslot_map[macrozone] = out_hourslots

    return MacroAreaMapData(
        macrozone_params=macrozone_data,
        timeslots_macroareas=timeslots_data,
        hour_slots=hour_slots_data,
        macroarea_id_map=macrozone_id_map,
        macroarea_name_map=macrozone_name_map,
        macroarea_label_map=macroarea_label_map,
        macroarea_timeslot_map=timeslot_map,
        macroarea_hourslot_map=hourslot_map,
    )


# Load data function
def load_data() -> None:
    """
    Load the distribution data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_files(
        settings.DATA_DIR,
        _MODULE_NAME,
        _PKL_FILES_DATA,
        _JSON_FILES_DATA,
    )

    data_store.update(**_postprocess(out_data))
