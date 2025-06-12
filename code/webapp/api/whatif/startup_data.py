from api.general.utils.startup_data import (
    CSVMapping,
    JsonMapping,
    PklMapping,
    StartupData,
)

from .scenarios import SCENARIOS

STARTUP_DATA = StartupData(
    module_name="WhatIf",
    pkl_files_data=PklMapping(
        p_coordinates="p_coords.pkl",
        p_scaler="p_scaler.pkl",
        s_coordinates="s_coords.pkl",
        s_scaler="s_scaler.pkl",
        **{
            "dict_data__" + scenario: f"{scenario}/input/dict_data.pkl"
            for scenario in SCENARIOS
        },
        s_coordinates__2nd="2nd/input/s_coords.pkl",
        s_scaler__2nd="2nd/input/s_scaler.pkl",
    ),
    json_files_data=JsonMapping(
        distances_p="distances_p.json",
        distances_s="distances_s.json",
        dict_zone="vicinity_zone_map.json",
    ),
    csv_files_data=CSVMapping(),
)
