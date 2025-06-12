from pathlib import Path
from typing import Any

from api.general.utils.loading import load_startup_files

from .data import DistanceData, PreDistanceData, WhatIfLoadedData, WhatIfScenarioData
from .scenarios import SCENARIOS
from .startup_data import STARTUP_DATA as _STARTUP_DATA


def preprocess_distances_p(data: PreDistanceData) -> DistanceData:
    return {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}


def preprocess_distances_s(data: PreDistanceData) -> DistanceData:
    return {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}


def postprocess(data: dict[str, Any]) -> WhatIfLoadedData:
    from typing import cast

    out_data_scenarios: dict[str, WhatIfScenarioData] = {}

    for scenario in SCENARIOS:
        out_data_scenarios[scenario] = WhatIfScenarioData(
            dict_data=data.pop("dict_data__" + scenario)
        )
        if scenario == "2nd":
            out_data_scenarios[scenario]["s_coordinates"] = data.pop(
                "s_coordinates__2nd"
            )
            out_data_scenarios[scenario]["s_scaler"] = data.pop("s_scaler__2nd")
        else:
            assert f"s_coordinates__{scenario}" not in data, (
                f"Unexpected key 's_coordinates__{scenario}' in data"
            )
            assert f"s_scaler__{scenario}" not in data, (
                f"Unexpected key 's_scaler__{scenario}' in data"
            )

    distances_p = preprocess_distances_p(cast(PreDistanceData, data["distances_p"]))
    distances_s = preprocess_distances_s(cast(PreDistanceData, data["distances_s"]))

    return WhatIfLoadedData(
        scenarios=out_data_scenarios,
        dict_zone=data["dict_zone"],
        distances_p=distances_p,
        distances_s=distances_s,
        p_coordinates=data["p_coordinates"],
        p_scaler=data["p_scaler"],
        s_coordinates=data["s_coordinates"],
        s_scaler=data["s_scaler"],
    )


def load_files(data_path: Path) -> WhatIfLoadedData:
    out_data = load_startup_files(
        data_path,
        _STARTUP_DATA,
    )

    return postprocess(out_data)
