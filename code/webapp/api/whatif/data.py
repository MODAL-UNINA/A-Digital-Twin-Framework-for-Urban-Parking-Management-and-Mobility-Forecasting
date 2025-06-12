from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from sklearn.pipeline import Pipeline


class WhatIfDataDict(TypedDict):
    cond: torch.Tensor
    data: torch.Tensor
    start_date: str
    end_date: str


WhatIfDataDictMapping = dict[str, WhatIfDataDict]
WhatIfPCoordinatesMapping = dict[int, tuple[int, int]]
WhatIfSCoordinatesMapping = dict[int, tuple[int, int]]


class WhatIfScenarioData(TypedDict):
    dict_data: WhatIfDataDictMapping
    s_coordinates: NotRequired[WhatIfSCoordinatesMapping]
    s_scaler: NotRequired[Pipeline]


WhatIfVicinityData = dict[str, list[str]]


PreDistanceData = dict[str, dict[str, float]]
DistanceData = dict[int, dict[int, float]]


class WhatIfLoadedData(TypedDict):
    scenarios: dict[str, WhatIfScenarioData]
    dict_zone: WhatIfVicinityData
    distances_p: DistanceData
    distances_s: DistanceData
    p_coordinates: WhatIfPCoordinatesMapping
    p_scaler: Pipeline
    s_coordinates: WhatIfSCoordinatesMapping
    s_scaler: Pipeline


class WhatIfData(TypedDict):
    data: WhatIfLoadedData
    data_path: Path
