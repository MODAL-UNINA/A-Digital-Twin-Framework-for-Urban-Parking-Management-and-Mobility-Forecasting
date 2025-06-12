from pathlib import Path
from typing import Any, TypedDict

import matplotlib

from api.general.utils.error_status import ErrorStatus
from api.general.utils.image import get_base64_image
from api.general.views import (
    get_area_id,
    get_hour_slots_items,
    get_zone_dict,
)

from .data import (
    DistanceData,
    WhatIfDataDictMapping,
    WhatIfLoadedData,
)

matplotlib.use("agg")

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.http import HttpRequest

from .backend import (
    FloatArray,
    GenerationData,
    create_cumulative_plot,
    create_heatmap,
    create_histograms_with_inset,
    create_radar_chart_map,
    get_generation,
    get_gif,
    prepare_generated_data,
)
from .startup import whatif_data_store


def get_available_whatif_date_range(data_dict: WhatIfDataDictMapping) -> dict[str, str]:
    dates_split = [s.split(" - ") for s in data_dict.keys()]

    return dict(
        min_date=min([v[0] for v in dates_split]),
        max_date=max([v[1] for v in dates_split]),
    )


def get_whatif_data() -> WhatIfLoadedData:
    return whatif_data_store["data"]


def get_whatif_data_path() -> Path:
    return whatif_data_store["data_path"]


def get_whatif_data_dict(scenario: str) -> WhatIfDataDictMapping:
    return get_whatif_data()["scenarios"][scenario]["dict_data"]


def get_distances_p() -> DistanceData:
    return get_whatif_data()["distances_p"]


def get_distances_s() -> DistanceData:
    return get_whatif_data()["distances_s"]


def get_available_whatif_scenario_dates(scenario: str) -> dict[str, str]:
    data_dict = get_whatif_data_dict(scenario)
    return get_available_whatif_date_range(data_dict)


def get_quantity(quantity_s: str | None) -> int | None:
    if quantity_s is None:
        return None

    return int(quantity_s)


class GenerationResult(TypedDict):
    selected_zones: list[int]
    start_date: str
    end_date: str


def run_generation(
    request: HttpRequest,
    scenario: str,
    zone_name: str,
    date: pd.Timestamp,
    quantity: int | None,
) -> GenerationResult | ErrorStatus:
    from typing import cast

    zone_dict = get_zone_dict()
    zones = [zone for zone in zone_dict.keys() if zone != "all_map"]

    if scenario == "3rd" and zone_name != "all_map":
        return ErrorStatus(error="Invalid zone")

    if scenario != "3rd" and zone_name not in zones:
        return ErrorStatus(error="Invalid zone")

    head_msg = f"User:{request.user}, {request.session.session_key}"

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - loading data")

    data = get_whatif_data()

    data_path = get_whatif_data_path()
    distances_p = get_distances_p()
    distances_s = get_distances_s()

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - generating data")
    import torch.multiprocessing as mp

    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(
        target=get_generation,
        args=(
            return_dict,
            scenario,
            date,
            zone_name,
            quantity,
            data_path,
            data,
            zone_dict,
            zones,
            distances_p,
            distances_s,
        ),
    )
    p.start()
    p.join()

    msg: str
    res, msg = return_dict["generation"]

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - cleaning up")

    if msg:
        return ErrorStatus(error=msg)

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - parsing generation")
    (key_found, output) = res

    date_str = date.strftime("%Y-%m-%d")

    data_dict = data["scenarios"][scenario]["dict_data"]

    data_key = data_dict[key_found]

    data_real_t = data_key["data"]
    data_real_t = data_real_t.permute(0, 2, 1, 3, 4)
    data_real = cast(
        FloatArray,
        data_real_t.numpy(),  # type: ignore
    )

    if scenario != "3rd":
        dict_zone = data["dict_zone"]
        dictionary = dict_zone[zone_name]
    else:
        dictionary = ["all_map"] + zones

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - preparing generated data")
    out_data = prepare_generated_data(
        scenario,
        zone_name,
        output,
        data_real,
        dictionary,
        data,
    )

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - converting data to compatible format")
    assert all(
        key in ["selected_zones"]
        or val is None
        or isinstance(val, (pd.DataFrame, np.ndarray))
        for key, val in out_data.items()
    )

    out_data_json: dict[str, Any] = {}

    for key, value in out_data.items():
        if key in ["selected_zones"] or value is None:
            out_data_json[key] = value
            continue

        if isinstance(value, np.ndarray):
            out_data_json[key] = value.tolist()
        else:
            assert isinstance(value, pd.DataFrame)
            out_data_json[key] = value.to_json(  # type: ignore
            )

    saved_out_key = "__".join(
        [scenario, date_str, zone_name, "null" if quantity is None else str(quantity)]
    )

    ans_time = f"{datetime.now()}"
    print(f"[{ans_time}] {head_msg} - saving data as {saved_out_key}")

    saved_out_data = dict(
        key_found=key_found,
        out_data=out_data_json,
        start_date=data_key["start_date"],
        end_date=data_key["end_date"],
    )

    if request.session.get("generator_storage") is None:
        request.session["generator_storage"] = {}
        print(f"{head_msg} - created")

    storage_data = request.session["generator_storage"]

    print(f"{head_msg} - current content:")
    for saved_key in storage_data:
        print(f"{head_msg} - - {saved_key}")

    if saved_out_key in storage_data:
        print(f"{head_msg} - {saved_out_key} found, overwriting")
    else:
        print(f"{head_msg} - {saved_out_key} not found, appending")

    storage_data[saved_out_key] = saved_out_data
    request.session["generator_storage"] = storage_data
    ans_time = f"{datetime.now()}"
    print(
        f"[{ans_time}] {head_msg} - updated content: {list(request.session['generator_storage'].keys())}"
    )

    selected_zones = out_data["selected_zones"]

    # res = {}

    # res["selected_zones"] = [get_area_id(zone) for zone in selected_zones]
    # res["start_date"] = data_key["start_date"]
    # res["end_date"] = data_key["end_date"]

    return GenerationResult(
        selected_zones=[get_area_id(zone) for zone in selected_zones],
        start_date=data_key["start_date"],
        end_date=data_key["end_date"],
    )


class SavedData(TypedDict):
    key_found: str
    out_data: GenerationData
    start_date: str
    end_date: str


def get_saved_generation(
    request: HttpRequest,
    scenario: str,
    zone_name: str,
    date: pd.Timestamp,
    quantity: int | None,
) -> SavedData | None:
    head_msg = f"User:{request.user}, {request.session.session_key}"

    storage_data = request.session.get("generator_storage", None)
    if storage_data is None:
        print(f"{head_msg} - empty")
        return None

    date_str = date.strftime("%Y-%m-%d")

    data_key = "__".join(
        [scenario, date_str, zone_name, "null" if quantity is None else str(quantity)]
    )
    if data_key not in storage_data:
        print(f"{head_msg} - {data_key} not found within {list(storage_data.keys())}")
        return None

    saved_data = storage_data[data_key]

    res: dict[str, Any] = {}
    for key in saved_data.keys():
        if key != "out_data":
            res[key] = saved_data[key]
            continue

        out_data_json = saved_data["out_data"]

        out_data: dict[str, Any] = {}
        for key, value in out_data_json.items():
            if key in ["selected_zones"] or out_data_json[key] is None:
                out_data[key] = value
            else:
                if isinstance(value, list):
                    out_data[key] = np.asarray(
                        value  # type: ignore
                    )
                else:
                    out_data[key] = pd.read_json(  # type: ignore
                        value, typ="frame"
                    )

        res["out_data"] = GenerationData(**out_data)

    return SavedData(**res)


def get_whatif_heatmaps(
    request: HttpRequest,
    scenario: str,
    zone_name: str,
    date: pd.Timestamp,
    quantity: int | None,
    kind: str,
    selected_day: str,
) -> ErrorStatus | dict[str, str]:
    saved_data = get_saved_generation(request, scenario, zone_name, date, quantity)
    if saved_data is None:
        return ErrorStatus(error="Data not found. Please generate it first.")

    start_date = saved_data["start_date"]
    out_data = saved_data["out_data"]
    hour_slots = {
        int(k) - 1: v for k, v in get_hour_slots_items().items() if int(k) != 0
    }

    out_real = create_heatmap(
        scenario, start_date, out_data, hour_slots, kind, "real", selected_day
    )

    if isinstance(out_real, dict):
        return out_real
    figs_real = out_real

    figs_real, img_str_real = get_gif(figs_real)

    for fig in figs_real:
        plt.close(fig)

    out_gen = create_heatmap(
        scenario, start_date, out_data, hour_slots, kind, "gen", selected_day
    )

    if isinstance(out_gen, dict):
        return out_gen
    figs_gen = out_gen

    figs_gen, img_str_gen = get_gif(figs_gen)

    for fig in figs_gen:
        plt.close(fig)

    return dict(real=img_str_real, gen=img_str_gen)


def get_whatif_distributions(
    request: HttpRequest,
    scenario: str,
    zone_name: str,
    date: pd.Timestamp,
    quantity: int | None,
    kind: str,
) -> ErrorStatus | dict[str, str]:
    saved_data = get_saved_generation(request, scenario, zone_name, date, quantity)
    if saved_data is None:
        return ErrorStatus(error="Data not found. Please generate it first.")

    out_data = saved_data["out_data"]

    out_hist = create_histograms_with_inset(scenario, out_data, kind)

    if isinstance(out_hist, dict):
        return out_hist
    fig_hist = out_hist

    fig_hist, img_str_hist = get_base64_image(fig_hist)
    plt.close(fig_hist)

    zone_dict = get_zone_dict()

    out_radar = create_radar_chart_map(scenario, out_data, kind, zone_dict)

    if isinstance(out_radar, dict):
        return out_radar
    fig_radar = out_radar

    fig_radar, img_str_radar = get_base64_image(fig_radar)
    plt.close(fig_radar)

    return dict(hist=img_str_hist, radar=img_str_radar)


def get_whatif_cumulative_plot(
    request: HttpRequest,
    scenario: str,
    zone_name: str,
    date: pd.Timestamp,
    quantity: int | None,
    kind: str,
    selected_adjacent_zone: str,
) -> ErrorStatus | dict[str, str]:
    saved_data = get_saved_generation(request, scenario, zone_name, date, quantity)
    if saved_data is None:
        return ErrorStatus(error="Data not found. Please generate it first.")

    start_date = saved_data["start_date"]
    out_data = saved_data["out_data"]
    zone_dict = get_zone_dict()

    out = create_cumulative_plot(
        scenario, start_date, out_data, kind, selected_adjacent_zone, zone_dict
    )

    if isinstance(out, dict):
        return out
    fig = out

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return dict(plot=img_str)
