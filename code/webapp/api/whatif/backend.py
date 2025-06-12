from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("agg")

import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common.generation.models import Encoder, Generator
from common.generation.models import ModelArgs as TorchModelArgs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from torch import nn

from api.general.startup import ZoneDictZoneDataMapping
from api.general.utils.error_status import ErrorStatus
from api.whatif.data import (
    WhatIfDataDictMapping,
    WhatIfLoadedData,
    WhatIfPCoordinatesMapping,
    WhatIfSCoordinatesMapping,
)
from api.whatif.scenarios import SCENARIOS

FloatData = np.float32
FloatArray = NDArray[FloatData]


def get_p_coordinates(
    data: WhatIfLoadedData, _scenario: str
) -> WhatIfPCoordinatesMapping:
    return data["p_coordinates"]


def get_p_scaler(data: WhatIfLoadedData, _scenario: str) -> Pipeline:
    return data["p_scaler"]


def get_s_coordinates(
    data: WhatIfLoadedData, scenario: str
) -> WhatIfSCoordinatesMapping:
    if scenario != "2nd":
        return data["s_coordinates"]

    scenario_data = data["scenarios"]["2nd"]
    assert "s_coordinates" in scenario_data, (
        "Expected 's_coordinates' in scenario data for '2nd', "
        f"but got {list(scenario_data.keys())}"
    )

    return scenario_data["s_coordinates"]


def get_s_scaler(data: WhatIfLoadedData, scenario: str) -> Pipeline:
    if scenario != "2nd":
        return data["s_scaler"]

    scenario_data = data["scenarios"]["2nd"]
    assert "s_scaler" in scenario_data, (
        "Expected 's_scaler' in scenario data for '2nd', "
        f"but got {list(scenario_data.keys())}"
    )

    return scenario_data["s_scaler"]


def generation(
    encoder: nn.Module, generator: nn.Module, mask: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        z_gen, _, _, indices, map = encoder(mask.to(device))
        noise = torch.randn(z_gen.shape[0], 100).to(device)
        z_gen = torch.cat([noise, z_gen], dim=1)
        output = generator(z_gen, map, indices)
    return output


def get_key_for_date(date: str, dict_data: WhatIfDataDictMapping) -> str | None:
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    for key in dict_data.keys():
        start_date, end_date = key.split(" - ")
        start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_obj = datetime.strptime(end_date, "%Y-%m-%d")
        if start_obj <= date_obj <= end_obj:
            return key
    return None


def load_models(
    data_path: Path, scenario: str, device: torch.device
) -> tuple[Generator, Encoder]:
    if scenario not in SCENARIOS:
        raise ValueError("Invalid scenario")

    latent_dim = 100
    kernel_size = 3
    padding = 1
    input_dim = 2 if scenario != "2nd" else 1
    cond_dim = 2 if scenario != "3rd" else 3
    hidden_dim = 32 if scenario != "3rd" else 64
    horizon = 6 * 7
    grid_size = 100

    model_args = TorchModelArgs(
        input_dim=input_dim,
        cond_dim=cond_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        padding=padding,
        horizon=horizon,
        grid_size=grid_size,
        use_proximity=False,
    )

    generator = Generator(model_args).to(device)
    encoder = Encoder(model_args).to(device)

    models_path = data_path / f"{scenario}" / "models"
    generator.load_state_dict(
        torch.load(  # type: ignore
            models_path / "generator.pth",
            map_location=device,
            weights_only=True,
        )
    )
    encoder.load_state_dict(
        torch.load(  # type: ignore
            models_path / "encoder_cond.pth",
            map_location=device,
            weights_only=True,
        )
    )
    generator.eval()
    encoder.eval()

    return generator, encoder


def get_generation(
    return_dict: "DictProxy[str, Any]",
    scenario: str,
    date: pd.Timestamp,
    zone_name: str,
    quantity: int | None,
    models_dir: Path,
    data: WhatIfLoadedData,
    zone_dict: ZoneDictZoneDataMapping,
    zones: list[str],
    distances_p: dict[int, dict[int, float]],
    distances_s: dict[int, dict[int, float]],
) -> None:
    import os
    from typing import cast

    # Seed params
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    date_to_check = date.strftime("%Y-%m-%d")

    scenario_data = data["scenarios"][scenario]

    data_dict = scenario_data["dict_data"]

    key_found = get_key_for_date(date_to_check, data_dict)
    if not key_found:
        return_dict["generation"] = None, "Invalid date"
        return

    mask = data_dict[key_found]["cond"].clone()
    mask = mask.permute(0, 2, 1, 3, 4)

    if quantity is None:
        if scenario == "2nd":
            quantity = 150

    p_keys_to_remove: list[int] = []
    s_keys_to_remove: list[int] = []

    coordinates_parkingmeters = data["p_coordinates"]
    coordinates_slots = data["s_coordinates"]

    if scenario == "1st":
        adjacent_zones = [zone for zone in zones if zone != zone_name]

        p_keys_to_remove = zone_dict[zone_name]["parcometro"]
        p_keys_to_remove = [int(key) for key in p_keys_to_remove]
        s_keys_to_remove = zone_dict[zone_name]["stalli"]
        s_keys_to_remove = [int(key) for key in s_keys_to_remove]

        p_remaining = [zone_dict[z]["parcometro"] for z in adjacent_zones]
        p_remaining = [item for sublist in p_remaining for item in sublist]
        s_remaining = [zone_dict[z]["stalli"] for z in adjacent_zones]
        s_remaining = [item for sublist in s_remaining for item in sublist]
        delta_p = pd.Series(
            {
                key: sum([np.exp(-distances_p[key][key2]) for key2 in p_keys_to_remove])
                for key in p_remaining
            }
        )

        rho_p = delta_p / (delta_p.max() + 1e-6)

        delta_s = pd.Series(
            {
                key: sum([np.exp(-distances_s[key][key2]) for key2 in s_keys_to_remove])
                for key in s_remaining
            }
        )

        rho_s = delta_s / (delta_s.max() + 1e-6)

        for z in adjacent_zones:
            p_adjust = zone_dict[z]["parcometro"]
            s_adjust = zone_dict[z]["stalli"]

            for p in p_adjust:
                p_mask_zero = (
                    mask[
                        0,
                        1,
                        :,
                        coordinates_parkingmeters[p][0],
                        coordinates_parkingmeters[p][1],
                    ]
                    == 0
                )
                mask[
                    0,
                    1,
                    :,
                    coordinates_parkingmeters[p][0],
                    coordinates_parkingmeters[p][1],
                ] = torch.from_numpy(  # type: ignore
                    np.where(
                        p_mask_zero,
                        0,
                        mask[
                            0,
                            1,
                            :,
                            coordinates_parkingmeters[p][0],
                            coordinates_parkingmeters[p][1],
                        ]
                        + rho_p[p],
                    )
                ).float()

            for s in s_adjust:
                s_mask_zero = (
                    mask[0, 0, :, coordinates_slots[s][0], coordinates_slots[s][1]] == 0
                )
                mask[0, 0, :, coordinates_slots[s][0], coordinates_slots[s][1]] = (
                    torch.from_numpy(  # type: ignore
                        np.where(
                            s_mask_zero,
                            0,
                            mask[
                                0,
                                0,
                                :,
                                coordinates_slots[s][0],
                                coordinates_slots[s][1],
                            ]
                            + rho_s[s],
                        )
                    ).float()
                )

        for key in p_keys_to_remove:
            lat, lon = coordinates_parkingmeters[key]
            mask[0, 1, :, lat, lon] = 0

        for key in s_keys_to_remove:
            lat, lon = coordinates_slots[key]
            mask[0, 0, :, lat, lon] = 0
    if scenario == "2nd":
        s_keys = zone_dict[zone_name]["stalli"]
        s_keys = [int(key) for key in s_keys]

        for key in s_keys:
            lat, lon = coordinates_slots[key]
            mask[0, 1, :, lat, lon] += quantity

        lat_indices = [lat for lat, _ in coordinates_slots.values()]
        lon_indices = [lon for _, lon in coordinates_slots.values()]

        mask_zero = mask[:, 1, :, lat_indices, lon_indices] == 0
        mask[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(  # type: ignore
            np.where(
                mask_zero,
                0,
                torch.log(1 / (mask[:, 1, :, lat_indices, lon_indices] + 1)),
            )
        ).float()
    if scenario == "3rd":
        for slot in coordinates_slots.keys():
            lat, lon = coordinates_slots[slot][0], coordinates_slots[slot][1]
            mask[0, 2, :18, lat, lon] = 1
            mask[0, 2, 18:, lat, lon] = 0
        for park in coordinates_parkingmeters.keys():
            lat, lon = (
                coordinates_parkingmeters[park][0],
                coordinates_parkingmeters[park][1],
            )
            mask[0, 2, :18, lat, lon] = 1
            mask[0, 2, 18:, lat, lon] = 0

    generator, encoder = load_models(models_dir, scenario, device)
    output_t = generation(encoder, generator, mask, device)

    if scenario == "1st":
        for k in p_keys_to_remove:
            lat, lon = coordinates_parkingmeters[k]
            output_t[0, 1, :, lat, lon] = 0
        for k in s_keys_to_remove:
            lat, lon = coordinates_slots[k]
            output_t[0, 0, :, lat, lon] = 0
    output = cast(
        FloatArray,
        output_t.detach().cpu().numpy(),  # type: ignore
    )

    return_dict["generation"] = (key_found, output), None


def get_dfs_parkingmeter(
    data_real: FloatArray,
    output: FloatArray,
    coordinates_parkingmeters: WhatIfPCoordinatesMapping,
    scaler_parkingmeters: Pipeline,
) -> tuple[
    FloatArray,
    FloatArray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    from typing import cast

    p_data = cast(FloatArray, data_real[0, 1])
    p_gen = cast(FloatArray, output[0, 1])
    data_p: list[FloatArray] = []
    gen_p: list[FloatArray] = []
    for key in coordinates_parkingmeters.keys():
        lat, lon = coordinates_parkingmeters[key]
        data_p.append(p_data[:, lat, lon])
        gen_p.append(p_gen[:, lat, lon])
    final_data_p = np.stack(data_p, axis=1)
    final_gen_p = np.stack(gen_p, axis=1)
    p_data_df = pd.DataFrame(
        final_data_p, columns=list(coordinates_parkingmeters.keys())
    )
    p_gen_df = pd.DataFrame(final_gen_p, columns=list(coordinates_parkingmeters.keys()))
    p_data_df.columns = [int(i) for i in coordinates_parkingmeters.keys()]
    p_gen_df.columns = [int(i) for i in coordinates_parkingmeters.keys()]
    p_data_df = p_data_df.reindex(  # type: ignore
        sorted(p_data_df.columns), axis=1
    )
    p_gen_df = p_gen_df.reindex(  # type: ignore
        sorted(p_gen_df.columns), axis=1
    )

    p_data_df_old = p_data_df.copy(deep=True)
    p_gen_df_old = p_gen_df.copy(deep=True)
    p_data_df = pd.DataFrame(
        scaler_parkingmeters.inverse_transform(  # type: ignore
            p_data_df
        ),
        columns=p_data_df.columns,
    )
    p_data_df = p_data_df.round(  # type: ignore
        0
    )
    p_gen_df = pd.DataFrame(
        scaler_parkingmeters.inverse_transform(  # type: ignore
            p_gen_df
        ),
        columns=p_gen_df.columns,
    )

    return p_data, p_gen, p_data_df, p_gen_df, p_data_df_old, p_gen_df_old


def get_dfs_parkingslot(
    data_real: FloatArray,
    output: FloatArray,
    coordinates_slots: WhatIfSCoordinatesMapping,
    scaler_slots: Pipeline,
) -> tuple[
    FloatArray, FloatArray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    from typing import cast

    s_data = cast(FloatArray, data_real[0, 0])
    s_gen = cast(FloatArray, output[0, 0])
    data_s: list[FloatArray] = []
    gen_s: list[FloatArray] = []
    for key in coordinates_slots.keys():
        lat, lon = coordinates_slots[key]
        data_s.append(s_data[:, lat, lon])
        gen_s.append(s_gen[:, lat, lon])
    final_data_s = np.stack(data_s, axis=1)
    final_gen_s = np.stack(gen_s, axis=1)
    s_data_df = pd.DataFrame(
        final_data_s, columns=[int(i) for i in coordinates_slots.keys()]
    )
    s_gen_df = pd.DataFrame(
        final_gen_s, columns=[int(i) for i in coordinates_slots.keys()]
    )
    s_data_df = s_data_df.reindex(  # type: ignore
        sorted(s_data_df.columns), axis=1
    )
    s_gen_df = s_gen_df.reindex(  # type: ignore
        sorted(s_gen_df.columns), axis=1
    )

    s_data_df_old = s_data_df.copy(deep=True)
    s_gen_df_old = s_gen_df.copy(deep=True)
    s_data_df = pd.DataFrame(
        scaler_slots.inverse_transform(  # type: ignore
            s_data_df
        ),
        columns=s_data_df.columns,
    )
    s_data_df = s_data_df.round(  # type: ignore
        0
    )
    s_gen_df = pd.DataFrame(
        scaler_slots.inverse_transform(  # type: ignore
            s_gen_df
        ),
        columns=s_gen_df.columns,
    )

    return s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old


class GenerationData(TypedDict):
    selected_zones: list[str]
    p_data: FloatArray | None
    p_gen: FloatArray | None
    p_data_df: pd.DataFrame | None
    p_gen_df: pd.DataFrame | None
    p_data_df_old: pd.DataFrame | None
    p_gen_df_old: pd.DataFrame | None
    s_data: FloatArray | None
    s_gen: FloatArray | None
    s_data_df: pd.DataFrame | None
    s_gen_df: pd.DataFrame | None
    s_data_df_old: pd.DataFrame | None
    s_gen_df_old: pd.DataFrame | None


def prepare_generated_data(
    scenario: str,
    zone_name: str,
    output: FloatArray,
    data_real: FloatArray,
    dictionary: list[str],
    data: WhatIfLoadedData,
) -> GenerationData:
    assert scenario in SCENARIOS

    p_data = None
    p_gen = None
    s_data = None
    s_gen = None
    p_data_df = None
    p_gen_df = None
    p_data_df_old = None
    p_gen_df_old = None
    s_data_df = None
    s_gen_df = None
    s_data_df_old = None
    s_gen_df_old = None

    p_coordinates = data["p_coordinates"]
    s_coordinates = data["s_coordinates"]
    p_scaler = data["p_scaler"]
    s_scaler = data["s_scaler"]

    if scenario == "1st":
        selected_zones = dictionary
        p_data, p_gen, p_data_df, p_gen_df, p_data_df_old, p_gen_df_old = (
            get_dfs_parkingmeter(
                data_real,
                output,
                p_coordinates,
                p_scaler,
            )
        )

        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )

    elif scenario == "2nd":
        selected_zones = [zone_name] + dictionary
        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )

    elif scenario == "3rd":
        selected_zones = dictionary
        s_data, s_gen, s_data_df, s_gen_df, s_data_df_old, s_gen_df_old = (
            get_dfs_parkingslot(
                data_real,
                output,
                s_coordinates,
                s_scaler,
            )
        )
    else:
        raise ValueError("Invalid scenario")

    return GenerationData(
        selected_zones=selected_zones,
        p_data=p_data,
        p_gen=p_gen,
        p_data_df=p_data_df,
        p_gen_df=p_gen_df,
        p_data_df_old=p_data_df_old,
        p_gen_df_old=p_gen_df_old,
        s_data=s_data,
        s_gen=s_gen,
        s_data_df=s_data_df,
        s_gen_df=s_gen_df,
        s_data_df_old=s_data_df_old,
        s_gen_df_old=s_gen_df_old,
    )


def get_gif(figs: list[Figure]) -> tuple[list[Figure], str]:
    import base64
    import io

    from PIL import Image
    from PIL.ImageFile import ImageFile

    images: list[ImageFile] = []

    for fig in figs:
        buffer = io.BytesIO()

        fig.savefig(  # type: ignore
            buffer, format="png", bbox_inches="tight", dpi=100
        )
        buffer.seek(0)

        image = Image.open(buffer)

        images.append(image)

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        save_all=True,
        append_images=images[1:],
        format="GIF",
        duration=500,
        loop=0,
        transparency=0,
        disposal=2,
    )

    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return figs, img_str


def create_heatmap(
    scenario: str,
    start_date: str,
    out_data: GenerationData,
    hour_slots: dict[int, str],
    kind: str,
    which_data: str,
    selected_day: str,
) -> list[Figure] | ErrorStatus:
    """
    Creates heatmap plots for parking data with a yellow color intensity for high values and dark color for low values.
    Adds a color bar to each plot, and saves the images.
    """

    from scipy.ndimage import gaussian_filter  # type: ignore

    def apply_aura_effect(data: FloatArray, sigma: float = 1.5) -> FloatArray:
        """Apply Gaussian blur to simulate aura effect"""
        from typing import cast

        return cast(
            FloatArray,
            gaussian_filter(  # type: ignore
                data, sigma=sigma
            ),
        )

    from matplotlib.colors import LinearSegmentedColormap

    colors = [(0, "white"), (1, "darkblue")]
    custom_cmap = LinearSegmentedColormap.from_list("white_to_darkblue", colors, N=256)

    if scenario not in SCENARIOS:
        return ErrorStatus(error="Invalid scenario")

    if scenario == "1st":
        if kind not in ["parkingmeter", "parkingslot"]:
            return ErrorStatus(error="Invalid kind")
    if scenario == "2nd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")
    if scenario == "3rd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")

    start_date_date = pd.Timestamp(start_date)
    selected_day_date = pd.Timestamp(selected_day)

    idx_start_data = (selected_day_date - start_date_date).days * 6
    idx_end_data = idx_start_data + 6

    data_plot_s = ""

    data_plot_s += "p_" if kind == "parkingmeter" else "s_"
    data_plot_s += "data" if which_data == "real" else "gen"

    data_plot = out_data[data_plot_s]
    assert data_plot is not None, "Data plot is None"

    figs: list[Figure] = []

    for it, t in enumerate(range(idx_start_data, idx_end_data)):
        fig, ax = plt.subplots()  # type: ignore
        aura_data = apply_aura_effect(data_plot[t], sigma=1.5)
        ax.imshow(  # type: ignore
            data_plot[t], cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=1
        )
        ax.imshow(  # type: ignore
            aura_data, cmap="Blues", interpolation="nearest", alpha=0.4
        )
        ax.invert_yaxis()
        ax.set_xticks(  # type: ignore
            []
        )
        ax.set_yticks(  # type: ignore
            []
        )
        ax.set_title(  # type: ignore
            f"Hour slot: {hour_slots[it]}"
        )
        fig.set_size_inches(10, 6)

        figs.append(fig)

    return figs


def create_histograms_with_inset(
    scenario: str, out_data: GenerationData, kind: str
) -> Figure | ErrorStatus:
    """
    Plot a histogram with an inset t-SNE visualization for parking meter data.
    """
    from typing import cast

    def flatten_data(df: pd.DataFrame) -> FloatArray:
        from typing import cast

        return cast(
            FloatArray,
            df.to_numpy().flatten(),  # type: ignore
        )

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore
    from sklearn.manifold import TSNE

    if scenario == "1st":
        if kind not in ["parkingmeter", "parkingslot"]:
            return ErrorStatus(error="Invalid kind")
    if scenario == "2nd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")
    if scenario == "3rd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"
    real_s = data_plot_s + "data_df_old"
    gen_s = data_plot_s + "gen_df_old"

    df_real_data = out_data[real_s]
    df_gen_data = out_data[gen_s]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    real_data_plot = flatten_data(df_real_data)
    gen_data_plot = flatten_data(df_gen_data)

    bin_edges = list(
        np.linspace(
            0,
            max(np.max(real_data_plot), np.max(gen_data_plot)),
            20,
        )
    )

    fig, ax_main = plt.subplots()  # type: ignore

    ax_main.hist(  # type: ignore
        real_data_plot,
        bins=bin_edges,
        alpha=0.5,
        color="orange",
        label="Real Data",
        density=True,
    )
    ax_main.hist(  # type: ignore
        gen_data_plot,
        bins=bin_edges,
        alpha=0.5,
        color="green",
        label="Generated Data",
        density=True,
    )
    ax_main.set_xlim(0.0, 1.1)

    tsne = TSNE(n_components=2, random_state=42)
    p_combined_data = cast(
        FloatArray,
        np.vstack(
            (
                df_real_data.values,  # type: ignore
                df_gen_data.values,  # type: ignore
            )
        ),
    )
    p_tsne = tsne.fit_transform(  # type: ignore
        p_combined_data
    )

    inset_ax = cast(
        Axes,
        inset_axes(  # type: ignore
            parent_axes=ax_main,
            width="45%",
            height="45%",
            borderpad=1,
        ),
    )
    inset_ax.scatter(  # type: ignore
        p_tsne[: len(df_real_data), 0],  # type: ignore
        p_tsne[: len(df_real_data), 1],  # type: ignore
        s=10,
        color="orange",
        label="Real Data",
        alpha=0.5,
    )
    inset_ax.scatter(  # type: ignore
        p_tsne[len(df_real_data) :, 0],  # type: ignore
        p_tsne[len(df_real_data) :, 1],  # type: ignore
        s=10,
        color="green",
        label="Generated Data",
        alpha=0.5,
    )

    inset_ax.text(  # type: ignore
        0.5,
        -0.04,
        "t-SNE",
        fontsize=10,
        ha="center",
        va="top",
        transform=inset_ax.transAxes,  # type: ignore
    )
    inset_ax.set_xticks(  # type: ignore
        []
    )
    inset_ax.set_yticks(  # type: ignore
        []
    )
    fig.set_size_inches(10, 6)
    return fig


def create_radar_chart_map(
    scenario: str,
    out_data: GenerationData,
    kind: str,
    zone_dict: ZoneDictZoneDataMapping,
) -> Figure | ErrorStatus:
    """
    Plot a Radar Chart (Spider Plot) for real and generated data for each zone.
    Each element in real_data and gen_data is a time series of occupancy.
    """

    if scenario == "1st":
        if kind not in ["parkingmeter", "parkingslot"]:
            return ErrorStatus(error="Invalid kind")
    if scenario == "2nd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")
    if scenario == "3rd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"

    zone_dict_kind = "parcometro" if kind == "parkingmeter" else "stalli"

    df_real_data = out_data[data_plot_s + "data_df"]
    df_gen_data = out_data[data_plot_s + "gen_df"]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    zones = [zone for zone in zone_dict.keys() if zone != "all_map"]

    real_data: list["pd.Series[pd.Float32Dtype]"] = []
    gen_data: list["pd.Series[pd.Float32Dtype]"] = []
    for key in zones:
        p_keys = zone_dict[key][zone_dict_kind]
        parc_data_df = df_real_data[p_keys]
        parc_gen_df = df_gen_data[p_keys]

        real_data.append(
            parc_data_df.mean(  # type: ignore
                axis=1
            )
        )
        gen_data.append(
            parc_gen_df.mean(  # type: ignore
                axis=1
            )
        )

    real_data_avg = [np.mean(series) for series in real_data]
    gen_data_avg = [np.mean(series) for series in gen_data]

    N = len(real_data_avg)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))  # type: ignore

    real_data_avg = np.concatenate((real_data_avg, [real_data_avg[0]]))
    gen_data_avg = np.concatenate((gen_data_avg, [gen_data_avg[0]]))

    max_value = float(max(np.max(real_data_avg), np.max(gen_data_avg)))

    ax.plot(  # type: ignore
        angles,
        real_data_avg,
        color="orange",
        linewidth=2,
        linestyle="solid",
        label="Real Data",
        alpha=0.9,
    )
    ax.fill(  # type: ignore
        angles, real_data_avg, color="orange", alpha=0.3
    )

    ax.plot(  # type: ignore
        angles,
        gen_data_avg,
        color="green",
        linewidth=2,
        linestyle="solid",
        label="Generated Data",
        alpha=0.9,
    )
    ax.fill(  # type: ignore
        angles, gen_data_avg, color="green", alpha=0.3
    )

    ax.legend(  # type: ignore
        loc="upper right",
        fontsize=7,
        bbox_to_anchor=(1.1, 1.1),
    )

    ax.set_xticks(  # type: ignore
        angles[:-1]
    )
    labels = [f"Zone {i}" for i in range(0, N)]

    ax.set_xticklabels(  # type: ignore
        []
    )
    for i, label in enumerate(labels):
        angle = angles[i]
        angle_deg = np.degrees(angle)

        rotation = angle_deg if angle_deg <= 90 or angle_deg >= 270 else angle_deg + 180
        ha = "center"

        ax.text(  # type: ignore
            angle,
            max_value * 1.1,
            label,
            horizontalalignment=ha,
            verticalalignment="center",
            fontsize=7,
            color="black",
            rotation=rotation,
            rotation_mode="anchor",
        )

    yticks = np.linspace(0, max_value, 10)
    ax.set_yticks(  # type: ignore
        yticks
    )
    ax.set_yticklabels(  # type: ignore
        []
    )
    ax.set_ylim(0, max_value)

    ax.set_facecolor("#F8F9F9")

    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig


def create_cumulative_plot(
    scenario: str,
    start_date: str,
    out_data: GenerationData,
    kind: str,
    selected_adjacent_zone: str,
    zone_dict: ZoneDictZoneDataMapping,
) -> Figure | ErrorStatus:
    timestamps = pd.date_range(  # type: ignore
        start=f"{start_date} 02:00:00", periods=42, freq="4H"
    )

    if scenario == "1st":
        if kind not in ["parkingmeter", "parkingslot"]:
            return ErrorStatus(error="Invalid kind")
    if scenario == "2nd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")
    if scenario == "3rd":
        if kind != "parkingslot":
            return ErrorStatus(error="Invalid kind")

    data_plot_s = "p_" if kind == "parkingmeter" else "s_"

    zone_dict_kind = "parcometro" if kind == "parkingmeter" else "stalli"

    selected_zones = out_data["selected_zones"]

    df_real_data = out_data[data_plot_s + "data_df"]
    df_gen_data = out_data[data_plot_s + "gen_df"]

    assert df_real_data is not None, "Real data DataFrame is None"
    assert df_gen_data is not None, "Generated data DataFrame is None"

    s_data_plot = pd.DataFrame()
    s_gen_plot = pd.DataFrame()

    if selected_adjacent_zone == "all_map":
        adjacent_zones = selected_zones
        for selected_zone in adjacent_zones:
            if selected_zone == "all_map":
                continue
            selected_kinds = zone_dict[selected_zone][zone_dict_kind]

            s_data_plot = pd.concat([s_data_plot, df_real_data[selected_kinds]], axis=1)
            s_gen_plot = pd.concat([s_gen_plot, df_gen_data[selected_kinds]], axis=1)
    else:
        selected_kinds = zone_dict[selected_adjacent_zone][zone_dict_kind]
        s_data_plot = df_real_data[selected_kinds]
        s_gen_plot = df_gen_data[selected_kinds]

    s_data_plot.index = timestamps
    s_gen_plot.index = timestamps

    s_real_values = np.zeros(s_data_plot.shape[0])
    s_generated_values = np.zeros(s_gen_plot.shape[0])
    for key in s_data_plot.columns:
        s_real_values += s_data_plot[key]
        s_generated_values += s_gen_plot[key]

    fig, ax = plt.subplots(  # type: ignore
    )
    ax.plot(  # type: ignore
        timestamps, s_real_values, label="Real", color="#00509E", lw=2.5
    )
    ax.plot(  # type: ignore
        timestamps,
        s_generated_values,
        label="Generated",
        color="#FF3B3B",
        lw=2.5,
    )
    ax.legend(  # type: ignore
        loc="upper right", fontsize=10
    )
    ax.grid(  # type: ignore
        True, linestyle="--", alpha=0.5, color="gray"
    )
    fig.tight_layout()
    fig.set_size_inches(10, 6)

    return fig
