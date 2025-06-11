from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic  # type: ignore
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

FloatArray = NDArray[np.float64]

ScenarioType = Literal["1st", "2nd", "3rd"]


def find_nearest_unique_point(
    lat: float,
    lon: float,
    grid_size: int,
    assigned_points: set[tuple[int, int]],
    vertex1_lat: float,
    vertex1_lon: float,
    vertex2_lat: float,
    vertex2_lon: float,
) -> tuple[int, int]:
    """
    Finds the nearest unique point on a grid that has not been assigned yet.

    Parameters:
    - lat (float): Latitude of the target point.
    - lon (float): Longitude of the target point.
    - grid_size (int): The number of points along each dimension of the grid.
    - assigned_points (set of tuples): A set of (latitude index, longitude index) pairs representing
    already assigned points.
    - vertex1_lat (float): Latitude of the first vertex of the grid.
    - vertex1_lon (float): Longitude of the first vertex of the grid.
    - vertex2_lat (float): Latitude of the second vertex of the grid.
    - vertex2_lon (float): Longitude of the second vertex of the grid.

    Returns:
    - lat_index, lon_index (tuple): The indices of the nearest unassigned point on the grid.
            If the nearest point is already assigned, it searches for the next available point in a spiral pattern.

    """

    lat_points = np.linspace(vertex1_lat, vertex2_lat, grid_size)
    lon_points = np.linspace(vertex1_lon, vertex2_lon, grid_size)

    # Find the nearest initial indices
    lat_index = int(np.argmin(np.abs(lat_points - lat)))
    lon_index = int(np.argmin(np.abs(lon_points - lon)))

    # Search for an unassigned point if the initial one is taken
    if (lat_index, lon_index) in assigned_points:
        for offset in range(1, grid_size):
            # Check neighboring cells around the initial point, moving in a spiral pattern
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    new_lat_index = lat_index + i
                    new_lon_index = lon_index + j
                    if (
                        0 <= new_lat_index < grid_size
                        and 0 <= new_lon_index < grid_size
                    ):
                        if (new_lat_index, new_lon_index) not in assigned_points:
                            return new_lat_index, new_lon_index
    # If the initial point is unique or an alternative point is found
    return lat_index, lon_index


def grid_building(
    parkingmeters: dict[int, Any],
    slots: dict[int, Any],
    grid_size: int,
    scenario: ScenarioType,
    mapping_dict: dict[str, Any],
    time_start: pd.Timestamp,
    time_end: pd.Timestamp,
) -> tuple[
    FloatArray,
    dict[int, tuple[int, int]] | FloatArray,
    dict[int, tuple[int, int]],
    MinMaxScaler,
    MinMaxScaler | None,
]:
    """
    Constructs a grid-based representation of parkingmeters and parking slots data over a given time range.
    The function normalizes the data, maps locations to a grid, and returns a matrix representation.

    Parameters:
    - parkingmeters (dict): Dictionary containing parkingmeter data with timestamps.
    - slots (dict): Dictionary containing parking slot data with timestamps.
    - grid_size (int): Size of the grid to build.
    - scenario (ScenarioType): Scenario type ('1st', '2nd', or '3rd').
    - mapping_dict (dict): Dictionary mapping zones to parkingmeters and slots.
    - time_start (str): Start time for the data range.
    - time_end (str): End time for the data range.

    Returns:
    - matrix (numpy array): The grid-based representation of parkingmeters and slots.
    - indices_slots (dict): Mapping of slots to grid positions.
    - indices_parkingmeter (dict, optional): Mapping of parkingmeters to grid positions (if applicable).
    - scaler_slots (MinMaxScaler): Scaler object used for slots.
    - scaler_parkingmeters (MinMaxScaler, optional): Scaler object used for parkingmeters (if applicable).
    """

    timestamp = pd.date_range(start=time_start, end=time_end, freq="4H")  # type: ignore

    parkingmeters_list: list[FloatArray] = []
    slots_list: list[FloatArray] = []
    for key in parkingmeters.keys():
        parkingmeters_list.append(
            parkingmeters[key]["data"].loc[time_start:time_end].values  # type: ignore
        )
    for key in slots.keys():
        slots_list.append(slots[key]["data"].loc[time_start:time_end].values)

    df_parkingmeters = pd.DataFrame(parkingmeters_list).T
    df_parkingmeters.columns = [key for key in parkingmeters.keys()]
    df_parkingmeters = df_parkingmeters.sort_index(axis=1)  # type: ignore
    df_parkingmeters.index = timestamp
    df_slots = pd.DataFrame(slots_list).T
    df_slots.columns = [key for key in slots.keys()]
    df_slots = df_slots.sort_index(axis=1)  # type: ignore
    df_slots.index = timestamp

    # Scalers for Parking meters
    scaler_parkingmeters = MinMaxScaler()
    df_parkingmeters_scaled = pd.DataFrame(
        scaler_parkingmeters.fit_transform(  # type: ignore
            df_parkingmeters
        ),
        columns=df_parkingmeters.columns,
        index=df_parkingmeters.index,
    )
    for key in parkingmeters.keys():
        parkingmeters[key]["data"] = df_parkingmeters_scaled[key]
        parkingmeters[key]["data"] = pd.Series(
            parkingmeters[key]["data"], index=timestamp
        )

    # Scalers for Slots
    scaler_slots = MinMaxScaler()
    df_slots_scaled = pd.DataFrame(
        scaler_slots.fit_transform(  # type: ignore
            df_slots
        ),
        columns=df_slots.columns,
        index=df_slots.index,
    )
    for key in slots.keys():
        slots[key]["data"] = df_slots_scaled[key]
        slots[key]["data"] = pd.Series(slots[key]["data"], index=timestamp)

    roads_dict: dict[float, pd.DataFrame] = {}
    if scenario == "2nd":
        # Roads matrix
        roads: list[float] = []
        for key in slots.keys():
            roads.append(float(slots[key]["id_strada"]))

        roads = list(set(roads))
        dict_road: dict[float, list[int]] = {}
        key = -1  # Initialize key to avoid KeyError
        for road in roads:
            dict_road[road] = []
            for key in slots.keys():
                if float(slots[key]["id_strada"]) == road:
                    dict_road[road].append(key)

        timestamp_ = slots[key]["data"].index

        for key, value in dict_road.items():
            roads_dict[key] = pd.DataFrame(index=timestamp_)
            columns_to_add: list["pd.Series[pd.Float64Dtype]"] = []
            for key2 in value:
                columns_to_add.append(slots[key2]["data"])
            roads_dict[key] = pd.concat(columns_to_add, axis=1)
            roads_dict[key] = roads_dict[key].apply(  # type: ignore
                lambda row: (row != 0).sum(),  # type: ignore
                axis=1,
            )

    # Minimum coordinates between Parking meters and Slots
    min_lat = float(
        min(
            min(parkingmeters[key]["lat"] for key in parkingmeters.keys()),
            min(slots[key]["lat"] for key in slots.keys()),
        )
    )
    min_lon = float(
        min(
            min(parkingmeters[key]["lng"] for key in parkingmeters.keys()),
            min(slots[key]["lng"] for key in slots.keys()),
        )
    )
    max_lat = float(
        max(
            max(parkingmeters[key]["lat"] for key in parkingmeters.keys()),
            max(slots[key]["lat"] for key in slots.keys()),
        )
    )
    max_lon = float(
        max(
            max(parkingmeters[key]["lng"] for key in parkingmeters.keys()),
            max(slots[key]["lng"] for key in slots.keys()),
        )
    )
    vertices_lat_min = min_lat - 2e-3
    vertices_lat_max = max_lat + 2e-3
    vertices_lon_min = min_lon - 2e-3
    vertices_lon_max = max_lon + 2e-3

    indices_parkingmeter: dict[int, tuple[int, int]] = {}
    # Spiral grid building for Parking meters
    if scenario in ["1st", "3rd"]:
        assigned_points: set[tuple[int, int]] = set()
        for zone in mapping_dict.keys():
            selected_parkingmeters = mapping_dict[zone]["parcometro"]
            for key in selected_parkingmeters:
                lat = parkingmeters[key]["lat"]
                lon = parkingmeters[key]["lng"]
                nearest_point = find_nearest_unique_point(
                    lat,
                    lon,
                    grid_size,
                    assigned_points,
                    vertices_lat_min,
                    vertices_lon_min,
                    vertices_lat_max,
                    vertices_lon_max,
                )
                indices_parkingmeter[key] = nearest_point
                assigned_points.add(nearest_point)

    # Spiral grid building for Slots
    indices_slots: dict[int, tuple[int, int]] = {}
    assigned_points: set[tuple[int, int]] = set()

    for zone in mapping_dict.keys():
        selected_slots = mapping_dict[zone]["stalli"]
        for key in selected_slots:
            lat = slots[key]["lat"]
            lon = slots[key]["lng"]
            nearest_point = find_nearest_unique_point(
                lat,
                lon,
                grid_size,
                assigned_points,
                vertices_lat_min,
                vertices_lon_min,
                vertices_lat_max,
                vertices_lon_max,
            )
            indices_slots[key] = nearest_point
            assigned_points.add(nearest_point)

    # Grid matrix building
    if scenario in ["1st", "3rd"]:
        matrix = np.zeros((grid_size, grid_size, 2, len(timestamp)))

        for key in slots.keys():
            matrix[indices_slots[key][0], indices_slots[key][1], 0, :] = slots[key][
                "data"
            ].values
        for key in parkingmeters.keys():
            matrix[indices_parkingmeter[key][0], indices_parkingmeter[key][1], 1, :] = (
                parkingmeters[key]["data"].values
            )

        return (
            matrix,
            indices_slots,
            indices_parkingmeter,
            scaler_slots,
            scaler_parkingmeters,
        )

    matrix = np.zeros((grid_size, grid_size, 1, len(timestamp)))
    road_matrix = np.zeros((grid_size, grid_size, 1, len(timestamp)))

    for key in slots.keys():
        matrix[indices_slots[key][0], indices_slots[key][1], 0, :] = slots[key][
            "data"
        ].values
    for key in roads_dict.keys():
        for key2 in slots.keys():
            if slots[key2]["id_strada"] == key:
                road_matrix[indices_slots[key2][0], indices_slots[key2][1], 0, :] = (
                    roads_dict[key].values  # type: ignore
                )

    return matrix, road_matrix, indices_slots, scaler_slots, None


def add_conditions(
    val_conditions: torch.Tensor,
    indices_parkingmeter: dict[int, tuple[int, int]],
    indices_slots: dict[int, tuple[int, int]],
    final_slots: dict[int, Any],
    final_parkingmeters: dict[int, Any],
    mapping_dict: dict[str, Any],
    scenario: ScenarioType,
    quantity: float | None = None,
    selected_zone: str | None = None,
) -> torch.Tensor:
    """
    Modify the val_conditions matrix based on different scenarios.

    Parameters:
    - val_conditions (torch.Tensor): A multi-dimensional tensor storing condition values for grid locations.
    - indices_parkingmeter (dict): Mapping of parking meter IDs to their respective grid coordinates.
    - indices_slots (dict): Mapping of slot IDs to their respective grid coordinates.
    - final_slots (dict): Dictionary containing final slot data.
    - final_parkingmeters (dict): Dictionary containing final parking meter data.
    - mapping_dict (dict): Dictionary mapping zones to parking meters and slots.
    - scenario (ScenarioType): The scenario type to apply modifications ('1st', '2nd', or '3rd').
    - quantity (float, optional): Quantity to add in the '2nd' scenario. Defaults to None.
    - selected_zone (str, optional): The zone to select for the '1st' and '2nd' scenarios. Defaults to None.

    Returns:
    - torch.Tensor: Updated val_conditions tensor after applying modifications.
    """

    if scenario == "1st":
        assert selected_zone is not None, (
            "selected_zone must be provided for scenario '1st'"
        )
        selected_parkingmeters = mapping_dict[selected_zone]["parcometro"]
        selected_slots = mapping_dict[selected_zone]["stalli"]

        # Implementation of zone closure via simulation
        for key in selected_parkingmeters:
            lat, lon = indices_parkingmeter[key][0], indices_parkingmeter[key][1]
            val_conditions[:, 1, :, lat, lon] = 0
        for key in selected_slots:
            lat, lon = indices_slots[key][0], indices_slots[key][1]
            val_conditions[:, 0, :, lat, lon] = 0

        remaining_zones = [
            zone for zone in mapping_dict.keys() if zone != selected_zone
        ]

        p_remaining = [mapping_dict[z]["parcometro"] for z in remaining_zones]
        p_remaining = [item for sublist in p_remaining for item in sublist]
        s_remaining = [mapping_dict[z]["stalli"] for z in remaining_zones]
        s_remaining = [item for sublist in s_remaining for item in sublist]

        delta_p: dict[int, float] = {}
        for key in p_remaining:
            x = final_parkingmeters[key]["lat"]
            y = final_parkingmeters[key]["lng"]
            delta_p[key] = 0.0
            for key2 in selected_parkingmeters:
                x2 = float(final_parkingmeters[key2]["lat"])
                y2 = float(final_parkingmeters[key2]["lng"])
                dist = float(
                    geodesic((x, y), (x2, y2)).kilometers  # type: ignore
                )
                delta_p[key] += np.exp(-dist)

        rho_p: dict[int, float] = {}
        for key in delta_p.keys():
            rho_p[key] = delta_p[key] / (max(delta_p.values()) + 1e-6)

        delta_s: dict[int, float] = {}
        for key in s_remaining:
            x = final_slots[key]["lat"]
            y = final_slots[key]["lng"]
            delta_s[key] = 0.0
            for key2 in selected_slots:
                x2 = final_slots[key2]["lat"]
                y2 = final_slots[key2]["lng"]
                dist = float(
                    geodesic((x, y), (x2, y2)).kilometers  # type: ignore
                )
                delta_s[key] += np.exp(-dist)

        rho_s: dict[int, float] = {}
        for key in delta_s.keys():
            rho_s[key] = delta_s[key] / (max(delta_s.values()) + 1e-6)

        for z in remaining_zones:
            p_adjust = mapping_dict[z]["parcometro"]
            s_adjust = mapping_dict[z]["stalli"]

            for p in p_adjust:
                p_mask_zero = (
                    val_conditions[
                        :, 1, :, indices_parkingmeter[p][0], indices_parkingmeter[p][1]
                    ]
                    == 0
                )
                val_conditions[
                    :, 1, :, indices_parkingmeter[p][0], indices_parkingmeter[p][1]
                ] = torch.from_numpy(  # type: ignore
                    np.where(
                        p_mask_zero,
                        0,
                        val_conditions[
                            :,
                            1,
                            :,
                            indices_parkingmeter[p][0],
                            indices_parkingmeter[p][1],
                        ]
                        + rho_p[p],
                    )
                ).float()

            for s in s_adjust:
                s_mask_zero = (
                    val_conditions[:, 0, :, indices_slots[s][0], indices_slots[s][1]]
                    == 0
                )
                val_conditions[:, 0, :, indices_slots[s][0], indices_slots[s][1]] = (
                    torch.from_numpy(  # type: ignore
                        np.where(
                            s_mask_zero,
                            0,
                            val_conditions[
                                :, 0, :, indices_slots[s][0], indices_slots[s][1]
                            ]
                            + rho_s[s],
                        )
                    ).float()
                )

    elif scenario == "2nd":
        assert selected_zone is not None, (
            "selected_zone must be provided for scenario '1st'"
        )
        selected_slots = mapping_dict[selected_zone]["stalli"]

        # Implementation of a multi-story facility via simulation
        for key in selected_slots:
            lat, lon = indices_slots[key][0], indices_slots[key][1]
            val_conditions[:, 1, :, lat, lon] += quantity

        lat_indices = [lat for lat, _lon in indices_slots.values()]
        lon_indices = [lon for _lat, lon in indices_slots.values()]
        mask_zero = val_conditions[:, 1, :, lat_indices, lon_indices] == 0
        val_conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(  # type: ignore
            np.where(
                mask_zero,
                0,
                torch.log(1 / (val_conditions[:, 1, :, lat_indices, lon_indices] + 1)),
            )
        ).float()

    else:
        # Implementation of rainy days via simulation
        for slot in indices_slots.keys():
            lat, lon = indices_slots[slot][0], indices_slots[slot][1]
            val_conditions[:, 2, :, lat, lon] = 1
        for park in indices_parkingmeter.keys():
            lat, lon = indices_parkingmeter[park][0], indices_parkingmeter[park][1]
            val_conditions[:, 2, :, lat, lon] = 1

    return val_conditions
