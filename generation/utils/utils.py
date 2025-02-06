# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from geopy.distance import geodesic

# %%

def find_nearest_unique_point(lat, lon, grid_size, assigned_points, vertice1_lat, vertice1_lon, vertice2_lat, vertice2_lon):
    """
    Finds the nearest unique point on a grid that has not been assigned yet.

    Parameters:
    - lat (float): Latitude of the target point.
    - lon (float): Longitude of the target point.
    - grid_size (int): The number of points along each dimension of the grid.
    - assigned_points (set of tuples): A set of (latitude index, longitude index) pairs representing already assigned points.
    - vertice1_lat (float): Latitude of the first vertice of the grid.
    - vertice1_lon (float): Longitude of the first vertice of the grid.
    - vertice2_lat (float): Latitude of the second vertice of the grid.
    - vertice2_lon (float): Longitude of the second vertice of the grid.

    Returns:
    - lat_index, lon_index (tuple): The indices of the nearest unassigned point on the grid.
            If the nearest point is already assigned, it searches for the next available point in a spiral pattern.
    
    """
    
    lat_points = np.linspace(vertice1_lat, vertice2_lat, grid_size)
    lon_points = np.linspace(vertice1_lon, vertice2_lon, grid_size)
    
    # Find the nearest initial indices
    lat_index = np.argmin(np.abs(lat_points - lat))
    lon_index = np.argmin(np.abs(lon_points - lon))
    
    # Search for an unassigned point if the initial one is taken
    if (lat_index, lon_index) in assigned_points:
        for offset in range(1, grid_size):
            # Check neighboring cells around the initial point, moving in a spiral pattern
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    new_lat_index = lat_index + i
                    new_lon_index = lon_index + j
                    if (0 <= new_lat_index < grid_size and 0 <= new_lon_index < grid_size):
                        if (new_lat_index, new_lon_index) not in assigned_points:
                            return new_lat_index, new_lon_index
    # If the initial point is unique or an alternative point is found
    return lat_index, lon_index


def grid_building(parkimeters, slots, grid_size, scenario, mapping_dict, time_start, time_end):
    """
    Constructs a grid-based representation of parkimeters and parking slots data over a given time range.
    The function normalizes the data, maps locations to a grid, and returns a matrix representation.
    
    Parameters:
    - parkimeters (dict): Dictionary containing parkimeter data with timestamps.
    - slots (dict): Dictionary containing parking slot data with timestamps.
    - grid_size (int): Size of the grid to build.
    - scenario (str): Scenario type ('1st', '2nd', or '3rd').
    - mapping_dict (dict): Dictionary mapping zones to parkimeters and slots.
    - time_start (str): Start time for the data range.
    - time_end (str): End time for the data range.
    
    Returns:
    - matrix (numpy array): The grid-based representation of parkimeters and slots.
    - indices_slots (dict): Mapping of slots to grid positions.
    - indices_parkimeter (dict, optional): Mapping of parkimeters to grid positions (if applicable).
    - scaler_slots (MinMaxScaler): Scaler object used for slots.
    - scaler_parkimeters (MinMaxScaler, optional): Scaler object used for parkimeters (if applicable).
    """

    timestamp = pd.date_range(start=time_start, end=time_end, freq='4H')

    parkimeters_list = []
    slots_list = []
    for key in parkimeters.keys():
        parkimeters_list.append(parkimeters[key]['data'].loc[time_start:time_end]).values
    for key in slots.keys():
        slots_list.append(slots[key]['data'].loc[time_start:time_end]).values 
    
    df_parkimeters = pd.DataFrame(parkimeters_list).T
    df_parkimeters.columns = [key for key in parkimeters.keys()]
    df_parkimeters = df_parkimeters.sort_index(axis=1)
    df_parkimeters.index = timestamp
    df_slots = pd.DataFrame(slots_list).T
    df_slots.columns = [key for key in slots.keys()]
    df_slots = df_slots.sort_index(axis=1)
    df_slots.index = timestamp

    # Scalers for Parkimeters
    scaler_parkimeters = MinMaxScaler()
    df_parkimeters_scaled = pd.DataFrame(scaler_parkimeters.fit_transform(parkimeters), columns=parkimeters.columns, index=parkimeters.index)
    for key in parkimeters.keys():
        parkimeters[key]['data'] = df_parkimeters_scaled[key]
        parkimeters[key]['data'] = pd.Series(parkimeters[key]['data'], index=timestamp)
    
    # Scalers for Slots
    scaler_slots = MinMaxScaler()
    df_slots_scaled = pd.DataFrame(scaler_slots.fit_transform(slots), columns=slots.columns, index=slots.index)
    for key in slots.keys():
        slots[key]['data'] = df_slots_scaled[key]
        slots[key]['data'] = pd.Series(slots[key]['data'], index=timestamp)
    
    if scenario == '2nd':
        # Roads matrix
        roads = []
        for key in slots.keys():
            roads.append(float(slots[key]['id_strada']))

        roads = list(set(roads))
        dict_road = {}
        for road in roads:
            dict_road[road] = []
            for key in slots.keys():
                if float(slots[key]['id_strada']) == road:
                    dict_road[road].append(key)

        timestamp = slots[key]['data'].index

        strade_dict = {}
        for key, value in dict_road.items():
            strade_dict[key] = pd.DataFrame(index=timestamp)
            columns_to_add = []
            for key2 in value:
                columns_to_add.append(slots[key2]['data'])
            strade_dict[key] = pd.concat(columns_to_add, axis=1)
            strade_dict[key] = strade_dict[key].apply(lambda row: (row != 0).sum(), axis=1)

    
    # Minimum coordinates between Parkimeters and Slots
    min_lat = min(
        min(parkimeters[key]['lat'] for key in parkimeters.keys()),
        min(min(slots[key]['lat'] for key in slots.keys()))
        )
    min_lon = min(
        min(parkimeters[key]['lon'] for key in parkimeters.keys()),
        min(min(slots[key]['lon'] for key in slots.keys()))
        )
    max_lat = max(
        max(parkimeters[key]['lat'] for key in parkimeters.keys()),
        max(max(slots[key]['lat'] for key in slots.keys()))
        )
    max_lon = max(
        max(parkimeters[key]['lon'] for key in parkimeters.keys()),
        max(max(slots[key]['lon'] for key in slots.keys()))
        )
    vertices_lat_min = min_lat - 2e-3
    vertices_lat_max = max_lat + 2e-3
    vertices_lon_min = min_lon - 2e-3
    vertices_lon_max = max_lon + 2e-3

    # Spyral grid building for Parkimeters
    if scenario == '1st' or scenario == '3rd':
        indices_parkimeter = {}
        assigned_points = set()
        for zone in mapping_dict.keys():
            selected_parkimeters = mapping_dict[zone]['parcometro']
            for key in selected_parkimeters:
                lat = parkimeters[key]["lat"]
                lon = parkimeters[key]["lng"]
                nearest_point = find_nearest_unique_point(lat, lon, grid_size, assigned_points, vertices_lat_min, vertices_lon_min, vertices_lat_max, vertices_lon_max)
                indices_parkimeter[key] = nearest_point
                assigned_points.add(nearest_point)
    
    # Spyral grid building for Slots
    indices_slots = {}
    assigned_points = set()

    for zone in mapping_dict.keys():
        selected_slots = mapping_dict[zone]['stalli']
        for key in selected_slots:
            lat = slots[key]["lat"]
            lon = slots[key]["lng"]
            nearest_point = find_nearest_unique_point(lat, lon, grid_size, assigned_points, vertices_lat_min, vertices_lon_min, vertices_lat_max, vertices_lon_max)
            indices_slots[key] = nearest_point
            assigned_points.add(nearest_point)


    # Grid matrix building
    if scenario == '1st' or scenario == '3rd':

        matrix = np.zeros((grid_size, grid_size, 2, len(timestamp)))

        for key in slots.keys():
            matrix[indices_slots[key][0], indices_slots[key][1], 0, :] = slots[key]['data'].values
        for key in parkimeters.keys():
            matrix[indices_parkimeter[key][0], indices_parkimeter[key][1], 1, :] = parkimeters[key]['data'].values
        
        return matrix, indices_slots, indices_parkimeter, scaler_slots, scaler_parkimeters
    
    elif scenario == '2nd':

        matrix = np.zeros((grid_size, grid_size, 1, len(timestamp)))
        road_matrix = np.zeros((grid_size, grid_size, 1, len(timestamp)))

        for key in slots.keys():
            matrix[indices_slots[key][0], indices_slots[key][1], 0, :] = slots[key]['data'].values
        for key in strade_dict.keys():
            for key2 in slots.keys():
                if slots[key2]['id_strada'] == key:
                    road_matrix[indices_slots[key2][0], indices_slots[key2][1], 0, :] = strade_dict[key].values
        
        return matrix, road_matrix, indices_slots, scaler_slots


def add_conditions(val_conditions, indices_parkimeter, indices_slots, final_slots, final_parkimeters, mapping_dict, args):
    """
    Modify the val_conditions matrix based on different scenarios.
    
    Parameters:
    - val_conditions (torch.Tensor): A multi-dimensional tensor storing condition values for grid locations.
    - indices_parkimeter (dict): Mapping of parkimeter IDs to their respective grid coordinates.
    - indices_slots (dict): Mapping of slot IDs to their respective grid coordinates.
    - final_slots (dict): Dictionary containing final slot data.
    - final_parkimeters (dict): Dictionary containing final parkimeter data.
    - mapping_dict (dict): Dictionary mapping zones to parkimeters and slots.
    - args (Namespace): Contains additional parameters, including the selected scenario and other attributes.
    
    Returns:
    - torch.Tensor: Updated val_conditions tensor after applying modifications.
    """
    
    if args.scenario == "1st":

        selected_parkimeters = mapping_dict[args.selected_zone]["parcometro"]
        selected_slots = mapping_dict[args.selected_zone]["stalli"]

        # Implementation of zone closure via simulation
        for key in selected_parkimeters:
            lat, lon = indices_parkimeter[key][0], indices_parkimeter[key][1]
            val_conditions[:, 1, :, lat, lon] = 0
        for key in selected_slots:
            lat, lon = indices_slots[key][0], indices_slots[key][1]
            val_conditions[:, 0, :, lat, lon] = 0

        remaining_zones = [
            zone for zone in mapping_dict.keys() if zone != args.selected_zone
        ]

        p_remaining = [mapping_dict[z]["parcometro"] for z in remaining_zones]
        p_remaining = [item for sublist in p_remaining for item in sublist]
        s_remaining = [mapping_dict[z]["stalli"] for z in remaining_zones]
        s_remaining = [item for sublist in s_remaining for item in sublist]

        delta_p = {}
        for key in p_remaining:
            x = final_parkimeters[key]["lat"]
            y = final_parkimeters[key]["lng"]
            delta_p[key] = 0
            for key2 in selected_parkimeters:
                x2 = final_parkimeters[key2]["lat"]
                y2 = final_parkimeters[key2]["lng"]
                dist = geodesic((x, y), (x2, y2)).kilometers
                delta_p[key] += np.exp(-dist)

        rho_p = {}
        for key in delta_p.keys():
            rho_p[key] = delta_p[key] / (max(delta_p.values()) + 1e-6)

        delta_s = {}
        for key in s_remaining:
            x = final_slots[key]["lat"]
            y = final_slots[key]["lng"]
            delta_s[key] = 0
            for key2 in selected_slots:
                x2 = final_slots[key2]["lat"]
                y2 = final_slots[key2]["lng"]
                dist = geodesic((x, y), (x2, y2)).kilometers
                delta_s[key] += np.exp(-dist)

        rho_s = {}
        for key in delta_s.keys():
            rho_s[key] = delta_s[key] / (max(delta_s.values()) + 1e-6)

        for z in remaining_zones:
            p_adjust = mapping_dict[z]["parcometro"]
            s_adjust = mapping_dict[z]["stalli"]

            for p in p_adjust:
                p_mask_zero = (
                    val_conditions[
                        :, 1, :, indices_parkimeter[p][0], indices_parkimeter[p][1]
                    ]
                    == 0
                )
                val_conditions[
                    :, 1, :, indices_parkimeter[p][0], indices_parkimeter[p][1]
                ] = torch.from_numpy(
                    np.where(
                        p_mask_zero,
                        0,
                        val_conditions[
                            :, 1, :, indices_parkimeter[p][0], indices_parkimeter[p][1]
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
                    torch.from_numpy(
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

    elif args.scenario == "2nd":

        selected_slots = mapping_dict[args.selected_zone]["stalli"]

        # Implementation of a multi-story facility via simulation
        for key in selected_slots:
            lat, lon = indices_slots[key][0], indices_slots[key][1]
            val_conditions[:, 1, :, lat, lon] += args.quantity

        lat_indices = [lat for lat, lon in indices_slots.values()]
        lon_indices = [lon for lat, lon in indices_slots.values()]
        mask_zero = val_conditions[:, 1, :, lat_indices, lon_indices] == 0
        val_conditions[:, 1, :, lat_indices, lon_indices] = torch.from_numpy(
            np.where(
                mask_zero,
                0,
                torch.log(1 / (val_conditions[:, 1, :, lat_indices, lon_indices] + 1)),
            )
        ).float()

    elif args.scenario == "3rd":

        # Implementation of rainy days via simulation
        for slot in indices_slots.keys():
            lat, lon = indices_slots[slot][0], indices_slots[slot][1]
            val_conditions[:, 2, :, lat, lon] = 1
        for park in indices_parkimeter.keys():
            lat, lon = indices_parkimeter[park][0], indices_parkimeter[park][1]
            val_conditions[:, 2, :, lat, lon] = 1


    return val_conditions
