import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from torch.utils.data import Dataset


def create_sequences_multivariate(series, indices, seq_length, horizon):
    """
    Creates sequences for multivariate time series forecasting.

    Parameters:
    - series: The multivariate time series data.
    - indices: The indices or time steps corresponding to each row in `series`.
    - seq_length: The length of the input sequence.
    - horizon: The number of future time steps to predict.

    Returns:
    - xs: Input sequences as torch tensors.
    - x_indices: Indices corresponding to the input sequences.
    - ys: Output sequences as torch tensors.
    - y_indices: Indices corresponding to the output sequences.
    """

    xs, ys = [], []
    x_indices, y_indices = [], []
    data = series
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i : i + seq_length, :]
        y = data[i + seq_length : i + seq_length + horizon, :]
        x_indx = indices[i : i + seq_length]
        y_indx = indices[i + seq_length : i + seq_length + horizon]
        x_indices.append(x_indx)
        y_indices.append(y_indx)
        xs.append(x)
        ys.append(y)

    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        np.array(x_indices),
        torch.tensor(np.array(ys), dtype=torch.float32),
        np.array(y_indices),
    )


def add_features(data, add_time_of_day=None, add_day_of_week=None, steps_per_day=24):
    """
    Adds additional time-related features (time of day and day of week) to a given time series dataset.

    Parameters:
    - data: The input time series data.
    - add_time_of_day: Boolean flag to indicate if the 'time of day' feature should be added.
    - add_day_of_week: Boolean flag to indicate if the 'day of the week' feature should be added.
    - steps_per_day: The number of steps per day (default is 24, assuming hourly data).

    Returns:
    - data_with_features: A NumPy array containing the original time series data plus any added time-related features.
    """

    data1 = np.expand_dims(data[0].values, axis=-1)
    _, n = data[0].shape
    feature_list = [data1]

    if add_time_of_day:
        # Numerical time_of_day
        tod = data[0].index.hour / steps_per_day
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # Numerical day_of_week
        dow = data[0].index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)

    return data_with_features


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data=None,
        seasonal=None,
        trend=None,
        residual=None,
        targets=None,
        exogenous=None,
        poi_data=None,
        mask=None,
    ):
        self.seasonal = seasonal
        self.trend = trend
        self.residual = residual
        self.data = data
        self.targets = targets
        self.exogenous = exogenous
        self.poi_data = poi_data
        self.mask = mask

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {}
        if self.data is not None:
            item["data"] = self.data[idx]
        if self.seasonal is not None:
            item["seasonal"] = self.seasonal[idx]
        if self.trend is not None:
            item["trend"] = self.trend[idx]
        if self.residual is not None:
            item["residual"] = self.residual[idx]
        if self.targets is not None:
            item["targets"] = self.targets[idx]
        if self.exogenous is not None:
            item["exogenous"] = self.exogenous[idx]
        if self.poi_data is not None:
            item["poi_data"] = self.poi_data
        if self.mask is not None:
            item["mask"] = self.mask
        return item


def haversine_matrix(gps_coordinates):
    """
    Compute a matrix of Haversine distances between all GPS points.

    Parameters:
    - gps_coordinates (tensor): A 2D tensor with shape (N, 2), where each row represents (latitude, longitude) in degrees.

    Returns:
    - distance_matrix (tensor): A 2D tensor of shape (N, N) containing the pairwise Haversine distances in kilometers.
    """
    latitudes = gps_coordinates[:, 0]
    longitudes = gps_coordinates[:, 1]

    latitudes = np.radians(latitudes)
    longitudes = np.radians(longitudes)

    lat1 = latitudes.unsqueeze(0)
    lat2 = latitudes.unsqueeze(1)
    lon1 = longitudes.unsqueeze(0)
    lon2 = longitudes.unsqueeze(1)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = (
        torch.sin(delta_lat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Earth radius in km
    R = 6371.0

    # Compute distance matrix
    distance_matrix = R * c

    return distance_matrix


def normalize_distance_matrix(distance_matrix):
    """
    Normalize the distance matrix to a [0, 1] scale.

    Parameters:
    - distance_matrix (tensor): A 2D tensor containing pairwise distances.

    Returns:
    - normalized_matrix (tensor): A 2D tensor where values are scaled between 0 and 1.
    """

    min_dist = distance_matrix.min()
    max_dist = distance_matrix.max()

    denominator = max_dist - min_dist
    if denominator == 0:
        return distance_matrix
    else:
        normalized_matrix = (distance_matrix - min_dist) / denominator

    return normalized_matrix


def split(data, exog_data, train_percentage):
    """
    Splits the dataset into training, validation, and test sets.
    It scales the data, applies STL decomposition, and adds time-related features (time of day, day of week).

    Parameters:
    - data: DataFrame containing the main time series data.
    - exog_data: DataFrame containing the exogenous variables.
    - train_percentage: Percentage of training data.


    Returns:
    - data_scaler: The MinMaxScaler used for scaling the data.
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    """

    # Define the end of the training period based on the train_percentage
    train_end = int(train_percentage * data.shape[0])

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]

    # Scale the training and test data using MinMaxScaler
    data_scaler = MinMaxScaler()
    train_data_scaled = pd.DataFrame(
        data_scaler.fit_transform(train_data),
        columns=train_data.columns,
        index=train_data.index,
    )
    test_data_scaled = pd.DataFrame(
        data_scaler.transform(test_data),
        columns=test_data.columns,
        index=test_data.index,
    )

    # Scale the exogenous data
    train_exog = exog_data.iloc[:train_end]
    test_exog = exog_data.iloc[train_end:]
    exog_scaler = MinMaxScaler()
    train_exog_scaled = pd.DataFrame(
        exog_scaler.fit_transform(train_exog),
        columns=train_exog.columns,
        index=train_exog.index,
    )
    test_exog_scaled = pd.DataFrame(
        exog_scaler.transform(test_exog),
        columns=test_exog.columns,
        index=test_exog.index,
    )
    assert train_data.shape[0] == train_exog.shape[0]
    assert test_data.shape[0] == test_exog.shape[0]
    assert (train_data.index == train_exog.index).all()
    assert (test_data.index == test_exog.index).all()

    # Apply STL decomposition (Seasonal-Trend decomposition using LOESS) to the scaled data
    stl_train_data = pd.DataFrame(columns=data.columns)
    trend_train_data = pd.DataFrame(columns=data.columns)
    seasonal_train_data = pd.DataFrame(columns=data.columns)
    residual_train_data = pd.DataFrame(columns=data.columns)

    stl_test_data = pd.DataFrame(columns=data.columns)
    trend_test_data = pd.DataFrame(columns=data.columns)
    seasonal_test_data = pd.DataFrame(columns=data.columns)
    residual_test_data = pd.DataFrame(columns=data.columns)

    # Perform STL decomposition for each column in the training and test data
    for col in train_data.columns:
        result = STL(train_data_scaled[col], seasonal=23).fit()
        stl_train_data[col] = result
        (
            trend_train_data[col],
            seasonal_train_data[col],
            residual_train_data[col],
        ) = (result.trend, result.seasonal, result.resid)

        result = STL(test_data_scaled[col], seasonal=23).fit()
        stl_test_data[col] = result
        (
            trend_test_data[col],
            seasonal_test_data[col],
            residual_test_data[col],
        ) = (result.trend, result.seasonal, result.resid)

    # Split the training data into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * train_data.shape[0])

    train_index = train_data.index[:train_size]
    val_index = train_data.index[train_size:]
    test_index = test_data.index

    # Prepare data with features
    train_data_1 = {
        "data": train_data_scaled,
        "seasonal": seasonal_train_data,
        "trend": trend_train_data,
        "residual": residual_train_data,
        "index": train_index,
    }
    test_data_1 = {
        "data": test_data_scaled,
        "seasonal": seasonal_test_data,
        "trend": trend_test_data,
        "residual": residual_test_data,
        "index": test_index,
    }

    train_data_with_features = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "index": train_index,
    }
    val_data_with_features = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "index": val_index,
    }
    test_data_with_features = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "index": test_index,
    }

    data_type = list(train_data_with_features.keys())
    data_type.remove("index")

    # Add time-based features (e.g., time of day, day of week) to the data
    for key in data_type:
        train_data_with_features[key] = add_features(
            [train_data_1[key]], add_time_of_day=True, add_day_of_week=True
        )
        test_data_with_features[key] = add_features(
            [test_data_1[key]], add_time_of_day=True, add_day_of_week=True
        )

        val_data_with_features[key] = train_data_with_features[key][train_size:]
        train_data_with_features[key] = train_data_with_features[key][:train_size]

    # Add exogenous features
    train_data_with_features["exog"] = train_exog_scaled[:train_size].values
    val_data_with_features["exog"] = train_exog_scaled[train_size:].values
    test_data_with_features["exog"] = test_exog_scaled.values

    return (
        data_scaler,
        train_data_with_features,
        val_data_with_features,
        test_data_with_features,
    )


def create_datasets(
    train_data_with_features,
    val_data_with_features,
    test_data_with_features,
    poi_tensor,
    mask,
    args,
):
    """
    Creates datasets for training, validation, and testing by generating sequences from multivariate time series data.

    Parameters:
    - train_data_with_features: Dictionary containing training data with features.
    - val_data_with_features: Dictionary containing validation data with features.
    - test_data_with_features: Dictionary containing test data with features.
    - poi_tensor: Tensor containing Points of Interest (POI) information.
    - mask: Mask tensor for POI data.
    - args: Argument object containing input_length (length of input sequences) and horizon (forecast horizon).

    Returns:
    - train_dataset: TimeSeriesDataset object for training.
    - val_dataset: TimeSeriesDataset object for validation.
    - test_dataset: TimeSeriesDataset object for testing.
    """

    # Initialize dictionaries to store input (X) and output (y) sequences for train, validation, and test sets
    X_train = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }
    X_val = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }
    X_test = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }

    y_train = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }
    y_val = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }
    y_test = {
        "data": None,
        "seasonal": None,
        "trend": None,
        "residual": None,
        "exog": None,
        "indices": None,
    }

    # Define the data types that we will process (all keys except "indices")
    data_type = list(X_train.keys())
    data_type.remove("indices")

    # Iterate over each data type and create sequences for train, validation, and test sets
    for key in data_type:
        X_train[key], X_train["indices"], y_train[key], y_train["indices"] = (
            create_sequences_multivariate(
                train_data_with_features[key],
                train_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

        X_val[key], X_val["indices"], y_val[key], y_val["indices"] = (
            create_sequences_multivariate(
                val_data_with_features[key],
                val_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

        X_test[key], X_test["indices"], y_test[key], y_test["indices"] = (
            create_sequences_multivariate(
                test_data_with_features[key],
                test_data_with_features["index"],
                args.input_length,
                args.horizon,
            )
        )

    # Create TimeSeriesDataset objects for train, validation, and test sets
    train_dataset = TimeSeriesDataset(
        X_train["data"],
        X_train["seasonal"],
        X_train["trend"],
        X_train["residual"],
        y_train["data"],
        X_train["exog"],
        poi_tensor,
        mask,
    )
    val_dataset = TimeSeriesDataset(
        X_val["data"],
        X_val["seasonal"],
        X_val["trend"],
        X_val["residual"],
        y_val["data"],
        X_val["exog"],
        poi_tensor,
        mask,
    )
    test_dataset = TimeSeriesDataset(
        X_test["data"],
        X_test["seasonal"],
        X_test["trend"],
        X_test["residual"],
        y_test["data"],
        X_test["exog"],
        poi_tensor,
        mask,
    )

    return train_dataset, val_dataset, test_dataset
