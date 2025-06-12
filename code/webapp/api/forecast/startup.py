from typing import Any

from django.conf import settings

from api.general.utils.loading import load_startup_files

# imports for data structures and preprocessing functions
from pathlib import Path  # isort: skip
import numpy as np  # isort: skip
import pandas as pd  # isort: skip
from sklearn.preprocessing import MinMaxScaler  # isort: skip
from .postprocess import postprocess  # isort: skip
from .data import (
    ForecastData,
    IndexMapData,
    OriginalForecastData,
    PreprocessedForecastData,
    WeatherData,
)  # isort: skip

# Private imports
from .startup_data import STARTUP_DATA as _STARTUP_DATA

# Private constants for module name and file mappings
_DATA_DIR = settings.DATA_DIR / "forecast"


# Private function to postprocess the data
def _postprocess(data: dict[str, Any]) -> ForecastData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    return postprocess(data, _DATA_DIR)


# Public variables and functions
# Data storage
data_store = ForecastData(
    forecast_data=OriginalForecastData(
        weather=WeatherData(
            prec=pd.DataFrame(),
            temp=pd.DataFrame(),
            wind=pd.DataFrame(),
            humidity=pd.DataFrame(),
        ),
        events=pd.DataFrame(),
        hourly={
            "transactions": pd.DataFrame(),
            "amount": pd.DataFrame(),
            "roads": pd.DataFrame(),
        },
        poi_dists={
            "transactions": pd.DataFrame(),
            "amount": pd.DataFrame(),
            "roads": pd.DataFrame(),
        },
        poi_categories={
            "transactions": pd.DataFrame(),
            "amount": pd.DataFrame(),
            "roads": pd.DataFrame(),
        },
        data_scaler={
            "transactions": MinMaxScaler(),
            "amount": MinMaxScaler(),
            "roads": MinMaxScaler(),
        },
        exog_scaler={
            "transactions": MinMaxScaler(),
            "amount": MinMaxScaler(),
            "roads": MinMaxScaler(),
        },
        model_args={},
        index_map=IndexMapData(
            parkimeters={},
            roads={},
        ),
    ),
    preprocessed_data=PreprocessedForecastData(
        hourly_scaled={
            "transactions": pd.DataFrame(),
            "amount": pd.DataFrame(),
            "roads": pd.DataFrame(),
        },
        exog_scaled={
            "transactions": pd.DataFrame(),
            "amount": pd.DataFrame(),
            "roads": pd.DataFrame(),
        },
        poi_tensor={
            "transactions": np.array([]),
            "amount": np.array([]),
            "roads": np.array([]),
        },
        mask={
            "transactions": np.array([]),
            "amount": np.array([]),
            "roads": np.array([]),
        },
    ),
    data_path=Path(),
)


# Load data function
def load_data() -> None:
    """
    Load the forecasting data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_startup_files(
        _DATA_DIR,
        _STARTUP_DATA,
    )

    data_store.update(**_postprocess(out_data))
