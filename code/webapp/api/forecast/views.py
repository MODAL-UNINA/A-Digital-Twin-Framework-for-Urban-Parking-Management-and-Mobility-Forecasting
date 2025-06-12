from typing import TypedDict

import matplotlib

matplotlib.use("agg")

import pandas as pd

from api.general.utils.error_status import ErrorStatus
from api.general.utils.image import get_base64_image
from api.general.views import get_zone_dict

from .backend import do_get_prediction, get_date_range
from .data import ForecastDataType
from .startup import data_store as forecasting_data_store


class AvailableDatetimes(TypedDict):
    min_date: pd.Timestamp
    max_date: pd.Timestamp


class AvailableDates(TypedDict):
    min_date: str
    max_date: str


class ForecastTransactionsResult(TypedDict):
    forecast_transactions: str


class ForecastAmountResult(TypedDict):
    forecast_amount: str


class ForecastRoadResult(TypedDict):
    forecast_road: str


def get_available_forecasting_dates() -> AvailableDates | ErrorStatus:
    model_args = forecasting_data_store["forecast_data"]["model_args"]
    preprocessed_data = forecasting_data_store["preprocessed_data"]
    hourly_scaled_map = preprocessed_data["hourly_scaled"]

    out_date = AvailableDatetimes(
        min_date=pd.Timestamp("1970-01-01"),
        max_date=pd.Timestamp("2100-01-01"),
    )

    all_data_types: list[ForecastDataType] = ["transactions", "amount", "roads"]
    data_type: ForecastDataType
    for data_type in all_data_types:
        date_range = get_date_range(
            data_type=data_type,
            hourly_scaled_map=hourly_scaled_map,
            model_args=model_args,
        )
        if "error" in date_range:
            return ErrorStatus(error=date_range["error"])

        min_date = date_range["min_date"]
        max_date = date_range["max_date"]

        if out_date["min_date"] < min_date:
            out_date["min_date"] = min_date
        if date_range["max_date"] <= max_date:
            out_date["max_date"] = max_date

    return AvailableDates(
        min_date=out_date["min_date"].date().strftime("%Y-%m-%d"),
        max_date=out_date["max_date"].date().strftime("%Y-%m-%d"),
    )


def get_available_forecasting_parkingmeters(zone_name: str) -> dict[int, str]:
    zone_dict = get_zone_dict()
    parkingmeters = zone_dict[zone_name]["parcometro"]
    return {
        0: "All parking meters",
        **{int(p): f"Parking meter {int(p)}" for p in parkingmeters},
    }


def get_available_forecasting_roads(zone_name: str) -> dict[str, str]:
    zone_dict = get_zone_dict()
    road_ids = zone_dict[zone_name]["strade"]
    road_names = zone_dict[zone_name]["strade_name"]

    index_map = forecasting_data_store["forecast_data"]["index_map"]

    out = {
        str(int(road_id)): road_name
        for road_id, road_name in zip(road_ids, road_names, strict=False)
    }

    out = {road_id: v for road_id, v in out.items() if road_id in index_map["roads"]}
    return {"0": "All roads", **out}


def get_plot_forecast_transactions(
    zone_name: str, date: pd.Timestamp, parkingmeter_id: int | None
) -> ForecastTransactionsResult | ErrorStatus:
    import matplotlib.pyplot as plt

    date_range = get_available_forecasting_dates()
    if "error" in date_range:
        return date_range

    data = forecasting_data_store
    zone_dict = get_zone_dict()

    start_date = pd.to_datetime(  # type: ignore
        date_range["min_date"]
    )
    end_date = pd.to_datetime(  # type: ignore
        date_range["max_date"]
    )

    if date < start_date or date > end_date:
        return ErrorStatus(error=f"Date {date} is out of range.")

    fig = do_get_prediction(
        zone_name=zone_name,
        date=date,
        parkingmeter_id=parkingmeter_id,
        road_id=None,
        data_type_s="transactions",
        data=data,
        zone_dict=zone_dict,
    )

    if isinstance(fig, dict):
        return fig
    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return ForecastTransactionsResult(forecast_transactions=img_str)


def get_plot_forecast_amount(
    zone_name: str, date: pd.Timestamp, parkingmeter_id: int | None
) -> ForecastAmountResult | ErrorStatus:
    import matplotlib.pyplot as plt

    date_range = get_available_forecasting_dates()
    if "error" in date_range:
        return date_range

    data = forecasting_data_store
    zone_dict = get_zone_dict()

    start_date = pd.to_datetime(  # type: ignore
        date_range["min_date"]
    )
    end_date = pd.to_datetime(  # type: ignore
        date_range["max_date"]
    )

    if date < start_date or date > end_date:
        return ErrorStatus(error=f"Date {date} is out of range.")

    fig = do_get_prediction(
        zone_name=zone_name,
        date=date,
        parkingmeter_id=parkingmeter_id,
        road_id=None,
        data_type_s="amount",
        data=data,
        zone_dict=zone_dict,
    )

    if isinstance(fig, dict):
        return fig

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return ForecastAmountResult(forecast_amount=img_str)


def get_plot_forecast_roads(
    zone_name: str, date: pd.Timestamp, road_id: int | None
) -> ForecastRoadResult | ErrorStatus:
    import matplotlib.pyplot as plt

    date_range = get_available_forecasting_dates()
    if "error" in date_range:
        return date_range

    data = forecasting_data_store
    zone_dict = get_zone_dict()

    start_date = pd.to_datetime(  # type: ignore
        date_range["min_date"]
    )
    end_date = pd.to_datetime(  # type: ignore
        date_range["max_date"]
    )

    if date < start_date or date > end_date:
        return ErrorStatus(error=f"Date {date} is out of range.")

    fig = do_get_prediction(
        zone_name=zone_name,
        date=date,
        parkingmeter_id=None,
        road_id=road_id,
        data_type_s="roads",
        data=data,
        zone_dict=zone_dict,
    )

    if isinstance(fig, dict):
        return fig

    fig, img_str = get_base64_image(fig)
    plt.close(fig)

    return ForecastRoadResult(forecast_road=img_str)
