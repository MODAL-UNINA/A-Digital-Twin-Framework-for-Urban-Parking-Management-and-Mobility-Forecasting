from typing import Any, TypedDict

from django.conf import settings

from api.general.utils.loading import (
    JsonMapping,
    PklMapping,
    load_files,
)

# Private constants for module name and file mappings
_MODULE_NAME = "Agent calendar"

_PKL_FILES_DATA = PklMapping(calendar="calendar.pkl")

_JSON_FILES_DATA = JsonMapping()


# Data structures
class AgentCalendarData(TypedDict):
    """
    TypedDict for the agent calendar data structure.
    """

    start: int
    end: int
    zone: list[str]


AgentCalendarDataMapping = dict[str, dict[str, AgentCalendarData]]


class CalendarData(TypedDict):
    """
    TypedDict for the calendar data structure.
    """

    calendar: AgentCalendarDataMapping


# Private function to postprocess the data
def _postprocess_calendar_data(calendar: AgentCalendarData) -> AgentCalendarData:
    import pandas as pd

    dates = list(calendar.keys())
    min_date = min(dates)
    max_date = max(dates)

    # assert that the available dates are in an interval
    if min_date >= max_date:
        raise ValueError("The minimum date must be less than the maximum date.")

    if (pd.Timestamp(max_date) - pd.Timestamp(min_date)).days + 1 != len(dates):
        raise ValueError(
            "The number of dates does not match the interval between min and max date."
        )

    return calendar


def _postprocess(data: dict[str, Any]) -> CalendarData:
    """
    Postprocess the loaded data.
    This function is called after loading the data from files.
    """
    data["calendar"] = _postprocess_calendar_data(data["calendar"])
    return CalendarData(**data)


# Public variables and functions
# Data storage
data_store = CalendarData(calendar={})


# Load data function
def load_data() -> None:
    """
    Load the agent calendar data in memory.
    This function is called at the startup of the application.
    """
    out_data = load_files(
        settings.DATA_DIR / "agent_calendar",
        _MODULE_NAME,
        _PKL_FILES_DATA,
        _JSON_FILES_DATA,
    )

    data_store.update(**_postprocess(out_data))
