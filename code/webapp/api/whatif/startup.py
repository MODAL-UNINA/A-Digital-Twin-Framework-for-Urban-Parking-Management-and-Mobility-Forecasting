from django.conf import settings

# imports for data structures and preprocessing functions
from pathlib import Path  # isort: skip
from .data import WhatIfData, WhatIfLoadedData  # isort: skip
from sklearn.pipeline import Pipeline  # isort: skip


def get_data() -> WhatIfData:
    """
    Returns the loaded what-if scenario data.
    """
    return whatif_data_store


def load_whatif_files(data_path: Path) -> WhatIfData:
    from .loaddata import load_files

    whatif_data_path = data_path / "whatif"
    data = load_files(whatif_data_path)

    return WhatIfData(data=data, data_path=whatif_data_path)


# Public variables and functions
# Data storage
whatif_data_store = WhatIfData(
    data=WhatIfLoadedData(
        scenarios={},
        dict_zone={},
        distances_p={},
        distances_s={},
        p_coordinates={},
        p_scaler=Pipeline([]),
        s_coordinates={},
        s_scaler=Pipeline([]),
    ),
    data_path=Path(),
)


# Load data function
def load_data() -> None:
    whatif_data_store.update(load_whatif_files(settings.DATA_DIR))
