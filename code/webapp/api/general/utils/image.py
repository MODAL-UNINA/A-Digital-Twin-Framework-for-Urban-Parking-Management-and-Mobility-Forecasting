import matplotlib

matplotlib.use("agg")

import base64
import io

from matplotlib.figure import Figure
from plotly import graph_objs as go  # type: ignore


def get_base64_image(fig: Figure) -> tuple[Figure, str]:
    # Create a buffer to store the plot image
    buffer = io.BytesIO()

    fig.savefig(  # type: ignore
        buffer, format="png", bbox_inches="tight", dpi=100
    )
    buffer.seek(0)

    # Encode the image as base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return fig, img_str


def get_base64_image_from_plotly(fig: go.Figure) -> str:
    buffer = io.BytesIO()

    fig.write_image(buffer, format="png")  # type: ignore
    buffer.seek(0)

    # Encode the image as base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return img_str
