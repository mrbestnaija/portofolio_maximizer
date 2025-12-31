import numpy as np
import pandas as pd

from etl.visualizer import TimeSeriesVisualizer
from forcester_ts.instrumentation import describe_dataframe


def _sample_data() -> pd.DataFrame:
    idx = pd.date_range(start="2025-01-01", periods=120, freq="D")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "Close": np.linspace(100, 140, num=len(idx)) + rng.normal(0, 1, len(idx)),
            "Volume": rng.integers(800_000, 1_200_000, len(idx)),
            "Gold": np.linspace(1800, 1900, num=len(idx)),
            "Oil": np.linspace(70, 75, num=len(idx)),
        },
        index=idx,
    )
    return data


def test_comprehensive_dashboard_handles_context_columns():
    data = _sample_data()
    viz = TimeSeriesVisualizer()
    fig = viz.plot_comprehensive_dashboard(
        data,
        column="Close",
        market_columns=["Gold", "Oil"],
    )

    try:
        assert len(fig.axes) >= 10
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_comprehensive_dashboard_renders_metadata_panel():
    data = _sample_data()
    viz = TimeSeriesVisualizer()
    metadata = describe_dataframe(data, columns=["Close"])
    fig = viz.plot_comprehensive_dashboard(
        data,
        column="Close",
        market_columns=["Gold"],
        metadata=metadata,
    )

    try:
        assert any("Dataset Summary" in text.get_text() for text in fig.texts)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_autofmt_xdate_axis_kwarg_is_safe():
    """Matplotlib should accept axis kwarg after visualizer patches Figure.autofmt_xdate."""
    import matplotlib.pyplot as plt

    _ = TimeSeriesVisualizer()  # ensure module import / monkey patch executed
    idx = pd.date_range(start="2025-01-01", periods=10, freq="D")
    values = np.linspace(0, 1, len(idx))

    fig, ax = plt.subplots()
    try:
        ax.plot(idx, values)
        # Previously raised: FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'
        fig.autofmt_xdate(axis='x')
    finally:
        plt.close(fig)
