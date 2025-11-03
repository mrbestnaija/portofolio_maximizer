import numpy as np
import pandas as pd

from etl.visualizer import TimeSeriesVisualizer


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
