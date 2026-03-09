"""
SmartCertify ML — Time Series Forecasting
Verification trend analysis using ARIMA and basic forecasting.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from app.config.settings import TIMESERIES_DATASET_PATH, PLOTS_DIR

logger = logging.getLogger(__name__)


def load_timeseries_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load and prepare time series data."""
    if path is None:
        path = TIMESERIES_DATASET_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def fit_arima(df: pd.DataFrame, order: tuple = (2, 1, 2)) -> Dict[str, Any]:
    """Fit ARIMA model and generate forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(df["verification_count"], order=order)
        fitted = model.fit()

        # Forecast 30 days
        forecast = fitted.forecast(steps=30)
        conf_int = fitted.get_forecast(steps=30).conf_int()

        # Determine trend
        recent_mean = df["verification_count"].tail(30).mean()
        forecast_mean = forecast.mean()
        trend = "increasing" if forecast_mean > recent_mean * 1.05 else \
                "decreasing" if forecast_mean < recent_mean * 0.95 else "stable"

        # Save forecast plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df.index[-90:], df["verification_count"].tail(90),
                    label="Actual", color="#3498db", lw=2)

            forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
            ax.plot(forecast_dates, forecast.values,
                    label="Forecast", color="#e74c3c", lw=2, linestyle="--")
            ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                           alpha=0.2, color="#e74c3c")

            ax.set_title("Verification Volume Forecast (ARIMA)", fontsize=15, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Verification Count", fontsize=12)
            ax.legend(fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)

            plot_path = PLOTS_DIR / "timeseries_forecast.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not save forecast plot: {e}")

        return {
            "model": "ARIMA",
            "order": order,
            "forecast_30d": forecast.values.tolist(),
            "forecast_mean": round(float(forecast_mean), 2),
            "trend": trend,
            "aic": round(float(fitted.aic), 2),
            "bic": round(float(fitted.bic), 2),
        }

    except ImportError:
        logger.warning("statsmodels not installed, using simple moving average fallback")
        return _simple_forecast(df)
    except Exception as e:
        logger.error(f"ARIMA fit failed: {e}")
        return _simple_forecast(df)


def _simple_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    """Simple moving average forecast as fallback."""
    ma_30 = df["verification_count"].rolling(30).mean()
    last_ma = float(ma_30.iloc[-1])
    trend_slope = float(ma_30.diff().tail(7).mean())

    forecast = [last_ma + trend_slope * i for i in range(1, 31)]
    forecast_mean = np.mean(forecast)

    recent_mean = df["verification_count"].tail(30).mean()
    trend = "increasing" if forecast_mean > recent_mean * 1.05 else \
            "decreasing" if forecast_mean < recent_mean * 0.95 else "stable"

    return {
        "model": "moving_average",
        "forecast_30d": [round(f, 2) for f in forecast],
        "forecast_mean": round(forecast_mean, 2),
        "trend": trend,
    }


def get_verification_trends(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Get verification trend analysis and forecast."""
    if df is None:
        try:
            df = load_timeseries_data()
        except FileNotFoundError:
            return {"error": "Time series data not available"}

    # Current stats
    recent_7d = df["verification_count"].tail(7).mean()
    recent_30d = df["verification_count"].tail(30).mean()
    overall_mean = df["verification_count"].mean()

    # Fit model and forecast
    forecast_result = fit_arima(df)

    return {
        "current_stats": {
            "avg_7d": round(float(recent_7d), 2),
            "avg_30d": round(float(recent_30d), 2),
            "overall_mean": round(float(overall_mean), 2),
            "total_days": len(df),
        },
        "forecast": forecast_result,
    }


def main():
    """Run time series analysis."""
    from app.config.settings import TIMESERIES_DATASET_PATH

    if not Path(TIMESERIES_DATASET_PATH).exists():
        from app.data.generate_synthetic import generate_timeseries_dataset
        df = generate_timeseries_dataset()
        df.to_csv(TIMESERIES_DATASET_PATH, index=False)

    print("Running time series analysis...")
    results = get_verification_trends()
    print(f"\n✅ Time Series Analysis:")
    print(f"   Current Stats: {results['current_stats']}")
    print(f"   Trend: {results['forecast']['trend']}")
    print(f"   30-day forecast mean: {results['forecast']['forecast_mean']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
