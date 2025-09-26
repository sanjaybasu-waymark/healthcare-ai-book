"""
Chapter 24 - Example 3
Extracted from Healthcare AI Implementation Guide
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def generate_synthetic_data(start_date, periods, freq, trend_slope, seasonal_amplitude, noise_std):
    """
    Generates synthetic patient demand data with trend and seasonality.
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    time_index = np.arange(periods)

    \# Trend
    trend = trend_slope * time_index

    \# Seasonality (e.g., weekly or monthly)
    seasonal = seasonal_amplitude * np.sin(time_index * (2 * np.pi / (periods / 4))) \# Example: 4 cycles over the period

    \# Noise
    noise = np.random.normal(0, noise_std, periods)

    demand = (trend + seasonal + noise).astype(int)
    demand[demand < 0] = 0 \# Ensure demand is non-negative

    data = pd.DataFrame({
        \'Date\': dates,
        \'Demand\': demand
    })
    data = data.set_index(\'Date\')
    return data

def forecast_patient_demand(data, order=(5,1,0), train_size_ratio=0.8):
    """
    Forecasts patient demand using an ARIMA model.

    Args:
        data (pd.DataFrame): Time series data with a \'Demand\' column and Date index.
        order (tuple): The (p,d,q) order of the ARIMA model.
        train_size_ratio (float): Proportion of data to use for training.

    Returns:
        tuple: (forecast_df, model_fit, mse) containing forecasted data, fitted model, and Mean Squared Error.
    """
    if not isinstance(data, pd.DataFrame) or \'Demand\' not in data.columns or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input data must be a pandas DataFrame with a 'Demand' column and a DatetimeIndex.")
    if len(data) < 10: \# ARIMA requires a reasonable amount of data
        raise ValueError("Insufficient data for ARIMA modeling. At least 10 observations are recommended.")
    if not all(pd.to_numeric(data[\'Demand\'], errors=\'coerce\').notna()):
        raise ValueError("\'Demand\' column must contain numeric values.")

    \# Split data into training and testing sets
    train_size = int(len(data) * train_size_ratio)
    if train_size == 0 or train_size >= len(data):
        raise ValueError("Training set size is invalid. Adjust train_size_ratio.")
    train, test = data[0:train_size], data[train_size:]

    try:
        \# Fit ARIMA model
        model = ARIMA(train[\'Demand\'], order=order)
        model_fit = model.fit()

        \# Make predictions
        start_index = len(train)
        end_index = len(data) - 1
        predictions = model_fit.predict(start=start_index, end=end_index, typ=\'levels\')

        \# Create a DataFrame for the forecast
        forecast_df = pd.DataFrame({
            \'Actual\': test[\'Demand\'],
            \'Predicted\': predictions
        }, index=test.index)

        \# Evaluate the model
        mse = mean_squared_error(test[\'Demand\'], predictions)

        return forecast_df, model_fit, mse
    except Exception as e:
        raise RuntimeError(f"ARIMA model fitting or prediction failed: {e}")

if __name__ == "__main__":
    \# Generate synthetic daily patient demand data for 2 years
    synthetic_data = generate_synthetic_data(
        start_date=\'2023-01-01\',
        periods=730, \# 2 years of daily data
        freq=\'D\',
        trend_slope=0.1,
        seasonal_amplitude=20,
        noise_std=5
    )

    print("Synthetic Data Head:")
    print(synthetic_data.head())
    print("\nSynthetic Data Tail:")
    print(synthetic_data.tail())

    try:
        \# Forecast patient demand
        forecast_results, model_fit, mse = forecast_patient_demand(synthetic_data, order=(5,1,0))

        print("\nForecast Results Head:")
        print(forecast_results.head())
        print(f"\nMean Squared Error of the forecast: {mse:.2f}")

        \# Plotting the results
        plt.figure(figsize=(12, 6))
        plt.plot(synthetic_data.index, synthetic_data[\'Demand\'], label=\'Historical Demand\')
        plt.plot(forecast_results.index, forecast_results[\'Actual\'], label=\'Actual Test Demand\', color=\'orange\')
        plt.plot(forecast_results.index, forecast_results[\'Predicted\'], label=\'Predicted Demand\', color=\'green\', linestyle=\'--\')
        plt.title(\'Patient Demand Forecasting with ARIMA\')
        plt.xlabel(\'Date\')
        plt.ylabel(\'Demand\')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(\'patient_demand_forecast.png\')
        print("\nPlot saved to patient_demand_forecast.png")

        print("\nARIMA Model Summary:")
        print(model_fit.summary())

    except (ValueError, RuntimeError) as e:
        print(f"An error occurred during forecasting: {e}")

    \# Example of error handling: what if data is too short?
    print("\n--- Example with insufficient data for ARIMA ---")
    short_data = generate_synthetic_data(start_date=\'2024-01-01\', periods=5, freq=\'D\', trend_slope=0.1, seasonal_amplitude=5, noise_std=1)
    try:
        forecast_patient_demand(short_data)
    except ValueError as e:
        print(f"Caught expected error for short data: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    \# Example of error handling: non-numeric data
    print("\n--- Example with non-numeric data ---")
    non_numeric_data = synthetic_data.copy()
    non_numeric_data.loc[non_numeric_data.index<sup>5</sup>, \'Demand\'] = \'abc\'
    try:
        forecast_patient_demand(non_numeric_data)
    except ValueError as e:
        print(f"Caught expected error for non-numeric data: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")