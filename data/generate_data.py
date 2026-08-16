"""
Energy Consumption Data Generator
=================================
Generates synthetic but realistic hourly energy consumption data
for 2 years (2022-2023) with the following features:
  - Temperature, humidity, wind speed (weather)
  - Hour of day, day of week, month (temporal)
  - Weekend and holiday flags
  - Energy consumption (target variable)

The energy consumption pattern is designed to mimic real-world behaviour:
  - U-shaped temperature effect (heating + cooling)
  - Morning and evening demand peaks
  - Lower weekend consumption
  - Seasonal variation

Author: Anwesha Mandal
"""

import pandas as pd
import numpy as np


def generate_energy_data(start_date="2022-01-01", periods=17520, seed=42):
    """
    Generate synthetic energy consumption data with realistic patterns.

    Parameters
    ----------
    start_date : str
        Start date for the time series.
    periods : int
        Number of hourly records (17520 ≈ 2 years).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 10 columns of energy consumption data.
    """
    np.random.seed(seed)

    # ── Datetime Index ──────────────────────────────────────────────
    dates = pd.date_range(start=start_date, periods=periods, freq="h")
    n = len(dates)

    # ── Weather Features ────────────────────────────────────────────
    # Temperature (°C): seasonal base + daily swing + random noise
    day_of_year = dates.dayofyear.values.astype(float)
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    hour = dates.hour.values.astype(float)
    daily_swing = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
    temperature = seasonal_temp + daily_swing + np.random.normal(0, 2, n)

    # Humidity (%): inversely related to temperature
    humidity = 70 - 0.5 * (temperature - 20) + np.random.normal(0, 8, n)
    humidity = np.clip(humidity, 20, 100)

    # Wind speed (km/h): exponential distribution (usually light winds)
    wind_speed = np.random.exponential(8, n) + 2
    wind_speed = np.clip(wind_speed, 0, 50)

    # ── Temporal Features ───────────────────────────────────────────
    month = dates.month.values
    day_of_week = dates.dayofweek.values          # 0=Mon … 6=Sun
    is_weekend = (day_of_week >= 5).astype(int)

    # Indian public holidays (simplified list)
    holiday_days = {
        (1, 1), (1, 26), (3, 8), (4, 14), (5, 1),
        (8, 15), (10, 2), (10, 24), (10, 25),
        (11, 1), (11, 14), (12, 25), (12, 31),
    }
    is_holiday = np.array(
        [1 if (d.month, d.day) in holiday_days else 0 for d in dates]
    )

    # ── Target: Energy Consumption (kWh) ────────────────────────────
    base = 150                                          # base load
    temp_effect = 2.5 * (temperature - 22) ** 2 / 10    # U-shaped
    hour_effect = (
        30 * np.exp(-0.5 * ((hour - 10) / 2) ** 2)     # morning peak
        + 40 * np.exp(-0.5 * ((hour - 20) / 2) ** 2)   # evening peak
        - 20 * np.exp(-0.5 * ((hour - 3) / 2) ** 2)    # night dip
    )
    weekend_effect = -25 * is_weekend
    holiday_effect = -30 * is_holiday
    humidity_effect = 0.3 * (humidity - 50)
    wind_effect = -0.5 * wind_speed
    noise = np.random.normal(0, 15, n)

    energy = (
        base + temp_effect + hour_effect + weekend_effect
        + holiday_effect + humidity_effect + wind_effect + noise
    )
    energy = np.clip(energy, 50, 500)

    # ── Assemble DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(
        {
            "datetime": dates,
            "temperature_c": np.round(temperature, 1),
            "humidity_pct": np.round(humidity, 1),
            "wind_speed_kmh": np.round(wind_speed, 1),
            "month": month,
            "day_of_week": day_of_week,
            "hour": hour.astype(int),
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "energy_consumption_kwh": np.round(energy, 2),
        }
    )

    # Introduce ~1 % missing values for realistic data-cleaning practice
    mask_temp = np.random.random(n) < 0.01
    df.loc[mask_temp, "temperature_c"] = np.nan

    mask_hum = np.random.random(n) < 0.005
    df.loc[mask_hum, "humidity_pct"] = np.nan

    return df


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating energy consumption dataset …")
    df = generate_energy_data()

    output_path = "energy_consumption.csv"
    df.to_csv(output_path, index=False)

    print(f"[OK] Dataset saved to {output_path}")
    print(f"   Shape       : {df.shape}")
    print(f"   Date range  : {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nSample rows:\n{df.head()}")
    print(f"\nDescriptive stats:\n{df.describe().round(2)}")
