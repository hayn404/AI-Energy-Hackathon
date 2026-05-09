"""Data loading, cleaning, TOU price computation, and feature engineering."""

import numpy as np
import pandas as pd
from datetime import date

try:
    import holidays as hol_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ── System constants ──────────────────────────────────────────────────────────
ETA      = np.sqrt(0.90)   # one-way efficiency ≈ 0.9487
C_BAT    = 16.0            # kWh usable battery capacity
P_BAT_MAX  = 8.0           # kW max charge / discharge
P_GRID_MAX = 6.0           # kW grid connection limit
SOC_INIT   = 0.50          # initial SoC for 2025 optimisation
DT         = 0.25          # hours per 15-min interval
LAT, LON   = 46.17, 9.87  # Sondrio, Italy

# ── Italian TOU tariff ────────────────────────────────────────────────────────
F1 = 0.2540   # €/kWh  weekday 08:00–19:00
F2 = 0.2682   # €/kWh  weekday 07:00–08:00, 19:00–23:00 | Sat 07:00–23:00
F3 = 0.2440   # €/kWh  nights, Sundays, national holidays


def get_italian_holidays(years=(2024, 2025)):
    if HAS_HOLIDAYS:
        h = set()
        for y in years:
            h |= set(hol_lib.Italy(years=y).keys())
        return h
    # Fallback: hard-coded 2024 + 2025
    return {
        date(2024, 1, 1), date(2024, 1, 6), date(2024, 4, 1),
        date(2024, 4, 25), date(2024, 5, 1), date(2024, 6, 2),
        date(2024, 8, 15), date(2024, 11, 1), date(2024, 12, 8),
        date(2024, 12, 25), date(2024, 12, 26),
        date(2025, 1, 1), date(2025, 1, 6), date(2025, 4, 20),
        date(2025, 4, 25), date(2025, 5, 1), date(2025, 6, 2),
        date(2025, 8, 15), date(2025, 11, 1), date(2025, 12, 8),
        date(2025, 12, 25), date(2025, 12, 26),
    }


HOLIDAYS = get_italian_holidays()


def tou_price(ts: pd.Timestamp) -> float:
    """Return the Italian TOU buy price for a given timestamp."""
    d, h, dow = ts.date(), ts.hour, ts.weekday()   # 0=Mon, 6=Sun
    if dow == 6 or d in HOLIDAYS:                   # Sunday / holiday → F3
        return F3
    if dow == 5:                                    # Saturday
        return F2 if 7 <= h < 23 else F3
    # Weekday Mon–Fri
    if h < 7:    return F3
    if h == 7:   return F2
    if h < 19:   return F1
    if h < 23:   return F2
    return F3


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV, fix types, sort, and return a clean DataFrame."""
    df = pd.read_csv(path, sep=';', decimal=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Drop DST fall-back duplicates (keep first occurrence)
    df = df.drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)

    # Fill 8 missing sell prices
    df['Selling_price_eur_kwh'] = df['Selling_price_eur_kwh'].ffill()

    # Add TOU buy price
    df['buy_price'] = df['timestamp'].apply(tou_price)

    # Verify energy balance and flag badly imbalanced rows
    df['balance_err'] = (df['load_p'] - df['pv_p'] - df['grid_p'] - df['battery_p']).abs()

    return df


# ── Weather ───────────────────────────────────────────────────────────────────

def fetch_weather(cache_path: str = 'data/weather_sondrio.csv') -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo and resample to 15-min."""
    import os
    if os.path.exists(cache_path):
        df_w = pd.read_csv(cache_path, parse_dates=['timestamp'])
        print(f"  Weather: loaded from cache ({len(df_w)} rows)")
        return df_w

    try:
        import openmeteo_requests, requests_cache
        from retry_requests import retry

        cache_session = requests_cache.CachedSession('.omcache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": "2024-01-01", "end_date": "2025-12-31",
            "hourly": ["temperature_2m", "apparent_temperature",
                       "cloud_cover", "precipitation", "wind_speed_10m"],
            "timezone": "Europe/Rome",
        }
        resp = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        hourly = resp.Hourly()
        ts = pd.date_range(
            start=pd.Timestamp(hourly.Time(), unit="s"),
            periods=hourly.Variables(0).ValuesAsNumpy().shape[0],
            freq="1h",
        )
        df_w = pd.DataFrame({
            "timestamp":           ts,
            "temperature_2m":      hourly.Variables(0).ValuesAsNumpy(),
            "apparent_temperature":hourly.Variables(1).ValuesAsNumpy(),
            "cloud_cover":         hourly.Variables(2).ValuesAsNumpy(),
            "precipitation":       hourly.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m":      hourly.Variables(4).ValuesAsNumpy(),
        })

        # Resample hourly → 15-min by linear interpolation
        df_w = df_w.set_index('timestamp').resample('15min').interpolate('linear')
        df_w = df_w.reset_index()
        df_w.to_csv(cache_path, index=False)
        print(f"  Weather: fetched and cached ({len(df_w)} rows)")
        return df_w

    except Exception as e:
        print(f"  Weather fetch failed ({e}). Proceeding without weather features.")
        return pd.DataFrame(columns=['timestamp', 'temperature_2m',
                                     'apparent_temperature', 'cloud_cover',
                                     'precipitation', 'wind_speed_10m'])


# ── Feature engineering ────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Direct time slot (integer 0-95, complements sin/cos)
    'hour_slot',
    # Cyclical time
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'doy_sin',  'doy_cos',  'month_sin', 'month_cos',
    # Calendar
    'is_weekend', 'is_holiday',
    # Short-term lags — fill gap between 15 min and 1 hour
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'lag_16',
    # Intermediate lags (6h, 12h — fill gap between 4h and 24h)
    'lag_24', 'lag_48',
    # Long lags
    'lag_96', 'lag_192', 'lag_672', 'lag_1344',
    # Rolling windows — short (current activity) + long (baseline level)
    'roll_mean_4', 'roll_mean_16', 'roll_mean_96',
    'roll_std_96', 'roll_mean_672', 'roll_max_96',
    # PV lags (occupancy proxy: low PV on cloudy days correlates with home presence)
    'pv_lag_1', 'pv_lag_4', 'pv_lag_96',
    # Momentum / rate-of-change
    'delta_1', 'delta_4', 'delta_96', 'delta_672',
    # Hour-of-week index (shortcut for weekday × time-of-day interactions)
    'hour_of_week',
    # Price signal
    'buy_price',
    # Weather (added dynamically if available)
]

WEATHER_COLS = ['temperature_2m', 'apparent_temperature', 'HDD', 'CDD', 'cloud_cover']


def build_features(df: pd.DataFrame, weather_df: pd.DataFrame | None = None,
                   include_weather: bool = True) -> pd.DataFrame:
    """Add all temporal, lag, rolling, and weather features to the DataFrame."""
    df = df.copy()
    ts = df['timestamp']

    # ── Direct time slot ────────────────────────────────────────────────────
    interval = ts.dt.hour * 4 + ts.dt.minute // 15   # 0..95
    df['hour_slot'] = interval.astype(np.float32)

    # ── Cyclical encodings ──────────────────────────────────────────────────
    df['hour_sin']   = np.sin(2 * np.pi * interval / 96)
    df['hour_cos']   = np.cos(2 * np.pi * interval / 96)
    df['dow_sin']    = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['dow_cos']    = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['doy_sin']    = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    df['doy_cos']    = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    df['month_sin']  = np.sin(2 * np.pi * ts.dt.month / 12)
    df['month_cos']  = np.cos(2 * np.pi * ts.dt.month / 12)

    # ── Calendar flags ──────────────────────────────────────────────────────
    df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(np.float32)
    df['is_holiday'] = ts.dt.date.isin(HOLIDAYS).astype(np.float32)

    # ── Load lag features (no leakage) ──────────────────────────────────────
    load = df['load_p']
    for lag in [1, 2, 3, 4, 8, 16, 24, 48, 96, 192, 672, 1344]:
        df[f'lag_{lag}'] = load.shift(lag)

    # ── Rolling statistics (shift(1) before rolling avoids leakage) ─────────
    s = load.shift(1)
    df['roll_mean_4']   = s.rolling(4).mean()    # 1-hour mean
    df['roll_mean_16']  = s.rolling(16).mean()   # 4-hour mean
    df['roll_mean_96']  = s.rolling(96).mean()
    df['roll_std_96']   = s.rolling(96).std()
    df['roll_mean_672'] = s.rolling(672).mean()
    df['roll_max_96']   = s.rolling(96).max()

    # ── PV lags (occupancy/weather proxy) ────────────────────────────────────
    pv = df['pv_p']
    for lag in [1, 4, 96]:
        df[f'pv_lag_{lag}'] = pv.shift(lag)

    # ── Momentum / rate-of-change features ───────────────────────────────────
    df['delta_1']   = load.shift(1) - load.shift(2)    # 15-min momentum
    df['delta_4']   = load.shift(1) - load.shift(5)    # 1-hour trend
    df['delta_96']  = load.shift(1) - load.shift(97)   # deviation from yesterday
    df['delta_672'] = load.shift(1) - load.shift(673)  # deviation from last week

    # ── Hour-of-week index (0..671) ───────────────────────────────────────────
    df['hour_of_week'] = (ts.dt.dayofweek * 96 + interval).astype(np.float32)

    # ── Weather ─────────────────────────────────────────────────────────────
    has_weather = (
        include_weather
        and weather_df is not None
        and len(weather_df) > 0
        and 'temperature_2m' in weather_df.columns
    )
    if has_weather:
        wdf = weather_df.copy()
        wdf['timestamp'] = pd.to_datetime(wdf['timestamp'])
        df = df.merge(wdf[['timestamp', 'temperature_2m',
                            'apparent_temperature', 'cloud_cover']],
                      on='timestamp', how='left')
        df['temperature_2m']       = df['temperature_2m'].ffill().bfill()
        df['apparent_temperature'] = df['apparent_temperature'].ffill().bfill()
        df['cloud_cover']          = df['cloud_cover'].ffill().bfill()
        df['HDD'] = (18.0 - df['temperature_2m']).clip(lower=0)
        df['CDD'] = (df['temperature_2m'] - 22.0).clip(lower=0)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return available feature columns (adds weather cols if present)."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    for c in WEATHER_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols
