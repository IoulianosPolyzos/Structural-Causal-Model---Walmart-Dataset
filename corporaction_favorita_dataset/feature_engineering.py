import pandas as pd
import numpy as np

def create_time_based_features(df, include_cyclic_features=True):
    df = df.copy()

    # Date handling
    if 'date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    valid_mask = df['date'].notna()

    if not valid_mask.any():
        print("No valid dates found")
        return df

    # Basic time features
    df.loc[valid_mask, 'DayOfWeek'] = df.loc[valid_mask, 'date'].dt.dayofweek
    df.loc[valid_mask, 'Month'] = df.loc[valid_mask, 'date'].dt.month
    df.loc[valid_mask, 'WeekOfYear'] = df.loc[valid_mask, 'date'].dt.isocalendar().week.astype(int)
    df.loc[valid_mask, 'Quarter'] = df.loc[valid_mask, 'date'].dt.quarter
    df.loc[valid_mask, 'Year'] = df.loc[valid_mask, 'date'].dt.year

    # Season
    df.loc[valid_mask, 'Season'] = ((df.loc[valid_mask, 'Month'] - 1) // 3) + 1

    # Progress features
    df.loc[valid_mask, 'Year_Progress'] = (
        df.loc[valid_mask, 'date'].dt.dayofyear / 365.0
    )

    df.loc[valid_mask, 'Month_Progress'] = (
        df.loc[valid_mask, 'date'].dt.day /
        df.loc[valid_mask, 'date'].dt.days_in_month
    )

    # Binary features
    df.loc[valid_mask, 'Is_Month_Start'] = df.loc[valid_mask, 'date'].dt.is_month_start.astype(int)
    df.loc[valid_mask, 'Is_Month_End'] = df.loc[valid_mask, 'date'].dt.is_month_end.astype(int)

    # Special periods
    df.loc[valid_mask, 'Is_Christmas_Season'] = (
        (df.loc[valid_mask, 'Month'] == 12) &
        (df.loc[valid_mask, 'date'].dt.day >= 15)
    ).astype(int)

    df.loc[valid_mask, 'Is_Summer'] = (
        df.loc[valid_mask, 'Month'].between(6, 8)
    ).astype(int)

    if include_cyclic_features:
        df = add_cyclic_features(df)

    return df


def add_cyclic_features(df):
    df = df.copy()

    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['WeekOfYear_sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
    df['WeekOfYear_cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)

    df['Season_sin'] = np.sin(2 * np.pi * df['Season'] / 4)
    df['Season_cos'] = np.cos(2 * np.pi * df['Season'] / 4)

    return df


def get_features(data=None, include_enhanced_holiday=True, include_cyclic_features=True):

    base_features = [
        'store_nbr', 'item_nbr', 'onpromotion',
        'store_type', 'cluster', 'family', 'class', 'perishable',
        'dcoilwtico', 'earthquake_impact', 'transactions'
    ]

    weather_features = [
        'temperature', 'humidity', 'precipitation', 'wind_speed', 'weather_condition'
    ]

    time_features = [
        'DayOfWeek', 'Month', 'WeekOfYear', 'Quarter', 'Season',
        'Year_Progress', 'Month_Progress', 'Is_Month_Start',
        'Is_Month_End', 'Is_Christmas_Season', 'Is_Summer'
    ]

    cyclic_features = []
    if include_cyclic_features:
        cyclic_features = [
            'DayOfWeek_sin', 'DayOfWeek_cos',
            'Month_sin', 'Month_cos',
            'WeekOfYear_sin', 'WeekOfYear_cos',
            'Season_sin', 'Season_cos'
        ]

    holiday_features = ['IsHoliday']
    if include_enhanced_holiday:
        holiday_features.extend(['holiday_proximity', 'is_near_holiday', 'holiday_weight'])

    all_features = (base_features + weather_features + time_features +
                    cyclic_features + holiday_features)

    all_features = list(dict.fromkeys(all_features))

    if data is not None:
        all_features = [f for f in all_features if f in data.columns]

    target = 'unit_sales'

    if data is not None:
        print(f"\n Feature breakdown:")
        print(f"   Base features: {len([f for f in all_features if f in base_features])}")
        print(f"   Weather features: {len([f for f in all_features if f in weather_features])}")
        print(f"   Time features: {len([f for f in all_features if f in time_features])}")
        print(f"   Cyclic features: {len([f for f in all_features if f in cyclic_features])}")
        print(f"   Holiday features: {len([f for f in all_features if f in holiday_features])}")
        print(f"   Total features: {len(all_features)}")

    return all_features, target




def create_holiday_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    holiday_dates = df.loc[df["IsHoliday"] == 1, "date"].dropna().unique()

    holiday_dates = pd.to_datetime(holiday_dates)

    if len(holiday_dates) == 0:
        df["holiday_proximity"] = 0
        df["is_near_holiday"] = 0
        df["holiday_weight"] = 0
        return df

    holiday_dates = np.array(holiday_dates, dtype="datetime64[D]")

    def min_distance(date):
        if pd.isna(date):
            return np.nan

        date = np.datetime64(pd.to_datetime(date), "D")
        return np.min(np.abs(holiday_dates - date)).astype(int)

    df["holiday_proximity"] = df["date"].apply(min_distance)

    df["is_near_holiday"] = (df["holiday_proximity"] <= 7).astype(int)

    df["holiday_weight"] = (
        df["IsHoliday"] * 2.0 +
        df["is_near_holiday"] * (1.5 / (df["holiday_proximity"] + 1))
    )

    return df





