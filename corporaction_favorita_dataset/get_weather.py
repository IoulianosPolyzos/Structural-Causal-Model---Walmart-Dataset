import pandas as pd
import requests
import numpy as np

# Exact coordinates for the 22 unique cities found in the Kaggle Favorita 'stores.csv'
ECUADOR_CITIES = {
    'Quito': {'lat': -0.2298, 'lon': -78.5250},
    'Guayaquil': {'lat': -2.1894, 'lon': -79.8891},
    'Cuenca': {'lat': -2.9001, 'lon': -79.0059},
    'Ambato': {'lat': -1.2417, 'lon': -78.6195},
    'Santo Domingo': {'lat': -0.2530, 'lon': -79.1753},
    'Manta': {'lat': -0.9676, 'lon': -80.7127},
    'Machala': {'lat': -3.2586, 'lon': -79.9605},
    'Latacunga': {'lat': -0.9316, 'lon': -78.6158},
    'Loja': {'lat': -3.9931, 'lon': -79.2042},
    'Riobamba': {'lat': -1.6698, 'lon': -78.6471},
    'Ibarra': {'lat': 0.3392, 'lon': -78.1222},
    'Guaranda': {'lat': -1.5926, 'lon': -79.0008},
    'Puyo': {'lat': -1.4837, 'lon': -77.9950},
    'Salinas': {'lat': -2.2232, 'lon': -80.9585},
    'Daule': {'lat': -1.8621, 'lon': -79.9776},
    'Babahoyo': {'lat': -1.8021, 'lon': -79.5344},
    'Quevedo': {'lat': -1.0286, 'lon': -79.4635},
    'Playas': {'lat': -2.6319, 'lon': -80.3880},
    'Libertad': {'lat': -2.2330, 'lon': -80.9103},
    'El Carmen': {'lat': -0.2762, 'lon': -79.4593},
    'Cayambe': {'lat': 0.0400, 'lon': -78.1452},
    'Esmeraldas': {'lat': 0.9592, 'lon': -79.6539}  # <-- Προστέθηκε
}


def map_wmo_code_to_condition(code):
    """
    Maps standard WMO weather codes to the 6 requested categories.
    """
    if pd.isna(code):
        return 'clear'

    code = int(code)
    if code in [0]:
        return 'clear'
    elif code in [1, 2, 3]:
        return 'clouds'
    elif code in [45, 48]:
        return 'fog'
    elif 51 <= code <= 67 or 80 <= code <= 82:
        return 'rain'
    elif 71 <= code <= 77 or code in [85, 86]:
        return 'snow'
    elif 95 <= code <= 99:
        return 'thunderstorm'
    else:
        return 'clear'


def fetch_weather_for_city(city, start_date, end_date):
    """
    Calls the Open-Meteo Archive API to fetch historical weather data for a specific city.
    """
    if city not in ECUADOR_CITIES:
        return pd.DataFrame()

    coords = ECUADOR_CITIES[city]
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Request hourly data to calculate accurate daily means and maximums
    params = {
        "latitude": coords['lat'],
        "longitude": coords['lon'],
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,weathercode,windspeed_10m",
        "timezone": "America/Guayaquil"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data for {city}: HTTP {response.status_code}")
        return pd.DataFrame()

    data = response.json()

    # Convert hourly API response to DataFrame
    df_hourly = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']).date,
        'temperature': data['hourly']['temperature_2m'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'precipitation': data['hourly']['precipitation'],
        'wind_speed': data['hourly']['windspeed_10m'],
        'weathercode': data['hourly']['weathercode']
    })

    # Aggregate hourly data into daily summaries
    df_daily = df_hourly.groupby('date').agg(
        temperature=('temperature', 'mean'),
        humidity=('humidity', 'mean'),
        precipitation=('precipitation', 'sum'),
        wind_speed=('wind_speed', 'max'),
        # Get the most severe weather code of the day
        weather_condition_code=('weathercode', 'max')
    ).reset_index()

    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['city'] = city

    # Apply mapping to convert WMO codes to text labels
    df_daily['weather_condition'] = df_daily['weather_condition_code'].apply(map_wmo_code_to_condition)
    df_daily.drop(columns=['weather_condition_code'], inplace=True)

    return df_daily


def add_weather_features(df):
    """
    Identifies cities and dates in the master dataset, fetches their weather data,
    and merges it back into the main DataFrame.
    """
    df = df.copy()

    # Ensure date column is datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    start_date = df['date'].min()
    end_date = df['date'].max()
    unique_cities = df['city'].dropna().unique()

    print(f"Fetching weather data from {start_date.date()} to {end_date.date()}...")

    all_weather_data = []

    for city in unique_cities:
        if city in ECUADOR_CITIES:
            print(f" -> Downloading weather for: {city}")
            city_weather = fetch_weather_for_city(city, start_date, end_date)
            if not city_weather.empty:
                all_weather_data.append(city_weather)
        else:
            print(f" -> WARNING: City '{city}' not found in coordinates list. Weather features will be NaN.")

    # Combine all city weather data into a single DataFrame
    weather_df = pd.concat(all_weather_data, ignore_index=True)

    # Merge weather data into the main dataset on date and city
    df = df.merge(weather_df, on=['date', 'city'], how='left')

    # Forward and backward fill any potential missing values within the same city
    weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'weather_condition']
    df[weather_cols] = df.groupby('city')[weather_cols].ffill().bfill()

    return df


