#%%
import os
import pandas as pd
import numpy as np
import importlib

# Import custom modules
import data_join
import get_weather
import plotting
import data_preprocessing
import feature_engineering

# Reload modules to ensure latest changes are applied
importlib.reload(feature_engineering)
importlib.reload(plotting)
importlib.reload(data_preprocessing)
importlib.reload(data_join)
importlib.reload(get_weather)

from feature_engineering import *
from data_preprocessing import *
from get_weather import *
from plotting import *
from data_join import create_master_dataset

#%%

print("1. LOADING DATASET")

# Load Favorita master dataset (using nrows for quick testing)
data = create_master_dataset()
#%%

print("2. FEATURE ENGINEERING")


print("Creating time-based features...")
data = create_time_based_features(data, include_cyclic_features=True)

print("\nChecking actual date format...")
sample_dates_train = data['date'].head(10).tolist() if 'date' in data.columns else "No Date column"
print(f"Sample dates from dataset: {sample_dates_train}")
print(f"Date column dtype: {data['date'].dtype}")

if 'date' in data.columns:
    unique_dates = data['date'].astype(str).unique()[:5]
    print(f"Unique date samples: {unique_dates}")

print("\nChecking existing columns before holiday features...")
print(f"Total columns: {len(data.columns)}")

print("\nCreating holiday features...")
data = create_holiday_features(data)
print(f"Total columns after holiday features: {len(data.columns)}")

print("\nFetching and adding weather features...")
data = add_weather_features(data)
print(f"Total columns after weather features: {len(data.columns)}")

print("\nFinal data check:")
print(f"Dataset shape: {data.shape}")
print(f"Remaining NaN values: {data.isna().sum().sum()}")

#%%
print("\nFinal data types:")
print(data.dtypes)

#%%

print("3. DATA CLEANING")


print("\nNegatives before cleaning:")
neg_report = check_negatives(data)
print(neg_report)

neg_cols = ["unit_sales"]
data = remove_negatives(data,neg_cols)

print("\nNegatives after cleaning:")
neg_report = check_negatives(data)
print(neg_report)

print("\nZeros:")
zero_percent = check_zeros(data)
print(zero_percent)

#%%
print("\nOutliers Check:")
outliers_report = check_outliers(data)
print(outliers_report)

#%%

print("4. FINALIZATION & TARGET EXTRACTION")

micro_sales = ((data['unit_sales'] > 0) & (data['unit_sales'] < 1)).sum()
small_sales = ((data['unit_sales'] >= 1) & (data['unit_sales'] < 10)).sum()
total_rows = len(data)

print(f"Total rows: {total_rows}")
print(f"Unit Sales between 0 and 1 (0, 1): {micro_sales} ({(micro_sales/total_rows)*100:.2f}%)")
print(f"Unit Sales between 1 and 10 [1, 10): {small_sales} ({(small_sales/total_rows)*100:.2f}%)")

# Extract final feature list
features, target = get_features(
    data=data,
    include_enhanced_holiday=True,
    include_cyclic_features=True
)
print(f"\nUsing {len(features)} features")
print(f"Target: {target}")

#%%
print("\nDataset columns after feature engineering:")
print(data.columns.tolist())

#%%
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"\nMissing features: {missing_features}")
    print("Available features:")
    for col in data.columns:
        print(f"   - {col}")
else:
    print(f"\nAll {len(features)} expected features are available in the dataset")

#%%

print("5. SPLIT ANALYSIS & PLOTTING")


df_exp = data.copy()

# --- 1. Store type split ---
# Favorita has types A, B, C, D, E. Group A/B vs the rest
df_exp['set_store_type'] = df_exp['store_type'].apply(lambda x: 'train' if x in ['A', 'B'] else 'test')

# --- 2. Store number split ---
# Favorita has 54 stores
df_exp['set_store_number'] = df_exp['store_nbr'].apply(lambda x: 'train' if x <= 30 else 'test')

# --- 3. Holiday split ---
df_exp['set_holiday'] = df_exp['IsHoliday'].apply(lambda x: 'train' if x == 0 else 'test')

# --- 4. Christmas split ---
df_exp['set_christmas'] = df_exp['Is_Christmas_Season'].apply(lambda x: 'train' if x == 0 else 'test')

# --- 5. Season split ---
df_exp['set_season'] = df_exp['Season'].apply(lambda x: 'test' if x == 3 else 'train')

# --- 6. City split ---
test_cities = ["Cuenca", "Manta", "Machala"]
df_exp['set_city'] = df_exp['city'].apply(lambda x: 'test' if x in test_cities else 'train')

# --- 7. Weather split ---
df_exp['set_weather'] = df_exp['weather_condition'].apply(lambda x: 'test' if x == 'clear' else 'train')


print("\nSTORE TYPE SPLIT")
df_exp['set'] = df_exp['set_store_type']
plot_store_type_split(df_exp)

print("\nSTORE NUMBER SPLIT")
df_exp['set'] = df_exp['set_store_number']
plot_store_number_split(df_exp)

print("\nHOLIDAY SPLIT")
df_exp['set'] = df_exp['set_holiday']
plot_holiday_split(df_exp)

print("\nCHRISTMAS SPLIT")
df_exp['set'] = df_exp['set_christmas']
plot_christmas_split(df_exp)

print("\nSEASON SPLIT")
df_exp['set'] = df_exp['set_season']
plot_season_split(df_exp)

print("\nCITY SPLIT")
df_exp['set'] = df_exp['set_city']
plot_city_split(df_exp)

print("\nWEATHER SPLIT")
df_exp['set'] = df_exp['set_weather']
plot_weather_split(df_exp)

print ("\nSales Over Time")
plot_sales_over_time(data)

print("\nCorrelation Heatmap")
plot_correlation_heatmap(data)

print("\nSales Distribution")
plot_distribution(data, "unit_sales")

#%%

print("6. SAVING FINAL DATASET")

print(f"Final dataset shape before saving: {data.shape}")

# Drop Unnamed columns if they accidentally appeared during merges
unnamed_columns = [col for col in data.columns if 'Unnamed' in col]
if unnamed_columns:
    print(f"Removing useless columns: {unnamed_columns}")
    data = data.drop(columns=unnamed_columns)

# Save to CSV
output_file = "final_data_favorita.csv"
data.to_csv(output_file, index=False, encoding="utf-8", float_format="%.4f")
print(f"Success! Final dataset saved to {output_file}")

data.to_parquet("final_data_favorita.parquet", index=False)
print(f"Success! Final parquet-dataset saved")

