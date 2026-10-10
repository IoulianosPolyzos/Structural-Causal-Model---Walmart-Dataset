#%%
import join_train_features
import get_weather
import plotting
import data_preprocessing
import feature_engineering
import importlib
importlib.reload(feature_engineering)
importlib.reload(plotting)
importlib.reload(data_preprocessing)
importlib.reload(join_train_features)
importlib.reload(get_weather)
from feature_engineering import *
from data_preprocessing import *
from get_weather import *
from plotting import *
import pandas as pd
from join_train_features import create_standard_df
import pandas as pd
import kagglehub

#%%
path = kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast")

print("Dataset path:", path)

file_path = os.path.join(path, "train.csv")

data = pd.read_csv(file_path)

data = create_standard_df(data, dataset_path=path)

micro_sales = ((data['Weekly_Sales'] > 0) & (data['Weekly_Sales'] < 1)).sum()
small_sales = ((data['Weekly_Sales'] >= 1) & (data['Weekly_Sales'] < 10)).sum()
total_rows = len(data)

print(f"Total rows: {total_rows}")
print(f"Weekly Sales that are between 0 and 1 (0, 1): {micro_sales} ({(micro_sales/total_rows)*100:.2f}%)")
print(f"Weekly Sales that are between 1 and 10 [1, 10): {small_sales} ({(small_sales/total_rows)*100:.2f}%)")

#%%
print("\ Creating enhanced features...")



print("Creating time-based features...")
data = create_time_based_features(data, include_cyclic_features=True)
print(" Checking actual date format...")

sample_dates_train = data['Date'].head(10).tolist() if 'Date' in data.columns else "No Date column"

print(f"Sample dates from file: {sample_dates_train}")

print(f"Date column dtype: {data['Date'].dtype}")

if 'Date' in data.columns:
    unique_dates = data['Date'].astype(str).unique()[:5]
    print(f"Unique date samples: {unique_dates}")


print("\n Checking existing columns before one-hot encoding...")
print(f"Total columns: {len(data.columns)}")
print("All columns:", data.columns.tolist())
print("Creating holiday features...")
data = create_holiday_features(data)

print(f"Total columns: {len(data.columns)}")
print("All columns:", data.columns.tolist())

print("\n Final data check:")
print(f"Dataset shape: {data.shape}")
print(f"Remaining NaN values: {data.isna().sum().sum()}")
#%%
print("\nFinal data types:")
print(data.dtypes)
#%%
neg_report = check_negatives(data)
print("Negatives")
print(neg_report)
#%%
neg_cols = ['Weekly_Sales','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
# Markdowns contain a lot of NaNs so in order not to remove NaNs rows , we insert assign 0 to these rows
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in markdown_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)
data = remove_negatives(data,neg_cols)

print("Zeros:")
zero_percent = check_zeros(data)

print(zero_percent)

data = data[data['Weekly_Sales'] > 0]

print("Zeros after cleaning...")
zero_percent = check_zeros(data)

print(zero_percent)

#%%
print("Negatives after cleaning ...")
neg_report = check_negatives(data)
print(neg_report)
#%%
outliers_report = check_outliers(data)
print(outliers_report)
#%%
print(type(data))
print(data is None)
#%%
data = get_city_and_weather(data)
#%%

print("Finalization:")
micro_sales = ((data['Weekly_Sales'] > 0) & (data['Weekly_Sales'] < 1)).sum()
small_sales = ((data['Weekly_Sales'] >= 1) & (data['Weekly_Sales'] < 10)).sum()
total_rows = len(data)

print(f"Total rows: {total_rows}")
print(f"Weekly Sales that are between 0 and 1 (0, 1): {micro_sales} ({(micro_sales/total_rows)*100:.2f}%)")
print(f"Weekly Sales that are between 1 and 10 [1, 10): {small_sales} ({(small_sales/total_rows)*100:.2f}%)")
# Feature engineering
features, target = get_features(
    data=data,
    include_enhanced_holiday=True,
    include_cyclic_features=True
)
print(f"\n Using {len(features)} features")
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
    print(f"\nAll {len(features)} features are available in the dataset")


#%%
import numpy as np


df_exp = data.copy()


# --- 1. Store type.sh split ---
df_exp['set_store_type'] = df_exp['Type'].apply(lambda x: 'train' if x == 'C' else 'test')

# --- 2. Store number split ---
df_exp['set_store_number'] = df_exp['Store'].apply(lambda x: 'train' if x <= 30 else 'test')

# --- 3. Holiday split ---
df_exp['set_holiday'] = df_exp['IsHoliday'].apply(lambda x: 'train' if x == 0 else 'test')

# --- 4. Christmas split ---
df_exp['set_christmas'] = df_exp['Is_Christmas_Season'].apply(lambda x: 'train' if x == 0 else 'test')

# --- 5. Season split ---
df_exp['set_season'] = df_exp['Season'].apply(lambda x: 'test' if x == 3 else 'train')


# --- 6. City split ---
test_cities = ["New York", "Los Angeles", "Chicago"]

df_exp['set_city'] = df_exp['city'].apply(lambda x: 'test' if x in test_cities else 'train')

# --- 7. Weather split ---
df_exp['set_weather'] = df_exp['weather_condition'].apply(lambda x: 'test' if x == 'Clear' else 'train')




print("\n STORE TYPE SPLIT")
df_exp['set'] = df_exp['set_store_type']
plot_store_type_split(df_exp)

print("\n STORE NUMBER SPLIT")
df_exp['set'] = df_exp['set_store_number']
plot_store_number_split(df_exp)

print("\n HOLIDAY SPLIT")
df_exp['set'] = df_exp['set_holiday']
plot_holiday_split(df_exp)

print("\n CHRISTMAS SPLIT")
df_exp['set'] = df_exp['set_christmas']
plot_christmas_split(df_exp)

print("\n SEASON SPLIT")
df_exp['set'] = df_exp['set_season']
plot_season_split(df_exp)

print("\n CITY SPLIT")
df_exp['set'] = df_exp['set_city']
plot_city_split(df_exp)

print("\n WEATHER SPLIT")
df_exp['set'] = df_exp['set_weather']
plot_weather_split(df_exp)


print ("\n Sales over time ")
plot_sales_over_time(data)

print("\n Correlation Heatmap")
plot_correlation_heatmap(data)

print("\n Sales Distribution")
plot_distribution(data,"Weekly_Sales")
#%%
print(f"Dataset shape: {data.shape}")
#%%
print(data.columns)
#%%
unnamed_columns = [col for col in data.columns if 'Unnamed' in col]
if unnamed_columns:
    print(f"Removing useless columns: {unnamed_columns}")
    data = data.drop(columns=unnamed_columns)
#%%
data.to_csv("final_data_walmart.csv",index=False,
    encoding="utf-8",
    float_format="%.4f")

data.to_parquet("final_data_walmart.parquet", index=False)