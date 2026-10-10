import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import learning_curve,TimeSeriesSplit



def plot_sales_over_time(df, date_col='date', target='unit_sales'):
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    ts = temp.groupby(date_col)[target].sum()

    plt.figure(figsize=(12,5))
    plt.plot(ts)
    plt.title("Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()

def plot_monthly_seasonality(df, month_col='Month', target='unit_sales'):
    monthly = df.groupby(month_col)[target].mean()

    plt.figure(figsize=(10,5))
    monthly.plot(kind='bar')
    plt.title("Monthly Seasonality")
    plt.xlabel("Month")
    plt.ylabel("Avg Sales")
    plt.show()

def plot_weekly_seasonality(df, week_col='WeekOfYear', target='unit_sales'):
    weekly = df.groupby(week_col)[target].mean()

    plt.figure(figsize=(12,5))
    weekly.plot()
    plt.title("Weekly Seasonality")
    plt.xlabel("Week of Year")
    plt.ylabel("Avg Sales")
    plt.show()


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10,6))
    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_top_categories(df, col, target='unit_sales', top_n=10):
    temp = df.groupby(col)[target].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10,5))
    temp.plot(kind='bar')
    plt.title(f"Top {top_n} {col} by Sales")
    plt.ylabel("Total Sales")
    plt.show()

def plot_distribution(df, col):
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()




import matplotlib.pyplot as plt
import pandas as pd

def plot_split_bar(df, condition_col, target='unit_sales',
                   train_name="Train", test_name="Test",
                   title="Split Analysis"):
    """
    Generic bar plot for dataset split analysis.
    Shows mean sales per group for train vs test.
    """

    train_df = df[df['set'] == 'train']
    test_df = df[df['set'] == 'test']

    train_vals = train_df.groupby(condition_col)[target].mean()
    test_vals = test_df.groupby(condition_col)[target].mean()

    all_idx = sorted(set(train_vals.index).union(set(test_vals.index)))

    train_vals = train_vals.reindex(all_idx).fillna(0)
    test_vals = test_vals.reindex(all_idx).fillna(0)

    x = range(len(all_idx))

    plt.figure(figsize=(12,6))
    plt.bar([i - 0.2 for i in x], train_vals, width=0.4, label=train_name)
    plt.bar([i + 0.2 for i in x], test_vals, width=0.4, label=test_name)

    plt.xticks(x, all_idx, rotation=45)
    plt.title(title)
    plt.ylabel("Mean Unit Sales")
    plt.legend()
    plt.show()

def plot_store_type_split(df):
    plot_split_bar(
        df,
        condition_col="store_type",
        title="Store Type Split (C train vs A/B test)"
    )
def plot_store_number_split(df):
    df = df.copy()
    df['store_group'] = df['store_nbr'].apply(lambda x: "1-30" if x <= 30 else "31-54")

    plot_split_bar(
        df,
        condition_col="store_group",
        title="Store Number Split (1-30 vs 31-54)"
    )
def plot_holiday_split(df):
    plot_split_bar(
        df,
        condition_col="IsHoliday",
        title="Normal Days vs Holidays"
    )
def plot_christmas_split(df):
    plot_split_bar(
        df,
        condition_col="Is_Christmas_Season",
        title="Normal Days vs Christmas Days"
    )
def plot_season_split(df):
    plot_split_bar(
        df,
        condition_col="Season",
        title="Seasonal Split (Winter/Spring/Autumn vs Summer)"
    )

def plot_city_split(df):
    plot_split_bar(
        df,
        condition_col="city",
        title="City Generalization Split"
    )

def plot_weather_split(df):
    plot_split_bar(
        df,
        condition_col="weather_condition",
        title="Weather Domain Shift (Clouds/Rain/Snow vs Clear)"
    )