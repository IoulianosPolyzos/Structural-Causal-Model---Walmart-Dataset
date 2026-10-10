import pandas as pd
import numpy as np
import os
import kagglehub
import py7zr


def create_master_dataset(nrows_train=None):
    """
    Downloads data, extracts it, loads it into pandas DataFrames,
    cleans, merges, and adds custom features (Holidays, Earthquake).
    Returns the final master DataFrame.
    """
    print("0. Downloading data directly from Kaggle...")
    data_dir = kagglehub.competition_download('favorita-grocery-sales-forecasting')
    print(f"Data is located at: {data_dir}")

    required_files = [
        'train.csv', 'stores.csv', 'items.csv',
        'transactions.csv', 'oil.csv', 'holidays_events.csv'
    ]

    print("1. Checking and extracting .7z archives...")
    for file in required_files:
        csv_path = os.path.join(data_dir, file)
        archive_path = csv_path + '.7z'

        # If CSV does not exist but .7z does, extract it
        if not os.path.exists(csv_path) and os.path.exists(archive_path):
            print(f"  -> Extracting: {file}.7z")
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(path=data_dir)

    print("2. Loading datasets into memory...")
    # 'nrows_train' is used to limit rows for quick testing. Set to None to load the full dataset.
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'), parse_dates=['date'], nrows=nrows_train)
    stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
    items = pd.read_csv(os.path.join(data_dir, 'items.csv'))
    transactions = pd.read_csv(os.path.join(data_dir, 'transactions.csv'), parse_dates=['date'])
    oil = pd.read_csv(os.path.join(data_dir, 'oil.csv'), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(data_dir, 'holidays_events.csv'), parse_dates=['date'])

    print("3. Cleaning and preparing individual datasets...")
    # Rename columns to avoid conflicts during merges
    stores.rename(columns={'type': 'store_type'}, inplace=True)
    holidays.rename(columns={'type': 'holiday_type'}, inplace=True)

    # Fill missing values in oil prices (forward fill for weekends, backward fill for safety)
    oil = oil.sort_values('date')
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()

    # Remove transferred holidays (they were celebrated on a different day)
    holidays_clean = holidays[holidays['transferred'] == False].copy()

    print("4. Performing base merges...")
    # Left joins on the main train dataset
    df = train.merge(stores, on='store_nbr', how='left')
    df = df.merge(items, on='item_nbr', how='left')
    df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
    df = df.merge(oil, on='date', how='left')

    print("5. Creating Geographical IsHoliday feature...")
    # Create a smaller DataFrame with unique dates and locations to save RAM
    store_dates = df[['date', 'store_nbr', 'city', 'state']].drop_duplicates()
    store_holidays = store_dates.merge(holidays_clean, on='date', how='left')

    # Define the geographical conditions for a valid holiday
    geo_condition = (
            (store_holidays['locale'] == 'National') |
            ((store_holidays['locale'] == 'Regional') & (store_holidays['locale_name'] == store_holidays['state'])) |
            ((store_holidays['locale'] == 'Local') & (store_holidays['locale_name'] == store_holidays['city']))
    )
    store_holidays['IsHoliday'] = np.where(geo_condition, 1, 0)

    # Aggregate back to (date, store_nbr) keeping the maximum value (1 if it's a holiday)
    holiday_flags = store_holidays.groupby(['date', 'store_nbr'])['IsHoliday'].max().reset_index()

    # Merge the IsHoliday flag back to the main DataFrame
    df = df.merge(holiday_flags, on=['date', 'store_nbr'], how='left')
    df['IsHoliday'] = df['IsHoliday'].fillna(0).astype(int)

    print("6. Adding Earthquake impact feature...")
    # Earthquake occurred on April 16, 2016. We observe a 1-month window of relief efforts
    is_earthquake_period = (df['date'] >= '2016-04-16') & (df['date'] <= '2016-05-16')

    # Target specific families of products that saw increased sales due to donations
    affected_families = ['BEVERAGES', 'GROCERY I', 'PERSONAL CARE', 'CLEANING']
    is_affected_family = df['family'].isin(affected_families)

    df['earthquake_impact'] = np.where(is_earthquake_period & is_affected_family, 1, 0)

    print("Processing completed successfully!")
    return df
