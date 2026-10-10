import pandas as pd

stores = pd.read_csv('/data/stores.csv')
holidays = pd.read_csv('/data/holidays_events.csv')

cities = set(stores['city'].dropna().unique())
states = set(stores['state'].dropna().unique())

unique_locales = holidays[['locale', 'locale_name']].drop_duplicates()

results = []
for _, row in unique_locales.iterrows():
    loc_type = row['locale']
    loc_name = row['locale_name']

    match_status = "No Match"
    if loc_name in cities and loc_name in states:
        match_status = "City & State"
    elif loc_name in cities:
        match_status = "City"
    elif loc_name in states:
        match_status = "State"
    elif loc_type == "National" and loc_name == "Ecuador":
        match_status = "Country (Ecuador)"

    results.append({
        'Locale_Type': loc_type,
        'Locale_Name': loc_name,
        'Matched_In_Stores': match_status
    })

results_df = pd.DataFrame(results)

print("Summary")
summary = results_df.groupby(['Locale_Type', 'Matched_In_Stores']).size().reset_index(name='Count')
print(summary.to_string(index=False))

print("\n" + "=" * 40 + "\n")

print("Regional Holidays")
print(results_df[results_df['Locale_Type'] == 'Regional'].to_string(index=False))

print("\n" + "=" * 40 + "\n")

unmatched = results_df[(results_df['Locale_Type'] != 'National') & (results_df['Matched_In_Stores'] == 'No Match')]
print(f"Local/Regional that dont match to any store) - Total: {len(unmatched)}")
print(unmatched.to_string(index=False))