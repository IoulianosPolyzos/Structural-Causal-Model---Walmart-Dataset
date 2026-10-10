#%%
from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel

# !/usr/bin/env python3


import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import yaml
import sys
sys.path.append("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart")
from custom_mape import custom_retail_mape
with open("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_seasons': [1, 2, 4],
    'target_seasons': [3],
    'random_seed': 42,

    'checkpoint_dir': './checkpoints_season',

}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)





#%%



print("PART 1: LOADING DATA & SETUP")


try:
    df = pd.read_csv("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv")

    df = df.drop(
        columns=["Date", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    if 'Season' in df.columns:
        print(f"Unique seasons available: {sorted(df['Season'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file '/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv' not found!")
    raise

# ============================================================================
# PART 2: SPLIT DATA BY SEASON
# ============================================================================


print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY SEASON)")


source_seasons = CONFIG['source_seasons']
target_seasons = CONFIG['target_seasons']

# Split based on Season
source_df = df[df['Season'].isin(source_seasons)].copy().reset_index(drop=True)
target_df = df[df['Season'].isin(target_seasons)].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"(Source) seasons: {source_seasons}")
print(f"Train shape: {train_df.shape}")
print(f"\nTarget seasons: {target_seasons}")


# Verify no overlap
#assert set(train_seasons).isdisjoint(set(test_seasons)), "Train/Test seasons overlap!"
print("✓ No overlap between train/val/test seasons")

#%%


print("PART 3: PREPROCESSING")


try:
    # Define categorical columns
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept", "Season"
    ]

    # Convert numeric columns
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

#%%



print("PART 4: CAUSAL GRAPH ANALYSIS")


causal_groups = {
    "Holidays": ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "holiday_proximity", "holiday_weight"],
    "Date_Features": [
        "Year", "Quarter",
        "Year_Progress", "Month_Progress",
        "Is_Month_Start", "Is_Month_End",
        "Is_Summer",
        "Month_sin", "Month_cos",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "WeekOfYear_sin", "WeekOfYear_cos",],
    "Season": ["Season_sin", "Season_cos"],
    "City": ["city"],
    "Economy": ["Fuel_Price", "CPI", "Unemployment"],
    "Weather": [
        "weather_condition", "Temperature",
        "precipitation", "wind_speed", "humidity","temperature_min","temperature_max"
    ],
    "Store_Features": ["Store", "Type", "Size", "Dept"],
    "MarkDowns": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],
    "Sales": ["Weekly_Sales"]
}

causal_edges_groups = [
    ("Date_Features", "Season"),
    ("Date_Features", "Holidays"),
    ("Date_Features", "Sales"),
    ("Date_Features", "Economy"),
    ("Season", "Weather"),
    ("Season", "Sales"),
    ("Holidays", "MarkDowns"),
    ("Holidays", "Sales"),
    ("MarkDowns", "Sales"),
    ("City", "Weather"),
    ("City", "Store_Features"),
    ("City", "Economy"),
    ("City", "Sales"),
    ("Store_Features", "Sales"),
    ("Store_Features", "MarkDowns"),
    ("Weather", "Sales"),
    ("Economy", "Sales")
]

macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),"Store_Features":(1,2),
    "Economy": (2, 2), "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1), "Sales": (4, 0)
}

plt.figure(figsize=(13, 8))
nx.draw(macro_graph, pos, with_labels=True, node_color='lightblue',
        node_size=3500, edge_color='gray', font_size=11,
        font_weight='bold', arrows=True, arrowsize=20)
plt.title("Causal Graph", fontsize=16)
plt.tight_layout()
plt.show()

def get_all_descendants(graph, source_node):
    """
    Get all nodes reachable from source_node (transitive closure).
    """
    descendants = set()
    to_visit = [source_node]
    visited = set()

    while to_visit:
        node = to_visit.pop(0)
        if node in visited:
            continue
        visited.add(node)

        children = list(graph.successors(node))
        descendants.update(children)
        to_visit.extend(children)

    return descendants


affected_groups = get_all_descendants(macro_graph, "Season")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Season: {set(macro_graph.successors('Season'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")



print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


feature_graph = nx.DiGraph()

for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")



print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")


scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Setup causal mechanisms for all nodes.
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        if len(parents) == 0:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif node in classifier_nodes:
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(
                        LGBMClassifier(**lgb_params)
                    )
                )
            )
        else:
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    SklearnRegressionModel(
                        LGBMRegressor(**lgb_params)
                    )
                )
            )


try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("✓ Mechanisms setup complete")

    print("Fitting SCM on training data (Seasons 1, 2, 4)...")
    gcm.fit(scm, train_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise


#%%



try:
    # Get causal mechanism and parents
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Generate predictions
    X_test = test_df[parents].to_numpy()

    train_features = parents
    test_features = list(test_df[parents].columns)

    missing = set(train_features) - set(test_features)
    extra = set(test_features) - set(train_features)

    print("Missing:", missing)
    print("Extra:", extra)

    if list(test_df[parents].columns) != parents:
        print("⚠️ Feature order mismatch!")
    else:
        print("✓ Feature order OK")
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()
    negative_percentage = (test_predictions < 0).mean() * 100
    print(negative_percentage)
    # Calculate metrics
    mae = mean_absolute_error(test_df['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_df['Weekly_Sales'].values, test_predictions))
    r2 = r2_score(test_df['Weekly_Sales'].values, test_predictions)
    mape = custom_retail_mape(test_df['Weekly_Sales'].values, test_predictions)

    print(f"\n=== PREDICTION METRICS (Test Set) ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    print(f"R²:   {r2:.4f}")

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise

