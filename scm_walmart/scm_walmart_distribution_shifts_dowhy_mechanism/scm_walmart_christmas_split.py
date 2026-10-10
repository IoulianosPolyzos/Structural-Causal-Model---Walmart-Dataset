#%%
#!/usr/bin/env python3

import os
import warnings
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm
from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
from matplotlib import pyplot as plt
import yaml



os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
print("PART 1: LOADING DATA & SETUP")
with open("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]
RANDOM_SEED = 42

try:
    df = pd.read_csv("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv")
    df = df.drop(columns=["Date", "Season", "DayOfWeek", "Month", "WeekOfYear"], errors='ignore')
    df = df.fillna(0)
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("ERROR: Data file not found!")
    raise

# ==========================================
# 2. DOMAIN SPLIT (Normal vs Christmas)
# ==========================================
print("\nPART 2: DOMAIN SHIFT SPLIT")
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

# OLD DOMAIN: Normal Days
old_data = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)
# NEW DOMAIN: Christmas Days
new_data = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)

print(f"Old Domain (Normal Days) shape: {old_data.shape}")
print(f"New Domain (Christmas Days) shape: {new_data.shape}")

# ==========================================
# 3. PREPROCESSING
# ==========================================
print("\nPART 3: PREPROCESSING")
categorical_cols_to_exclude = ["city", "Type", "weather_condition", "Store", "Dept"]
binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start", "Is_Month_End"]
categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
classifier_nodes = binary_nodes + categorical_nodes

for col in df.columns:
    if col not in categorical_cols_to_exclude:
        old_data[col] = pd.to_numeric(old_data[col], errors='coerce').fillna(0)
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce').fillna(0)

for col in binary_nodes + categorical_nodes:
    if col in old_data.columns:
        old_data[col] = old_data[col].astype(str)
        new_data[col] = new_data[col].astype(str)

print("Preprocessing complete.")

# ==========================================
# 4. CAUSAL GRAPH (DAG) DEFINITION
# ==========================================
print("\nPART 4: BUILDING CAUSAL GRAPH")
causal_groups = {
    "Holidays": ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "holiday_proximity", "holiday_weight"],
    "Date_Features": ["Year", "Quarter", "Year_Progress", "Month_Progress", "Is_Month_Start", "Is_Month_End", "Is_Summer", "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos", "WeekOfYear_sin", "WeekOfYear_cos"],
    "Season": ["Season_sin", "Season_cos"],
    "City": ["city"],
    "Economy": ["Fuel_Price", "CPI", "Unemployment"],
    "Weather": ["weather_condition", "Temperature", "precipitation", "wind_speed", "humidity", "temperature_min", "temperature_max"],
    "Store_Features": ["Store", "Type", "Size", "Dept"],
    "MarkDowns": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],
    "Sales": ["Weekly_Sales"]
}

causal_edges_groups = [
    ("Date_Features", "Season"), ("Date_Features", "Holidays"), ("Date_Features", "Sales"), ("Date_Features", "Economy"),
    ("Season", "Weather"), ("Season", "Sales"),
    ("Holidays", "MarkDowns"), ("Holidays", "Sales"),
    ("MarkDowns", "Sales"),
    ("City", "Weather"), ("City", "Store_Features"), ("City", "Economy"), ("City", "Sales"),
    ("Store_Features", "Sales"), ("Store_Features", "MarkDowns"),
    ("Weather", "Sales"), ("Economy", "Sales")
]

# Build Feature-Level Graph
feature_graph = nx.DiGraph()
for group, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for src_grp, tgt_grp in causal_edges_groups:
    for src_col in causal_groups[src_grp]:
        for tgt_col in causal_groups[tgt_grp]:
            feature_graph.add_edge(src_col, tgt_col)

# Remove restricted edge
if feature_graph.has_edge('Is_Christmas_Season', 'Weekly_Sales'):
    feature_graph.remove_edge('Is_Christmas_Season', 'Weekly_Sales')

print(f"Feature Graph constructed: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# ==========================================
# 5. ASSIGN CAUSAL MECHANISMS (SCM)
# ==========================================
print("\nPART 5: SCM SETUP")
scm = gcm.StructuralCausalModel(feature_graph)

for node in feature_graph.nodes:
    parents = list(feature_graph.predecessors(node))
    if len(parents) == 0:
        scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
    elif node in classifier_nodes:
        scm.set_causal_mechanism(node, gcm.ClassifierFCM(SklearnClassificationModel(LGBMClassifier(**lgb_params))))
    else:
        scm.set_causal_mechanism(node, gcm.AdditiveNoiseModel(SklearnRegressionModel(LGBMRegressor(**lgb_params))))

print("Mechanisms assigned.")

# ==========================================
# 6. DISTRIBUTIONAL CHANGE ATTRIBUTION
# ==========================================
print("\nPART 6: ATTRIBUTING DISTRIBUTIONAL CHANGES (ROOT CAUSE ANALYSIS)")

SAMPLE_SIZE = 1500
old_data_sample = old_data.sample(n=min(SAMPLE_SIZE, len(old_data)), random_state=RANDOM_SEED)
new_data_sample = new_data.sample(n=min(SAMPLE_SIZE, len(new_data)), random_state=RANDOM_SEED)

print(f"Comparing Normal Days (old) vs Christmas Days (new) using {len(old_data_sample)} vs {len(new_data_sample)} samples...")
print("Please wait, computing Shapley values (this may take a few minutes)...")

attributions = gcm.distribution_change(
    scm,
    old_data_sample,
    new_data_sample,
    'Weekly_Sales'
)

print("\n--- Attribution Scores (Which features caused the shift in Weekly_Sales?) ---")
sorted_attributions = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)

for node, score in sorted_attributions:
    if abs(score) > 0.001:
        print(f"  {node:<25}: {score:.5f}")

top_nodes = [x[0] for x in sorted_attributions[:10]]
top_scores = [x[1] for x in sorted_attributions[:10]]

split_name = "Normal_vs_Christmas"

plt.figure(figsize=(10, 6))
plt.barh(top_nodes[::-1], top_scores[::-1], color='coral')
plt.title(f'Top 10 Drivers of Distribution Shift in Weekly_Sales\n({split_name.replace("_", " ")})')
plt.xlabel('Attribution Score (Shapley Value)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

filename = f'distribution_shift_{split_name}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nSaved us: {filename}")

plt.show()