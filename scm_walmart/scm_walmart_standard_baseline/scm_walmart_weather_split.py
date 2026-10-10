#%%
from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel

#!/usr/bin/env python3


import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from lightgbm import LGBMRegressor, LGBMClassifier

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
    'train_weather': ["Rain","Snow","Clouds"],
    'test_weather': ["Clear"],
    'random_seed': 42,

    'checkpoint_dir': './checkpoints_weather',

}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)




def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Save SCM model and metadata to disk.
    """
    try:
        checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save SCM
        scm_path = checkpoint_dir / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "stage": stage_name,
            "parents": parents,
            "metrics": metrics,
            "feature_count": len(parents),
            "nodes": list(feature_graph.nodes()),
            "edges": list(feature_graph.edges())
        }
        metadata_path = checkpoint_dir / f"metadata_{stage_name}_{timestamp}.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f" Checkpoint saved: {scm_path.name}")
        return str(scm_path), str(metadata_path)
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise


def load_checkpoint(scm_path, metadata_path):
    """
    Load SCM model and metadata from disk.
    """
    try:
        with open(scm_path, "rb") as f:
            scm = pickle.load(f)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        print(f" Checkpoint loaded from {Path(scm_path).name}")
        return scm, metadata
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise

#%%
print("PART 1: LOADING DATA & SETUP")


try:
    df = pd.read_csv("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv")
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    print(f"Unique cities: {df['weather_condition'].nunique()}")
    print(f"Cities: {sorted(df['weather_condition'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY CITIES



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY weather_conditions)")


train_weather = CONFIG['train_weather']
test_weather = CONFIG['test_weather']

train_df = df[df['weather_condition'].isin(train_weather)].copy().reset_index(drop=True)
test_df_full = df[df['weather_condition'].isin(test_weather)].copy().reset_index(drop=True)


adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train weather conditions: {train_weather}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest weather conditions: {test_weather}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify no overlap
assert set(train_weather).isdisjoint(set(test_weather)), "Train/Test weather conditions overlap!"
print(" No overlap between train/val/test weather_conditions")
#%%
print("PART 3: PREPROCESSING")


try:



    # Define categorical columns
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Convert numeric columns
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f" Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise
#%%
print("PART 4: CAUSAL GRAPH ANALYSIS")

# Definition of high-level causal groups representing aggregated feature domains.
# Each group corresponds to a semantically coherent set of observed variables.

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

# Definition of macro-level causal relationships between feature groups.
# These edges represent assumed structural dependencies in the data-generating process.
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
    #("Weather", "Sales"),
    ("Economy", "Sales")
]

# Construction of macro-level causal DAG.
# Nodes represent feature groups; edges represent inter-group causal influence.
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

# Fixed layout for visualization consistency across experiments.
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),"Store_Features":(1,2),
    "Economy": (2, 2), "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1), "Sales": (4, 0)
}

# Visualization of macro causal structure.
plt.figure(figsize=(13, 8))
nx.draw(
    macro_graph, pos,
    with_labels=True,
    node_color='lightblue',
    node_size=3500,
    edge_color='gray',
    font_size=11,
    font_weight='bold',
    arrows=True,
    arrowsize=20
)
plt.title("Causal Graph — Graph Final", fontsize=16)
plt.tight_layout()
plt.show()

def get_all_descendants(graph, source_node):
    """
    Computes the transitive closure of descendants for a given source node.
    This represents all nodes affected by an intervention on the source variable.
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


# Identification of all nodes affected by an intervention on Weather.
# This simulates a domain shift propagated through the causal graph.
affected_groups = get_all_descendants(macro_graph, "Weather")

affected_feature_nodes = []

# Expansion from group-level affected nodes to feature-level variables.
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Weather: {set(macro_graph.successors('Weather'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")

print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")

# Construction of feature-level DAG by expanding group-level edges.
# Each group edge is converted into full bipartite connections between features.
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

# Initialization of Structural Causal Model (SCM) over the feature-level DAG.
scm = gcm.StructuralCausalModel(feature_graph)

def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assigns causal mechanisms to each node in the SCM.
    - Root nodes: Empirical distribution
    - Classification nodes: probabilistic classifier models
    - Continuous nodes: additive noise regression models
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
    print(" Mechanisms setup complete")

    # Fitting SCM by learning all conditional mechanisms from training data.
    print("Fitting SCM on training data...")
    gcm.fit(scm, train_df)

    print(" Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Persistence of trained SCM for reproducibility and later counterfactual evaluation.
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,

        # Parent set of target variable defines inference interface.
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),

        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df)
        },
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")
#%%
print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Topological ordering ensures that causal dependencies are respected
# (i.e., parent nodes are processed before their children in the DAG)
ordered_nodes = list(nx.topological_sort(feature_graph))

# Select only nodes that are causally affected by the shifted upstream factor
# (e.g., Weather / Store / Season depending on the experiment setup)
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to new test weather conditions (learning new sales behavior)...")

try:
    refit_count = 0

    # Sequentially re-estimate structural mechanisms for affected variables
    # using adaptation data only, ensuring no information leakage from holdout/test sets
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")

            # Local mechanism re-training under shifted domain distribution
            # This corresponds to updating P(X_i | Pa(X_i)) under covariate/concept shift
            fit_causal_model_of_target(scm, node, adaptation_df)

            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    # Summary of successful structural updates
    print(f"\n Concept & Covariate shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    # Persist adapted SCM state for reproducibility and evaluation
    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,
        },
        stage_name="domain_adapted"
    )

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise
#%%
print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")

try:
    # Evaluation is performed on a strictly held-out test set that has not been
    # used during training or adaptation phases, ensuring an unbiased estimate
    # of out-of-distribution generalization performance.
    test_sample = holdout_test_df.reset_index(drop=True)

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")

    # Extract the learned structural mechanism for the target variable
    # In Structural Causal Models, this corresponds to P(Y | Pa(Y))
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")

    # Identify causal parents of the target node in the learned DAG
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Construct test design matrix using only causal parents (no leakage)
    X_test = test_sample[parents].to_numpy()

    # Generate predictions using the learned structural equation model
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    # Standard regression evaluation metrics on holdout distribution
    mae = mean_absolute_error(test_sample['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_sample['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_sample['Weekly_Sales'].values,test_predictions)
    r2 = r2_score(test_sample['Weekly_Sales'].values, test_predictions)

    print(f"\n PREDICTION METRICS (Holdout Set - After Adaptation) ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    print(f"R²:   {r2:.4f}")

    metrics_after_adaptation = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise
#%%
print("PART 9: FAIR COMPARISON - BASELINE vs ADAPTED ON SAME HOLDOUT SET")

def evaluate_sales_on_holdout(model_scm, eval_df, label):
    """
    Evaluation of the structural causal model's target mechanism
    under identical holdout conditions for unbiased comparison.

    The function evaluates P(Weekly_Sales | Pa(Weekly_Sales)) learned
    under different training regimes (baseline vs adapted SCM).
    """

    # Extract causal parents of the target variable from the learned DAG
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Retrieve learned structural mechanism for the target node
    sales_mechanism = model_scm.causal_mechanism("Weekly_Sales")

    # Construct evaluation design matrix using only causal parents
    X_eval = eval_df[parents].to_numpy()

    # Ground-truth outcomes from the holdout distribution
    y_true = eval_df["Weekly_Sales"].to_numpy()

    # Model predictions from learned structural equation
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()

    # Standard regression metrics for predictive accuracy comparison
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": custom_retail_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    print(f"\n{label}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}")
    print(f"R²:   {metrics['R2']:.4f}")


    return metrics


try:
    # Load baseline structural causal model (pre-adaptation state)
    # This represents the source domain distribution
    baseline_scm, baseline_metadata = load_checkpoint(
        scm_path_initial,
        metadata_path_initial
    )

    # Ensure identical evaluation distribution for fair comparison
    comparison_df = holdout_test_df.reset_index(drop=True)

    # Evaluate baseline SCM on holdout target domain
    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        "BASELINE SCM (before adaptation)"
    )

    # Evaluate adapted SCM on the same holdout distribution
    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        "ADAPTED SCM (after adaptation)"
    )

    # ---------------- FAIR COMPARISON TABLE ----------------
    print("\n FAIR HOLDOUT COMPARISON ")
    print(f"{'Metric':<10} {'Baseline':<12} {'Adapted':<12} {'Delta':<12}")
    print("-" * 52)

    # Compute performance deltas (improvement = positive MAE/RMSE reduction, R² increase)
    mae_delta = baseline_metrics_holdout["MAE"] - adapted_metrics_holdout["MAE"]
    rmse_delta = baseline_metrics_holdout["RMSE"] - adapted_metrics_holdout["RMSE"]
    mape_delta = baseline_metrics_holdout["MAPE"] - adapted_metrics_holdout["MAPE"]
    r2_delta = adapted_metrics_holdout["R2"] - baseline_metrics_holdout["R2"]

    print(
        f"{'MAE':<10} "
        f"{baseline_metrics_holdout['MAE']:<12.4f} "
        f"{adapted_metrics_holdout['MAE']:<12.4f} "
        f"{mae_delta:+.4f}"
    )

    print(
        f"{'MAPE':<10} "
        f"{baseline_metrics_holdout['MAPE']:<15.4f} "
        f"{adapted_metrics_holdout['MAPE']:<15.4f} "
        f"{mape_delta:+.4f}"
    )

    print(
        f"{'RMSE':<10} "
        f"{baseline_metrics_holdout['RMSE']:<12.4f} "
        f"{adapted_metrics_holdout['RMSE']:<12.4f} "
        f"{rmse_delta:+.4f}"
    )
    print(
        f"{'R2':<10} "
        f"{baseline_metrics_holdout['R2']:<12.4f} "
        f"{adapted_metrics_holdout['R2']:<12.4f} "
        f"{r2_delta:+.4f}"
    )

    # Decision criterion: consistent improvement in both error and explained variance
    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and \
       adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]:
        print("\n Adaptation improved performance on the same holdout set.")
    else:
        print("\n WARNING: Adaptation did not clearly improve performance on the same holdout set.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise