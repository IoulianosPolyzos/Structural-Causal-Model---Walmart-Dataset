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
import yaml
import sys
sys.path.append("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm")
from custom_mape import custom_retail_mape
with open("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]


CONFIG = {
    'train_cities': ["Houston", "Philadelphia", "Phoenix", "San Jose", "Jacksonville","Austin"],
    'test_cities': ["New York", "Los Angeles", "Chicago"],
    'random_seed': 42,

    'checkpoint_dir': './checkpoints_city',

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
    df = pd.read_csv("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv")
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    print(f"Unique cities: {df['city'].nunique()}")
    print(f"Cities: {sorted(df['city'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/VAL/TEST BY CITIES)")


train_cities = CONFIG['train_cities']
test_cities = CONFIG['test_cities']

train_df = df[df['city'].isin(train_cities)].copy().reset_index(drop=True)
test_df_full = df[df['city'].isin(test_cities)].copy().reset_index(drop=True)

# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)

adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train cities: {train_cities}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest cities: {test_cities}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify no overlap
assert set(train_cities).isdisjoint(set(test_cities)), "Train/Test cities overlap!"
print(" No overlap between train/val/test cities")
#%%
print("PART 3: PREPROCESSING")


try:

    # Initial confirmation that location-based feature (city) has been encoded.
    # This assumes prior transformation ensuring consistency across datasets.


    # Define categorical columns that should not be coerced into numeric types.
    # These variables represent discrete entities or labels and must be preserved as categorical.
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Attempt to convert all non-categorical columns to numeric format.
    # This enforces type.sh consistency and ensures compatibility with downstream statistical models.
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                # Log conversion failures without interrupting execution,
                # allowing inspection of problematic features.
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Define binary variables.
    # These are treated as categorical (string-encoded) to allow classification-based mechanisms
    # within the Structural Causal Model (SCM).
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]

    # Convert binary variables to string-based categorical representation.
    # This avoids unintended ordinal interpretations and aligns with probabilistic classifiers.
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Define multi-class categorical variables.
    # These variables encode nominal information with no inherent ordering.
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]

    # Ensure consistent string typing across all datasets.
    # This guarantees stable encoding and avoids mismatches during model fitting or inference.
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    # Aggregate all nodes that will be modeled using classification mechanisms in the SCM.
    # This distinction is critical for assigning appropriate structural equations.
    classifier_nodes = binary_nodes + categorical_nodes

    print(f" Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    # Global exception handling to ensure traceability of preprocessing failures.
    print(f"ERROR: Preprocessing failed: {e}")
    raise
#%%
print("PART 4: CAUSAL GRAPH ANALYSIS")


# Define groups of variables representing higher-level latent concepts.
# This grouping enables reasoning at an abstract causal level before expanding to feature granularity.
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

# Specify directed causal relations between groups.
# These edges encode domain-informed assumptions about the data-generating process.
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
    #("City", "Sales"),  # Removed to avoid imposing a direct effect; enforces mediated pathways.
    ("Store_Features", "Sales"),
    ("Store_Features", "MarkDowns"),
    ("Weather", "Sales"),
    ("Economy", "Sales")
]

# Construct a macro-level Directed Acyclic Graph (DAG).
# This representation facilitates interpretability and causal pathway tracing.
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

# Define layout for consistent and interpretable visualization.
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),"Store_Features":(1,2),
    "Economy": (2, 2), "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1), "Sales": (4, 0)
}

# Visualize the macro-level causal graph.
# The layout reflects assumed structural and temporal ordering.
plt.figure(figsize=(13, 8))
nx.draw(macro_graph, pos, with_labels=True, node_color='lightblue',
        node_size=3500, edge_color='gray', font_size=11,
        font_weight='bold', arrows=True, arrowsize=20)
plt.title("Causal Graph — Graph Final", fontsize=16)
plt.tight_layout()
plt.show()

def get_all_descendants(graph, source_node):
    """
    Compute the transitive closure of a node in a directed graph.

    In causal terms, this corresponds to identifying all downstream variables
    that may be affected by an intervention do(source_node = x).
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


# Analyze the downstream impact of "City".
# This effectively estimates the intervention scope do(City = c).
affected_groups = get_all_descendants(macro_graph, "City")

# Map affected groups to their corresponding feature-level variables.
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

# Report direct and indirect causal effects.
print(f"Direct children of City: {set(macro_graph.successors('City'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")

# Provide an explicit causal interpretation of MarkDowns dependency.
# This highlights a mediated pathway rather than a direct causal link.
print("\n YES, MarkDowns are refitted because:")
print("  City → Store → MarkDowns (transitive effect)")
print("  Store changes with city, so MarkDowns distribution changes too")



print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


# Initialize the feature-level DAG.
# Each node corresponds to an observed variable in the dataset.
feature_graph = nx.DiGraph()

# Add all individual features as nodes.
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

# Expand group-level edges into feature-level dependencies.
# This induces a dense mapping between variables across groups.
for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")



print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")


# Instantiate a Structural Causal Model (SCM) over the feature graph.
# Each node will be associated with a structural equation.
scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assign structural mechanisms to each node.

    - Root nodes (no parents) are modeled via empirical distributions.
    - Categorical variables are modeled using probabilistic classifiers.
    - Continuous variables are modeled using additive noise models (ANMs).

    This hybrid modeling strategy enables flexible approximation of complex,
    nonlinear causal relationships.
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        # Exogenous variables: modeled non-parametrically
        if len(parents) == 0:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())

        # Classification-based mechanism for discrete variables
        elif node in classifier_nodes:
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(
                        LGBMClassifier(**lgb_params)
                    )
                )
            )

        # Regression-based mechanism for continuous variables
        else:
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    SklearnRegressionModel(
                        LGBMRegressor(**lgb_params)
                    )
                )
            )


# Fit the SCM using observational training data.
# This step estimates all structural equations under the assumed causal graph.
try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print(" Mechanisms setup complete")

    print("Fitting SCM on training data...")
    gcm.fit(scm, train_df)
    print(" Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Save a checkpoint for reproducibility and downstream causal analysis.
# This enables later interventions, counterfactual queries, and auditing.
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'initial_fit_complete', 'train_set_size': len(train_df)},
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")
#%%



print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

ordered_nodes = list(nx.topological_sort(feature_graph))

nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]


nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to new test cities (learning new sales behavior)...")

try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Concept & Covariate shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'domain_adaptation_complete', 'nodes_refitted': refit_count},
        stage_name="domain_adapted"
    )
except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise
#%%


print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")


try:
    # Sample from the HOLDOUT TEST DATA which the model has NEVER seen
    test_sample = holdout_test_df.reset_index(drop=True)

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")

    # Get causal mechanism and parents
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Generate predictions
    X_test = test_sample[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    # Calculate metrics
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
    """Evaluate Weekly_Sales mechanism on the exact same holdout set."""
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))
    sales_mechanism = model_scm.causal_mechanism("Weekly_Sales")

    X_eval = eval_df[parents].to_numpy()
    y_true = eval_df["Weekly_Sales"].to_numpy()
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Mape": custom_retail_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    print(f"\n{label}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}")
    print(f"R²:   {metrics['R2']:.4f}")

    return metrics


try:
    # 1. Load the PRE-ADAPTATION SCM from checkpoint
    baseline_scm, baseline_metadata = load_checkpoint(scm_path_initial, metadata_path_initial)

    # 2. Use the EXACT SAME HOLDOUT SET for both models
    comparison_df = holdout_test_df.reset_index(drop=True)

    # 3. Evaluate baseline and adapted SCM on the same rows
    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        "BASELINE SCM (before adaptation)"
    )

    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        "ADAPTED SCM (after adaptation)"
    )

    # 4. Fair side-by-side comparison
    print("\n FAIR HOLDOUT COMPARISON ")
    print(f"{'Metric':<10} {'Baseline':<12} {'Adapted':<12} {'Delta':<12}")
    print("-" * 52)

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

    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and \
       adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]:
        print("\n Adaptation improved performance on the same holdout set.")
    else:
        print("\n WARNING: Adaptation did not clearly improve performance on the same holdout set.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise