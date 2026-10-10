#%%
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
from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel
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
    'train_domain': 'Normal Days (IsChristmasSeasons=0)',
    'test_domain': 'Christmas Days (Is_Christmas_Season=1)',
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_christmas',

}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)




def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Save SCM model and metadata to disk.

    Args:
        scm: Fitted StructuralCausalModel
        feature_graph: networkx DiGraph of causal structure
        parents (list): Parent nodes of Weekly_Sales
        metrics (dict): Performance metrics
        stage_name (str): Name of checkpoint stage
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

    Args:
        scm_path (str): Path to SCM pickle file
        metadata_path (str): Path to metadata pickle file

    Returns:
        tuple: (scm_walmart, metadata)
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
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Is_Christmas_Season distribution:")
    print(f"  Normal Days (0): {(df['Is_Christmas_Season'] == 0).sum()} records")
    print(f"  Christmas Days (1): {(df['Is_Christmas_Season'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY Christmas DOMAIN



print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs Christmas)")


# Ensure Is_Christmas_Season is numeric (0 or 1)
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

train_df = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)
test_df_full = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)



adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain: {CONFIG['train_domain']}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest domain: {CONFIG['test_domain']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify domain separation
assert (train_df['Is_Christmas_Season'] == 0).all(), "Train set contains non-zero Is_Christmas_Season!"
assert (adaptation_df['Is_Christmas_Season'] == 1).all(), "Adaptation set contains non-Christmas days!"
assert (holdout_test_df['Is_Christmas_Season'] == 1).all(), "Holdout set contains non-Christmas days!"
print(" Domain separation verified: No overlap between Normal/Christmas days")

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
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
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

# Define semantically coherent groups of variables, representing higher-level causal concepts.
# This abstraction allows reasoning at both macro (group) and micro (feature) levels.
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

# Define directed causal relationships between variable groups.
# These edges encode domain knowledge assumptions regarding data-generating mechanisms.
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

# Construct a macro-level Directed Acyclic Graph (DAG) over feature groups.
# This representation supports interpretability and causal pathway analysis.
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

# Define a fixed layout to enhance visual interpretability of causal hierarchy.
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),"Store_Features":(1,2),
    "Economy": (2, 2), "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1), "Sales": (4, 0)
}

# Visualize the macro-level causal structure.
# The layout reflects assumed temporal and structural ordering of variables.
plt.figure(figsize=(13, 8))
nx.draw(macro_graph, pos, with_labels=True, node_color='lightblue',
        node_size=3500, edge_color='gray', font_size=11,
        font_weight='bold', arrows=True, arrowsize=20)
plt.title("Causal Graph — Graph Final", fontsize=16)
plt.tight_layout()
plt.show()

def get_all_descendants(graph, source_node):
    """
    Compute the transitive closure of a given node in a directed graph.

    From a causal inference perspective, this corresponds to identifying
    all downstream variables potentially affected by interventions on the source node.
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


# Identify all variables causally affected by "Holidays".
# This corresponds to estimating the intervention scope do(Holidays = x).
affected_groups = get_all_descendants(macro_graph, "Holidays")

# Map affected groups to individual feature-level variables.
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

# Report direct and indirect causal influence.
print(f"Direct children of Holidays: {set(macro_graph.successors('Holidays'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")





print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


# Initialize a fine-grained causal graph at the feature level.
# Each node now corresponds to an individual observed variable.
feature_graph = nx.DiGraph()

# Add all feature nodes explicitly.
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

# Expand group-level causal assumptions into feature-level dependencies.
# This results in a fully connected bipartite mapping between groups.
for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Remove a specific edge to prevent an undesired direct causal shortcut,
# ensuring that domain constraints (e.g., mediated effects) are respected.
feature_graph.remove_edge('Is_Christmas_Season','Weekly_Sales')

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")



print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")


# Initialize a Structural Causal Model (SCM) over the feature-level graph.
# Each node will be associated with a structural equation.
scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assign causal mechanisms to each node in the SCM.

    - Root nodes are modeled via empirical distributions.
    - Discrete variables are modeled using probabilistic classifiers.
    - Continuous variables are modeled via additive noise models (ANMs).

    This hybrid specification allows flexible approximation of complex,
    non-linear causal relationships.
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        # Exogenous variables: no parents → empirical distribution
        if len(parents) == 0:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())

        # Classification setting for categorical/discrete variables
        elif node in classifier_nodes:
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(
                        LGBMClassifier(**lgb_params)
                    )
                )
            )

        # Regression setting for continuous variables
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
# This step estimates all structural equations under the assumption of causal sufficiency.
try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print(" Mechanisms setup complete")

    print("Fitting SCM on training data (Normal Days)...")
    gcm.fit(scm, train_df)

    print(" Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Persist the trained SCM for reproducibility and downstream analysis.
# This checkpoint enables later counterfactual simulations and intervention queries.
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df),
            'train_domain': CONFIG['train_domain']
        },
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")
#%%



print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Include Weekly_Sales because we have Concept Shift
# Holiday patterns fundamentally change sales behavior
ordered_nodes = list(nx.topological_sort(feature_graph))

nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]


nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to holiday domain (learning new sales behavior during holidays)...")

try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            # Use ONLY ADAPTATION_DF for retraining (no leakage from holdout)
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Concept & Covariate shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,
            'adaptation_domain': CONFIG['test_domain']
        },
        stage_name="domain_adapted"
    )
except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

#%%



print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")


try:
    # Sample from the HOLDOUT TEST DATA which the model has NEVER seen
    test_sample_size = min(CONFIG['test_sample_size'], len(holdout_test_df))
    test_sample = holdout_test_df.sample(
        n=test_sample_size,
        random_state=CONFIG['random_seed']
    ).reset_index(drop=True)

    if len(test_sample) < CONFIG['test_sample_size']:
        print(
            f"WARNING: Test sample size ({len(test_sample)}) < "
            f"configured size ({CONFIG['test_sample_size']}). "
            f"Results may have higher variance."
        )

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")

    # Get causal mechanism and parents
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Generate predictions
    X_test = test_sample[parents].to_numpy()

    train_features = parents
    test_features = list(test_sample[parents].columns)

    missing = set(train_features) - set(test_features)
    extra = set(test_features) - set(train_features)

    print("Missing:", missing)
    print("Extra:", extra)

    if list(test_sample[parents].columns) != parents:
        print("️ Feature order mismatch!")
    else:
        print(" Feature order OK")
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
    # 1. Load the PRE-ADAPTATION SCM from checkpoint
    baseline_scm, baseline_metadata = load_checkpoint(scm_path_initial, metadata_path_initial)

    # 2. Use the EXACT SAME HOLDOUT SET for both models
    comparison_df = holdout_test_df.reset_index(drop=True)

    # 3. Evaluate baseline and adapted SCM on the same rows
    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        "BASELINE SCM (trained on Normal Days only)"
    )

    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        "ADAPTED SCM (adapted to Holiday Days)"
    )

    # 4. Fair side-by-side comparison
    
    print("FAIR HOLDOUT COMPARISON (Holiday Days)")
    
    print(f"{'Metric':<10} {'Baseline':<15} {'Adapted':<15} {'Delta':<15}")
    print("-" * 55)

    mae_delta = baseline_metrics_holdout["MAE"] - adapted_metrics_holdout["MAE"]
    rmse_delta = baseline_metrics_holdout["RMSE"] - adapted_metrics_holdout["RMSE"]
    mape_delta = baseline_metrics_holdout["MAPE"] - adapted_metrics_holdout["MAPE"]
    r2_delta = adapted_metrics_holdout["R2"] - baseline_metrics_holdout["R2"]

    print(
        f"{'MAE':<10} "
        f"{baseline_metrics_holdout['MAE']:<15.4f} "
        f"{adapted_metrics_holdout['MAE']:<15.4f} "
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
        f"{baseline_metrics_holdout['RMSE']:<15.4f} "
        f"{adapted_metrics_holdout['RMSE']:<15.4f} "
        f"{rmse_delta:+.4f}"
    )

    print(
        f"{'R2':<10} "
        f"{baseline_metrics_holdout['R2']:<15.4f} "
        f"{adapted_metrics_holdout['R2']:<15.4f} "
        f"{r2_delta:+.4f}"
    )

    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and \
            adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]:
        print("\n Adaptation improved performance on holiday days.")
        print("  The model learned different sales patterns specific to holidays.")
    else:
        print("\n WARNING: Adaptation did not clearly improve performance on holiday days.")
        print("  Holiday sales may follow similar patterns to normal days,")
        print("  or the adaptation set may be insufficient for learning.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise
