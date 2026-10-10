#%%
# !/usr/bin/env python3

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
    'train_stores': list(range(1, 31)),  # Stores 1 to 30
    'test_stores': list(range(31, 46)),  # Stores 31 to 45
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_store_id',

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
    df = df.fillna(0)

    # Ensure Store is an integer for correct filtering
    df['Store'] = pd.to_numeric(df['Store'], errors='coerce').fillna(-1).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Stores present: {sorted(df['Store'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY STORE NUMBER DOMAIN (WITH ADAPTATION SET)



print("PART 2: DOMAIN SHIFT SPLIT (STORES 1-30 vs 31-45)")


# Split based on Store Number
train_df = df[df['Store'].isin(CONFIG['train_stores'])].copy().reset_index(drop=True)
test_df_full = df[df['Store'].isin(CONFIG['test_stores'])].copy().reset_index(drop=True)



# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,  #  5% adaptation, 95% holdout
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain (Stores 1-30) shape: {train_df.shape}")
print(f"\nTest domain (Stores 31-45)")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

print(" Domain separation verified: No overlap between Store Numbers")

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
        if col in train_df.columns:
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


# Macro-level causal abstraction (store-level shift study)

# In this experiment, we model a structural causal system
# where "Store_Features" acts as a latent proxy for store identity.
#
# The goal is to evaluate how interventions at the store level
# propagate through downstream variables (promotions, sales, etc.).


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


# Structural causal assumptions (macro DAG)

# Edges encode hypothesized causal propagation pathways:
#   - temporal features → seasonality & demand
#   - holidays → promotions → sales
#   - store identity → local demand & pricing strategies


causal_edges_groups = [
    ("Date_Features", "Season"),
    ("Date_Features", "Holidays"),
    ("Date_Features", "Sales"),
    ("Date_Features", "Economy"),
    ("Season", "Weather"),
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


# Construction of macro-level causal DAG

# The graph encodes a directed acyclic structure representing
# assumed causal dependencies among latent feature groups.


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
plt.title("Causal Graph — Graph Final", fontsize=16)
plt.tight_layout()
plt.show()


# Transitive causal influence (structural propagation analysis)

# We compute the transitive closure of the DAG to identify
# all downstream variables affected by a given intervention.
#
# This corresponds to the causal "cone of influence".


def get_all_descendants(graph, source_node):
    """
    Compute transitive closure of causal descendants.

    Returns all nodes reachable via directed paths from source_node,
    capturing both direct and indirect causal effects.
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


affected_groups = get_all_descendants(macro_graph, "Store_Features")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Store ID: {set(macro_graph.successors('Store_Features'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")

print("\n Causal interpretation:")
print("  Store-level interventions propagate to promotions and sales")
print("  inducing structural distribution shifts in downstream variables.")

print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


# Fine-grained SCM construction (feature-level expansion)

# Macro variables are decomposed into observable features,
# producing a fully specified structural causal model.


feature_graph = nx.DiGraph()

for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

feature_graph.remove_edge('Store','Weekly_Sales')

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")


# Structural Causal Model (SCM) specification

# Each node is assigned a structural mechanism:
#   - root nodes → empirical distributions
#   - categorical nodes → probabilistic classifiers
#   - continuous nodes → regression-based structural equations


scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assign structural equations to each node in the SCM.
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

    print("Fitting SCM on training data (Store-regime 1–30)...")
    gcm.fit(scm, train_df)
    print(" Initial structural fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise


# Model checkpointing for reproducibility and auditability

# Stores SCM parameters, graph structure, and metadata for
# future reproducibility and comparative analysis.


try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df),
            'train_domain': "Stores 1-30"
        },
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")
#%%
print("PART 7: SUPERVISED DOMAIN ADAPTATION FOR NEW STORES")


# Domain adaptation under covariate + concept shift

# This stage implements supervised structural adaptation of the SCM
# under a new store regime (Stores 31–45).
#
# The underlying assumption is that store identity induces:
#   (i) covariate shift (distributional changes in inputs)
#   (ii) potential concept shift (changes in P(Weekly_Sales | X))
#
# Therefore, selected structural mechanisms are updated using
# target-domain (adaptation_df) observations.



ordered_nodes = list(nx.topological_sort(feature_graph))


# Selection of nodes affected by structural intervention

# Only nodes lying in the causal downstream of Store_Features
# are considered for retraining, ensuring localised adaptation
# rather than full model re-estimation.


nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Store Features + MarkDowns + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to new stores (learning new local behaviors)...")

try:
    refit_count = 0


    # Structural mechanism re-estimation under target domain

    # Each selected node is refit using only adaptation samples
    # to avoid leakage from holdout evaluation data.


    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")

            fit_causal_model_of_target(scm, node, adaptation_df)

            refit_count += 1

        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")


    # Checkpointing adapted SCM

    # Stores the post-adaptation structural model for comparison
    # against the baseline SCM under identical evaluation regime.


    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,
            'adaptation_domain': "Stores 31-45"
        },
        stage_name="domain_adapted"
    )

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise
#%%
print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")


# Holdout evaluation after causal domain adaptation
# This section evaluates model generalization on unseen data
# from a different domain (Stores 31-45 / Season / Holiday shift)


try:
    # Sample strictly from HOLDOUT SET (never used in training/adaptation)
    test_sample_size = min(CONFIG['test_sample_size'], len(holdout_test_df))

    test_sample = holdout_test_df.sample(
        n=test_sample_size,
        random_state=CONFIG['random_seed']
    ).reset_index(drop=True)

    # Safety check: warn if holdout is smaller than expected config
    if len(test_sample) < CONFIG['test_sample_size']:
        print(
            f"WARNING: Test sample size ({len(test_sample)}) < "
            f"configured size ({CONFIG['test_sample_size']}). "
            f"Results may have higher variance."
        )

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")


    # Causal prediction step:
    # We extract the learned SCM mechanism for Weekly_Sales
    # and use its trained ML regression model

    sales_mechanism = scm.causal_mechanism("Weekly_Sales")

    # Parent features define the causal input space of the target node
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Convert evaluation data into model input matrix
    X_test = test_sample[parents].to_numpy()

    # Predict Weekly_Sales using learned structural equation
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()


    # Standard regression evaluation metrics

    mae = mean_absolute_error(test_sample['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_sample['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_sample['Weekly_Sales'].values, test_predictions)
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


# Utility function for consistent evaluation of Weekly_Sales SCM
# This ensures both baseline and adapted models are compared
# on EXACTLY the same data distribution (no leakage, fair test)


def evaluate_sales_on_holdout(model_scm, eval_df, label):

    # Extract causal parents of target node Weekly_Sales
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Retrieve learned structural equation (prediction mechanism)
    sales_mechanism = model_scm.causal_mechanism("Weekly_Sales")

    # Build feature matrix aligned with causal parent structure
    X_eval = eval_df[parents].to_numpy()

    # Ground truth target values
    y_true = eval_df["Weekly_Sales"].to_numpy()

    # Model predictions from learned causal mechanism
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()


    # Standard regression evaluation metrics

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

    # Load baseline SCM (pre-domain adaptation model)

    baseline_scm, baseline_metadata = load_checkpoint(
        scm_path_initial,
        metadata_path_initial
    )

    # Use identical holdout dataset for fair comparison
    comparison_df = holdout_test_df.reset_index(drop=True)


    # Evaluate baseline model

    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        "BASELINE SCM (Learned only on Stores 1-30)"
    )


    # Evaluate adapted model (after causal retraining)

    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        "ADAPTED SCM (Adapted to Stores 31-45)"
    )


    # Side-by-side comparison table

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
        f"{'MAPE':<10} "
        f"{baseline_metrics_holdout['MAPE']:<15.4f} "
        f"{adapted_metrics_holdout['MAPE']:<15.4f} "
        f"{mape_delta:+.4f}"
    )

    print(
        f"{'R2':<10} "
        f"{baseline_metrics_holdout['R2']:<12.4f} "
        f"{adapted_metrics_holdout['R2']:<12.4f} "
        f"{r2_delta:+.4f}"
    )


    # Interpretation of domain adaptation effect

    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"]:
        print("\n Adaptation successfully captured the Store Shift.")
    else:
        print("\n⚠ WARNING: Adaptation did not clearly improve performance.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise