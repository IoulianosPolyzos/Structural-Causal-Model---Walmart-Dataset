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
    'train_store_types': ['C'],
    'test_store_types': ['A', 'B'],
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_store_type',

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

    print(f"Dataset shape: {df.shape}")
    print(f"Store Type distribution:")
    for store_type in sorted(df['Type'].unique()):
        print(f"  Type {store_type}: {(df['Type'] == store_type).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY STORE TYPE DOMAIN (WITH ADAPTATION SET)



print("PART 2: DOMAIN SHIFT SPLIT (TYPE C vs TYPES A/B)")


# Split based on Store Type
train_df = df[df['Type'].isin(CONFIG['train_store_types'])].copy().reset_index(drop=True)
test_df_full = df[df['Type'].isin(CONFIG['test_store_types'])].copy().reset_index(drop=True)



# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,  #  5% adaptation, 95% holdout
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain (Types {CONFIG['train_store_types']}) shape: {train_df.shape}")
print(f"\nTest domain (Types {CONFIG['test_store_types']})")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

print(" Domain separation verified: No overlap between Store Types")

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


# Definition of causal macro-groups (high-level latent variables)
# Each group represents a conceptual driver of Weekly_Sales

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


# Macro-level causal structure (group-to-group relationships)
# This defines the assumed structural causal DAG at concept level

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

# Build macro causal graph (group-level DAG)
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

# Fixed layout for interpretability of causal structure visualization
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),"Store_Features":(1,2),
    "Economy": (2, 2), "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1), "Sales": (4, 0)
}

# Visualize macro-level causal graph
plt.figure(figsize=(13, 8))
nx.draw(
    macro_graph,
    pos,
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


# Utility: transitive causal effect traversal (BFS)
# Used to identify all downstream variables affected by a node

def get_all_descendants(graph, source_node):
    """
    Returns all nodes reachable from source_node (transitive closure).
    Captures full causal propagation chain in DAG.
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


# Identify causal impact propagation from Store_Features

affected_groups = get_all_descendants(macro_graph, "Store_Features")

# Expand group-level impact → feature-level nodes
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Store_Features: {set(macro_graph.successors('Store_Features'))}")
print(f"All affected groups (transitive): {affected_groups}")

print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")

print("\nInterpretation (causal shift rationale):")
print("MarkDowns and Sales are refitted because:")
print("  Store_Type → Store_Features → MarkDowns")
print("  Store_Type → Store_Features → Sales")
print("  Different store types induce different demand and promotion distributions.")



print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


# Expand macro causal graph into feature-level SCM graph
# Each feature becomes an individual causal node

feature_graph = nx.DiGraph()

for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Remove explicit leakage edge (target leakage prevention)
feature_graph.remove_edge('Type','Weekly_Sales')

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")



# PART 6: FIT INITIAL STRUCTURAL CAUSAL MODEL (SCM)


print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Initialize SCM from causal DAG
scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assigns structural causal mechanisms to each node:
    - Root nodes → empirical distribution
    - Categorical nodes → classification models
    - Continuous nodes → regression models
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

    print("Fitting SCM on training data (Store Type C)...")
    gcm.fit(scm, train_df)

    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise


# Save checkpoint for reproducibility + domain shift tracking
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df),
            'train_domain': str(CONFIG['train_store_types'])
        },
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")
#%%
print("PART 7: SUPERVISED DOMAIN ADAPTATION FOR STORE TYPES")

# Topological sorting ensures causal consistency during sequential refitting.
# Parent nodes are updated before child nodes to preserve structural validity of the SCM.
ordered_nodes = list(nx.topological_sort(feature_graph))

# Selection of nodes affected by the identified causal shift.
# This corresponds to the transitive closure of the intervention set (Store_Features).
# Only nodes influenced by the domain change are eligible for adaptation.
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Store Features + MarkDowns + TARGET included - {len(nodes_to_refit)} nodes):")

# Reporting affected subgraph induced by the domain shift.
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to new store types (learning new sales scale and behavior)...")

try:
    refit_count = 0

    # Iterative mechanism-level adaptation under covariate + concept shift.
    # Each node’s conditional distribution is re-estimated using adaptation_df only,
    # simulating target-domain retraining without contamination from holdout data.
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")

            # Structural Causal Model mechanism update:
            # replaces source-domain conditional mechanism with target-domain estimate.
            # This enables localized adaptation without retraining full SCM.
            fit_causal_model_of_target(scm, node, adaptation_df)

            refit_count += 1

        except Exception as e:
            # Robust training: failure of individual mechanisms does not collapse SCM.
            print(f"WARNING: Failed to refit {node}: {e}")

    # Summary of successful mechanism updates across affected subgraph.
    print(f"\n Shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    # Persist adapted SCM state for controlled comparison with baseline model.
    # Ensures reproducibility of counterfactual evaluation under identical test conditions.
    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,

        # Parent set of target node defines inference interface for evaluation.
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),

        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,

            # Encodes target shift regime for experimental traceability.
            'adaptation_domain': str(CONFIG['test_store_types'])
        },
        stage_name="domain_adapted"
    )

except Exception as e:
    # Global failure handler for SCM adaptation pipeline.
    # Ensures propagation of critical errors for debugging and reproducibility tracking.
    print(f"ERROR: Domain adaptation failed: {e}")
    raise
#%%
print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")

# Evaluation of the adapted Structural Causal Model on an unseen holdout dataset.
# This step estimates generalization performance under the target (shifted) domain.

try:
    # Subsampling the holdout set for computational efficiency and variance control.
    # The sampling is deterministic due to fixed random seed for reproducibility.
    test_sample_size = min(CONFIG['test_sample_size'], len(holdout_test_df))
    test_sample = holdout_test_df.sample(
        n=test_sample_size,
        random_state=CONFIG['random_seed']
    ).reset_index(drop=True)

    # Sanity check: ensures requested sample size was feasible.
    # Deviations may increase estimator variance.
    if len(test_sample) < CONFIG['test_sample_size']:
        print(
            f"WARNING: Test sample size ({len(test_sample)}) < "
            f"configured size ({CONFIG['test_sample_size']}). "
            f"Results may have higher variance."
        )

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")

    # Extract causal mechanism corresponding to target variable (Weekly_Sales).
    # This mechanism encodes the learned conditional distribution:
    # P(Weekly_Sales | Parents(Weekly_Sales))
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")

    # Retrieve parent set of the target node from the causal graph.
    # This defines the feature space used for prediction consistency.
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Construct design matrix aligned with causal parent ordering.
    X_test = test_sample[parents].to_numpy()

    # Generate predictions using the learned structural mechanism.
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    # Standard regression evaluation metrics under holdout distribution.
    mae = mean_absolute_error(test_sample['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_sample['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_sample['Weekly_Sales'].values,test_predictions)
    r2 = r2_score(test_sample['Weekly_Sales'].values, test_predictions)

    # Report predictive performance on unseen domain data.
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
    # Failure handling for evaluation stage.
    # Ensures visibility of runtime issues in prediction pipeline.
    print(f"ERROR: Test evaluation failed: {e}")
    raise
#%%
print("PART 10: FAIR COMPARISON - NORMAL DAYS MODEL vs HOLIDAY ADAPTED MODEL")

# This section performs a controlled evaluation comparing:
# (1) Baseline SCM trained on source domain (no adaptation)
# (2) Adapted SCM after mechanism-level updates under target shift
# The evaluation is performed on an identical holdout dataset to ensure fairness.

def evaluate_sales_on_holdout(model_scm, eval_df, label):
    # Retrieve causal parents of the target node from the structural graph.
    # This ensures consistent feature ordering across models.
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Extract learned causal mechanism for the target variable.
    sales_mechanism = model_scm.causal_mechanism("Weekly_Sales")

    # Construct evaluation design matrix aligned with causal parents.
    X_eval = eval_df[parents].to_numpy()

    # Ground truth target values from holdout distribution.
    y_true = eval_df["Weekly_Sales"].to_numpy()

    # Generate predictions using the structural prediction model.
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()

    # Standard regression performance metrics.
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": custom_retail_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    # Report model performance for qualitative comparison.
    print(f"\n{label}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}")
    print(f"R²:   {metrics['R2']:.4f}")


    return metrics

try:
    # Load baseline SCM trained exclusively on source domain data.
    # This represents the non-adapted reference model.
    baseline_scm, baseline_metadata = load_checkpoint(scm_path_initial, metadata_path_initial)

    # Ensure identical evaluation dataset for fair comparison.
    comparison_df = holdout_test_df.reset_index(drop=True)

    # Evaluate baseline model performance on holdout domain.
    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        f"BASELINE SCM (Learned only on Store Type C)"
    )

    # Evaluate adapted SCM after domain shift adjustment.
    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        f"ADAPTED SCM (Adapted to Store Types A & B)"
    )

    # Summary table for side-by-side comparison of predictive performance.
    print("\n FAIR HOLDOUT COMPARISON ")
    print(f"{'Metric':<10} {'Baseline':<12} {'Adapted':<12} {'Delta':<12}")
    print("-" * 52)

    mae_delta = baseline_metrics_holdout["MAE"] - adapted_metrics_holdout["MAE"]
    rmse_delta = baseline_metrics_holdout["RMSE"] - adapted_metrics_holdout["RMSE"]
    mape_delta = baseline_metrics_holdout["MAPE"] - adapted_metrics_holdout["MAPE"]
    r2_delta = adapted_metrics_holdout["R2"] - baseline_metrics_holdout["R2"]

    print(
        f"{'MAE':<10} {baseline_metrics_holdout['MAE']:<12.4f} "
        f"{adapted_metrics_holdout['MAE']:<12.4f} {mae_delta:+.4f}"
    )

    print(
        f"{'MAPE':<10} "
        f"{baseline_metrics_holdout['MAPE']:<15.4f} "
        f"{adapted_metrics_holdout['MAPE']:<15.4f} "
        f"{mape_delta:+.4f}"
    )

    print(
        f"{'RMSE':<10} {baseline_metrics_holdout['RMSE']:<12.4f} "
        f"{adapted_metrics_holdout['RMSE']:<12.4f} {rmse_delta:+.4f}"
    )
    print(
        f"{'R2':<10} {baseline_metrics_holdout['R2']:<12.4f} "
        f"{adapted_metrics_holdout['R2']:<12.4f} {r2_delta:+.4f}"
    )

    # Decision heuristic based on predictive improvement under domain shift.
    # Note: this evaluates predictive utility, not causal validity.
    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"]:
        print("\n Adaptation successfully captured the Store Type Shift.")
    else:
        print("\n WARNING: Adaptation did not clearly improve performance.")

except Exception as e:
    # Global evaluation failure handler.
    # Ensures traceability of comparison pipeline issues.
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise