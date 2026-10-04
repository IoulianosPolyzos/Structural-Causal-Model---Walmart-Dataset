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
sys.path.append("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm")
from custom_mape import custom_retail_mape
with open("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]


CONFIG = {
    'train_domain': 'Normal Days (IsHoliday=0)',
    'test_domain': 'Holiday Days (IsHoliday=1)',
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_holidays',

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
        tuple: (scm, metadata)
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
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"IsHoliday distribution:")
    print(f"  Normal Days (0): {(df['IsHoliday'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['IsHoliday'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY HOLIDAY DOMAIN



print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs HOLIDAY)")


# Ensure IsHoliday is numeric (0 or 1)
df['IsHoliday'] = df['IsHoliday'].astype(int)

train_df = df[df['IsHoliday'] == 0].copy().reset_index(drop=True)
test_df_full = df[df['IsHoliday'] == 1].copy().reset_index(drop=True)



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
assert (train_df['IsHoliday'] == 0).all(), "Train set contains non-zero IsHoliday!"
assert (adaptation_df['IsHoliday'] == 1).all(), "Adaptation set contains non-holiday days!"
assert (holdout_test_df['IsHoliday'] == 1).all(), "Holdout set contains non-holiday days!"
print(" Domain separation verified: No overlap between Normal/Holiday days")

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
# PART 4: CAUSAL GRAPH ANALYSIS (MACRO-LEVEL STRUCTURE)


print("PART 4: CAUSAL GRAPH ANALYSIS")


# Definition of macro-level causal abstractions

# The dataset variables are grouped into semantically meaningful clusters
# representing latent causal concepts (e.g., Holidays, Weather, Economy).
# This hierarchical representation reduces complexity and improves interpretability
# of the causal structure.

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


# Macro-level causal edges (group-to-group relationships)

# These directed edges encode domain assumptions about causal influence
# between latent concepts, forming a directed acyclic graph (DAG).
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


# Construction of macro-level causal graph

# NetworkX directed graph is used to represent causal dependencies
# between high-level feature groups.
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)


# Visualization layout specification

# Manual positioning ensures interpretability of causal hierarchy.
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),
    "Store_Features": (1, 2),
    "Economy": (2, 2),
    "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1),
    "Sales": (4, 0)
}

# Plot macro-level causal DAG
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



# CAUSAL EFFECT PROPAGATION ANALYSIS


def get_all_descendants(graph, source_node):
    """
    Compute transitive closure of a node in a directed graph.

    In causal inference terms, this identifies all downstream variables
    that are potentially affected by interventions on the source node.
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


# Identify all downstream effects of intervening on Holidays
affected_groups = get_all_descendants(macro_graph, "Holidays")

# Expand group-level effects to feature-level variables
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

# Report causal impact propagation
print(f"Direct children of Holidays: {set(macro_graph.successors('Holidays'))}")
print(f"All affected groups (transitive): {affected_groups}")

print(f"\nAffected feature nodes ({len(affected_feature_nodes)}):")
for node in affected_feature_nodes:
    print(f"  - {node}")

# Interpretability statement (domain justification)
print("\n YES, MarkDowns and Sales are refitted because:")
print("  Holidays → MarkDowns (direct effect)")
print("  Holidays → Sales (direct effect)")
print("  Holiday periods induce structural changes in demand and promotions")



# PART 5: FEATURE-LEVEL CAUSAL GRAPH EXPANSION


print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")

# Initialize feature-level DAG (fine-grained representation)
feature_graph = nx.DiGraph()

# Add all observed variables as nodes
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

# Expand group-level edges into full bipartite feature dependencies
for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Remove direct edge violating modeling assumption (manual constraint)
feature_graph.remove_edge('IsHoliday','Weekly_Sales')

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")



# PART 6: STRUCTURAL CAUSAL MODEL FITTING


print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Instantiate Structural Causal Model over feature-level DAG
scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assign structural causal mechanisms to each node.

    - Root nodes: empirical distributions (non-parametric)
    - Categorical nodes: classifier-based functional causal models
    - Continuous nodes: additive noise models with regression estimators
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


# Fit SCM using observational training data
try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)

    print(" Mechanisms setup complete")

    print("Fitting SCM on training data (Normal Days)...")
    gcm.fit(scm, train_df)

    print(" Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise


# Save trained SCM checkpoint for reproducibility and downstream analysis
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
# PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)


print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")


# Causal ordering of variables

# A topological ordering of the causal graph ensures that parent nodes are
# processed before their descendants. This is critical for maintaining
# structural consistency during sequential refitting of mechanisms.
ordered_nodes = list(nx.topological_sort(feature_graph))


# Identification of nodes affected by domain shift

# Only variables downstream of affected causal groups are selected for
# retraining. This implements a principled form of localized adaptation
# under distribution shift.
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")

# Display selected nodes for adaptation
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to holiday domain (learning new sales behavior during holidays)...")


# Structural adaptation under domain shift

# This procedure performs selective retraining of structural mechanisms
# using only the adaptation dataset. This avoids data leakage from the
# holdout test set and ensures valid evaluation under domain shift.
try:
    refit_count = 0

    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")

            # Local structural retraining for node-specific mechanism
            # using adaptation data only (covariate + concept shift handling)
            fit_causal_model_of_target(scm, node, adaptation_df)

            refit_count += 1

        except Exception as e:
            # Fail-safe mechanism: continue adaptation even if individual node fails
            print(f"WARNING: Failed to refit {node}: {e}")


    # Summary of adaptation process

    print(f"\n Concept & Covariate shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")


    # Persist adapted SCM for evaluation and comparison

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,
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

# PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)


print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")


# Out-of-sample evaluation under domain shift

# The holdout test set represents a strictly unseen subset of the target
# domain (holiday period). This ensures unbiased evaluation of the adapted
# structural causal model under distribution shift conditions.
try:

    # Sampling protocol

    # A fixed-size random sample is drawn to control computational cost
    # while maintaining statistical representativeness.
    test_sample_size = min(CONFIG['test_sample_size'], len(holdout_test_df))

    test_sample = holdout_test_df.sample(
        n=test_sample_size,
        random_state=CONFIG['random_seed']
    ).reset_index(drop=True)

    # Warn if sampling constraint reduces statistical power
    if len(test_sample) < CONFIG['test_sample_size']:
        print(
            f"WARNING: Test sample size ({len(test_sample)}) < "
            f"configured size ({CONFIG['test_sample_size']}). "
            f"Results may have higher variance."
        )

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")


    # Structural prediction via causal mechanism

    # The prediction is obtained from the learned structural mechanism of
    # the target variable (Weekly_Sales), conditioned on its parent set
    # in the learned causal graph.
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Construct design matrix using causal parents only
    X_test = test_sample[parents].to_numpy()

    # Generate predictions using learned structural function
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()


    # Predictive performance evaluation

    # Standard regression metrics are used to quantify predictive accuracy
    # under distribution shift conditions.
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

# PART 9: FAIR COMPARISON - BASELINE vs ADAPTED ON SAME HOLDOUT SET


print("PART 9: FAIR COMPARISON - BASELINE vs ADAPTED ON SAME HOLDOUT SET")


# Evaluation function under consistent causal parent space

# This function ensures that both baseline and adapted SCMs are evaluated
# under identical feature conditioning sets (i.e., same causal parents),
# enabling a fair counterfactual comparison of model performance.
def evaluate_sales_on_holdout(model_scm, eval_df, label):
    """
    Evaluate the structural mechanism of Weekly_Sales on a fixed dataset.

    This evaluation isolates predictive performance of the learned causal
    mechanism under a fixed structural representation.
    """
    # Extract causal parents of target variable
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    # Retrieve learned structural mechanism
    sales_mechanism = model_scm.causal_mechanism("Weekly_Sales")

    # Construct evaluation design matrix
    X_eval = eval_df[parents].to_numpy()
    y_true = eval_df["Weekly_Sales"].to_numpy()

    # Generate predictions from learned structural function
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()

    # Compute standard regression metrics
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Mape": custom_retail_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    # Report results
    print(f"\n{label}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}")
    print(f"R²:   {metrics['R2']:.4f}")



    return metrics



# Baseline vs adapted SCM evaluation pipeline

try:
    # -
    # Load baseline (pre-adaptation) SCM
    # -
    # This model represents the structural causal system learned
    # exclusively from the source (normal-day) distribution.
    baseline_scm, baseline_metadata = load_checkpoint(
        scm_path_initial,
        metadata_path_initial
    )

    # Use identical holdout dataset for fair comparison
    comparison_df = holdout_test_df.reset_index(drop=True)

    # -
    # Evaluate baseline SCM
    # -
    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm,
        comparison_df,
        "BASELINE SCM (trained on Normal Days only)"
    )

    # -
    # Evaluate adapted SCM
    # -
    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        comparison_df,
        "ADAPTED SCM (adapted to Holiday Days)"
    )

    # -
    # Side-by-side comparative analysis
    # -
    print("FAIR HOLDOUT COMPARISON (Holiday Days)")

    print(f"{'Metric':<10} {'Baseline':<15} {'Adapted':<15} {'Delta':<15}")
    print("-" * 55)

    # Compute deltas (positive = improvement depending on metric)
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

    # -
    # Interpretation of domain adaptation effect
    # -
    # Improvement is defined jointly by:
    # - Lower MAE
    # - Lower RMSE
    # - Higher R²
    if (adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and
        adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]):

        print("\n Adaptation improved performance on holiday days.")
        print("  The model successfully learned shifted sales dynamics under intervention-like conditions.")

    else:
        print("\nWARNING: Adaptation did not clearly improve performance on holiday days.")
        print("  This may indicate:")
        print("  (i) weak distribution shift between domains,")
        print("  (ii) insufficient adaptation sample size, or")
        print("  (iii) stable causal mechanisms across domains.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise