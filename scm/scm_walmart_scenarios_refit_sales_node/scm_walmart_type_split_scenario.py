# %%
"""
This script implements a causal-based domain adaptation framework.
It leverages Structural Causal Models (SCMs) to address distribution shifts
(covariate and concept shifts) across different spatial domains (Store Types).
"""

from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel

# !/usr/bin/env python3

import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

# Suppress warnings to maintain clean execution logs during experimental runs
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Predictive modeling and structural estimation dependencies
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
sys.path.append("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm")
from custom_mape import custom_retail_mape


# CONFIGURATION AND SETUP

# Load experimental hyperparameters from configuration file
try:
    with open("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    lgb_params = config["lgb_params"]
except FileNotFoundError:
    print(
        "No /home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml found, using default LGBM parameters.")
    lgb_params = {'n_estimators': 100, 'random_state': 42}

# Define domain split parameters: Source domain (Type C) vs Target domain (Type A, B)
CONFIG = {
    'source_store_types': ['C'],
    'target_store_types': ['A', 'B'],
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_type',
}

# Create checkpoint directory for model serialization
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)


def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Serializes the SCM and its structural metadata for reproducibility
    and multi-stage experimental pipelines.
    """
    try:
        checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scm_path = checkpoint_dir / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

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

        print(f"✓ Checkpoint saved: {scm_path.name}")
        return str(scm_path), str(metadata_path)
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise


def load_checkpoint(scm_path, metadata_path):
    """Restores the SCM and graph structure from persistent storage."""
    try:
        with open(scm_path, "rb") as f:
            scm = pickle.load(f)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        print(f"✓ Checkpoint loaded from {Path(scm_path).name}")
        return scm, metadata
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise


# %%


# PART 1: DATA INGESTION

print("PART 1: LOADING DATA & SETUP")

try:
    # Load observational data and remove temporal confounders not modeled explicitly
    df = pd.read_csv("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv")
    df = df.drop(
        columns=["Date", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Zero-imputation for missing observational records

    # Ensure Store identifier consistency for domain stratification
    df['Store'] = pd.to_numeric(df['Store'], errors='coerce').fillna(-1).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Stores present: {sorted(df['Store'].unique())}")

except FileNotFoundError:
    print(
        "ERROR: Data file '/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv' not found!")
    raise


# PART 2: DOMAIN SHIFT PARTITIONING

print("\n" + "=" * 70)
print("PART 2: DOMAIN SHIFT SPLIT (STORES C vs A&B)")

# Stratify dataset to induce an Out-of-Distribution (OOD) scenario based on 'Type'
source_df = df[df['Type'].isin(CONFIG['source_store_types'])]
target_df = df[df['Type'].isin(CONFIG['target_store_types'])]

# Partition target domain into an adaptation subset (for mechanism refitting)
# and a strictly held-out subset (for unbiased evaluation)
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source domain (Type C) shape: {source_df.shape}")
print(f"\nTarget domain (Type A & B)")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

print("✓ Domain separation verified: No overlap between Store Numbers")

# %%


# PART 3: PREPROCESSING & DATA TYPING

print("PART 3: PREPROCESSING")

try:
    # Segregate continuous from categorical variables to assign appropriate causal mechanisms
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Enforce numeric types for continuous causal nodes
    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Cast boolean/binary indicators to string representations for classification mechanisms
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start",
                    "Is_Month_End"]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Cast categorical features
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        source_df[col] = source_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%


# PART 4: DIRECTED ACYCLIC GRAPH (DAG) FORMULATION

print("PART 4: CAUSAL GRAPH ANALYSIS")

# Define macro-level causal groupings based on domain knowledge
causal_groups = {
    "Holidays": ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "holiday_proximity", "holiday_weight"],
    "Date_Features": [
        "Year", "Quarter",
        "Year_Progress", "Month_Progress",
        "Is_Month_Start", "Is_Month_End",
        "Is_Summer",
        "Month_sin", "Month_cos",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "WeekOfYear_sin", "WeekOfYear_cos", ],
    "Season": ["Season_sin", "Season_cos"],
    "City": ["city"],
    "Economy": ["Fuel_Price", "CPI", "Unemployment"],
    "Weather": [
        "weather_condition", "Temperature",
        "precipitation", "wind_speed", "humidity", "temperature_min", "temperature_max"
    ],
    "Store_Features": ["Store", "Type", "Size", "Dept"],
    "MarkDowns": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],
    "Sales": ["Weekly_Sales"]
}

# Postulate structural causal edges between macro-groups
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

# Construct the macro-graph to infer transitive causal effects
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)


def get_all_descendants(graph, source_node):
    """Retrieves the transitive closure of descendants for a given node."""
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


# Identify all covariates causally downstream of the domain intervention ('Store_Features')
affected_groups = get_all_descendants(macro_graph, "Store_Features")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Store_Features: {set(macro_graph.successors('Store_Features'))}")
print(f"All affected groups (transitive): {affected_groups}")


# PART 5: FEATURE-LEVEL CAUSAL GRAPH

print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")

feature_graph = nx.DiGraph()

# Expand macro-groups into granular feature-level nodes
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

# Establish directed edges across individual features based on macro-edges
for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Domain assumption: 'Type' influences sales indirectly through other store characteristics
feature_graph.remove_edge('Type', 'Weekly_Sales')
print("Type successors : ", list(feature_graph.successors("Type")))
for edge in feature_graph.edges:
    print(edge)

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")


# PART 6: SCM ESTIMATION ON SOURCE DOMAIN

print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assigns non-parametric Functional Causal Models (FCMs) to each node.
    Root nodes utilize empirical distributions, while downstream nodes are modeled
    via Additive Noise Models (ANM) using gradient boosting estimators.
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        if len(parents) == 0:
            # Exogenous variables
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif node in classifier_nodes:
            # Categorical endogenous variables
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(LGBMClassifier(**lgb_params))
                )
            )
        else:
            # Continuous endogenous variables (ANM framework)
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    SklearnRegressionModel(LGBMRegressor(**lgb_params))
                )
            )


try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on training data...")
    gcm.fit(scm, source_df)  # Observational fitting phase
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

try:
    # Serialize the base source SCM
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'initial_fit_complete'},
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save checkpoint: {e}")

# %%


# PART 8: SUPERVISED DOMAIN ADAPTATION

print("PART 8: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Topological sorting ensures ancestral dependencies are respected during refitting
ordered_nodes = list(nx.topological_sort(feature_graph))

# Isolate causal mechanisms vulnerable to shift based on DAG topography
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit ({len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"  ✓ {node}")

print(f"\nAdapting mechanisms to adaptation set (learning new behavior for features)...")

try:
    # Update localized mechanisms using the limited target domain adaptation set
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n✓ Concept shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

# %%


# PART 9: COUNTERFACTUAL / SYNTHETIC TARGET DATA GENERATION

print("PART 9: SYNTHETIC DATA GENERATION & ML TRAINING")

train_types = ['C']
test_types = ['A', 'B']


# Define soft intervention policy mapping source environments to target environments
def type_intervention_fn(x):
    return np.random.choice(test_types)


num_synthetic_samples = len(target_df)

# Generate interventional distributions sampling from the adapted SCM
synthetic_dataset = gcm.interventional_samples(
    scm,
    interventions={'Type': type_intervention_fn},
    num_samples_to_draw=num_synthetic_samples
)

print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")

# Post-generation heuristic filtering for domain constraints (non-negative sales/markdowns)
cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

synthetic_dataset[cols] = synthetic_dataset[cols].clip(lower=0)

print(f"Synthetic samples after removing negatives: {len(synthetic_dataset)}")
print("\nFeature statistics comparison:")
for col in ['MarkDown1', 'MarkDown2', 'Fuel_Price', 'CPI']:
    if col in synthetic_dataset.columns:
        print(f"\n{col}:")
        print(f"  Synthetic: mean={synthetic_dataset[col].mean():.2f}")
        print(f"  Test:      mean={holdout_test_df[col].mean():.2f}")
        print(f"  Ratio:     {holdout_test_df[col].mean() / (synthetic_dataset[col].mean() + 1e-6):.2f}")
print("\nWeekly_Sales parents:", list(feature_graph.predecessors("Weekly_Sales")))
print("Number of parents:", feature_graph.in_degree("Weekly_Sales"))

# Enforce consistent categorical levels between synthetic and observational cohorts
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

print(f"Synthetic Dataset Shape: {synthetic_dataset.shape}")

print("\nTrain ML model on Synthetic Dataset...")

# Train surrogate predictive model optimized purely on synthetic target distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)
# %%

# Distributional sanity checks
print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")

print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {source_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['Weekly_Sales'].std():.4f}")
# %%
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

# %%


# PART 10: DOWNSTREAM EVALUATION (SYNTHETIC MODEL)

print("PART 10: EVALUATION ON TEST SET")

print("\nEvaluation on Test Set ...")

# Prepare unbiased holdout set for rigorous evaluation
X_test = holdout_test_df.drop(columns=['Weekly_Sales'])

X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

# Inference phase
y_pred = ml_model_syn.predict(X_test)

# Metric compilation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Type': holdout_test_df['Type'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%


# HYBRID DATASET FORMULATION

print("\nHybrid Dataset (Source + Synthetic Target)...")

# Combine factual source observations with synthetic counterfactuals to augment support
train_aligned = source_df[synthetic_dataset.columns].copy()

hybrid_dataset = pd.concat([train_aligned, synthetic_dataset], ignore_index=True)

for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    hybrid_dataset[col] = pd.Categorical(hybrid_dataset[col].astype(str), categories=all_categories)

print(f"Hybrid Dataset Shape: {hybrid_dataset.shape} (Real: {len(train_aligned)}, Synthetic: {len(synthetic_dataset)})")

X_hybrid = hybrid_dataset.drop(columns=['Weekly_Sales'])
y_hybrid = hybrid_dataset['Weekly_Sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

# Train hybrid robust model
ml_model_hbr.fit(X_hybrid, y_hybrid)
# %%

# Hybrid cohort validation
print("\n>>> RAW HYBRID DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {hybrid_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {hybrid_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {hybrid_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {hybrid_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(hybrid_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")

print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {source_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['Weekly_Sales'].std():.4f}")

# %%
print("\nEvaluation on Test Set ...")

X_test = holdout_test_df.drop(columns=['Weekly_Sales'])

X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

# Evaluate hybrid regressor generalization
y_pred = ml_model_hbr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)
print(f"\nPrediction Results:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")
results_df = pd.DataFrame({
    'Type': holdout_test_df['Type'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%
# %%

# Refit the direct target mechanism for diagnostic bounds
fit_causal_model_of_target(scm, "Weekly_Sales", hybrid_dataset)


def evaluate_sales_on_holdout(model_scm, eval_df, label):
    """
    Evaluates the inherent NP-SEM predicting Sales purely
    based on its structural parents.
    """
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


adapted_metrics_holdout = evaluate_sales_on_holdout(
    scm,
    holdout_test_df,
    "ADAPTED SCM (after adaptation)"
)

# %%
# %%
from sklearn.model_selection import KFold
from xgboost import XGBClassifier



# DEGRADATION DECOMPOSITION & DIAGNOSTICS

def degradation_decomp_regression(
        source_X,
        source_y,
        target_X_raw,
        target_y_raw,
        model,
        data_sum=20000,
        K=8,
        domain_classifier=None, draw_calibration=False, save_calibration_png='calibration.png'
):
    """
    Decomposes the out-of-distribution (OOD) performance gap into components
    attributable to Covariate Shift (X-shift) and Concept Shift (Y|X-shift).
    Uses density ratio estimation via a probabilistic classifier.
    """
    perm1 = np.random.permutation(target_X_raw.shape[0])
    target_X = target_X_raw[perm1[:data_sum], :]
    target_y = target_y_raw[perm1[:data_sum]]

    piA = np.zeros(source_X.shape[0])
    piB = np.zeros(target_X.shape[0])

    permA = np.random.permutation(source_X.shape[0])
    permB = np.random.permutation(target_X.shape[0])

    # K-fold cross-fitting to mitigate overfitting in density ratio estimation
    kf = KFold(n_splits=K, shuffle=False)

    A_train_index_list, A_test_index_list = [], []
    B_train_index_list, B_test_index_list = [], []

    for train_idx, test_idx in kf.split(source_X):
        A_train_index_list.append(train_idx)
        A_test_index_list.append(test_idx)

    for train_idx, test_idx in kf.split(target_X):
        B_train_index_list.append(train_idx)
        B_test_index_list.append(test_idx)

    for i in range(K):
        # Concatenate folds for domain discriminator training
        trainX = np.concatenate([
            source_X[permA[A_train_index_list[i]]],
            target_X[permB[B_train_index_list[i]]]
        ], axis=0)

        trainT = np.zeros(trainX.shape[0])
        trainT[len(A_train_index_list[i]):] = 1.0

        if domain_classifier is None:
            clf = XGBClassifier(random_state=0)
        else:
            clf = domain_classifier

        clf.fit(trainX, trainT)

        # Estimate propensity scores Pr(Domain=Target | X)
        piA[permA[A_test_index_list[i]]] = clf.predict_proba(
            source_X[permA[A_test_index_list[i]]]
        )[:, 1]

        piB[permB[B_test_index_list[i]]] = clf.predict_proba(
            target_X[permB[B_test_index_list[i]]]
        )[:, 1]

    if draw_calibration:
        plot_calibration(piA, piB, save_dir=save_calibration_png)

    # Calculate density ratios (Importance Weights)
    alpha = target_X.shape[0] / (source_X.shape[0] + target_X.shape[0])

    wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
    wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))

    # Normalize weights
    wA /= np.sum(wA)
    wB /= np.sum(wB)

    # Compute empirical losses
    pred_source = model.predict(source_X)
    pred_target = model.predict(target_X)

    loss_source = np.abs(pred_source - source_y)
    loss_target = np.abs(pred_target - target_y)

    # Empirical risk limits
    errorA = np.mean(loss_source)
    errorB = np.mean(loss_target)

    # Importance-weighted risk estimators (approximating target integration)
    sx_A = np.dot(wA, loss_source)
    sx_B = np.dot(wB, loss_target)

    return errorA, errorB, sx_A, sx_B


def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None,
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir='.', balance=False):
    """Generates reliability diagrams for the domain discrimination classifier."""
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(3):
        ax[i].set_box_aspect(1)
        ax[i].set_xlim(0, 1)
    fig.suptitle("Calibration: {}".format(name), fontsize="x-large")

    if p_weights is None: p_weights = np.ones_like(prop_p)
    if q_weights is None: q_weights = np.ones_like(prop_q)

    p_sample_weights = p_weights.copy()
    q_sample_weights = q_weights.copy()
    if balance:
        p_sample_weights = p_sample_weights / p_sample_weights.sum()
        q_sample_weights = q_sample_weights / q_sample_weights.sum()

    conf_scores, bin_edges = np.histogram(np.concatenate([1 - prop_p, prop_q]), bins=nbins, density=True,
                                          weights=np.concatenate([p_sample_weights,
                                                                  q_sample_weights]),
                                          range=(0, 1))
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2

    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
    # print(bin_mids)
    # print(nanmask * conf_scores / (conf_scores + conf_scores[::-1]))
    ax[0].plot(bin_mids, nanmask * conf_scores / (conf_scores + conf_scores[::-1]), color='green')
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Proportion correct')
    ax[0].set_xlabel('Predicted probability')
    ax[0].set_title('Prop calibration: combined')

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_p]), bins=nbins, weights=p_sample_weights,
                                          range=(0, 1))
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[1].plot(bin_mids, conf_scores, color='green')
    ax[1].set_title('Density: P')
    ax[1].set_xlabel('Predicted probability of Q')
    ax[1].set_ylim(bottom=0)

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_q]), bins=nbins, weights=q_sample_weights,
                                          range=(0, 1))
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[2].plot(bin_mids, conf_scores, color='green')
    ax[2].set_title('Density: Q')
    ax[2].set_xlabel('Predicted probability of Q')
    ax[2].set_ylim(bottom=0)

    fig.tight_layout()


# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# REGION ANALYSIS 1: TARGET vs SYNTHETIC COVARIATE GEOMETRY

# Identifies disjoint sub-regions in the feature space causing maximum performance divergence
synth_df = synthetic_dataset.copy()
target_df = holdout_test_df.copy()

drop_cols = ['Weekly_Sales']

X_syn = synth_df.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL) - Map to contiguous representation for Tree Estimators
combined = pd.concat([X_syn, X_target], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_nodes)

X_syn_enc = combined_encoded.iloc[:len(X_syn)]
X_target_enc = combined_encoded.iloc[len(X_syn):]

feature_names = X_syn_enc.columns.tolist()

X_source_full = X_syn_enc.values
X_target = X_target_enc.values

y_source_full = synth_df['Weekly_Sales'].values
y_target = target_df['Weekly_Sales'].values

np.random.seed(42)
perm = np.random.permutation(len(X_source_full))
train_idx = perm[:int(0.7 * len(X_source_full))]
id_test_idx = perm[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]

X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"Features: {len(feature_names)}")
np.random.seed(42)
perm = np.random.permutation(len(X_target))
train_target = perm[:int(0.7 * len(X_target))]
id_test_target = perm[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]

X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")
print(f"Features: {len(feature_names)}")


model_disde = LGBMRegressor(**lgb_params)
model_disde.fit(X_source_train, y_source_train)

print("\n>>> Running Degradation Decomposition...")
reg_domain_classifier = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    use_label_encoder=False,
    eval_metric='logloss'
)

# Decompose model error to isolate covariate structural vulnerabilities
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_full, y_source_full,
    X_target, y_target,
    model_disde,
    data_sum=20000,
    K=8,
    domain_classifier=reg_domain_classifier,
    draw_calibration=True,
)

print("-" * 45)
print(f"Total Performance Degradation (p2p - q2q): {p2p - q2q:.4f}")
print(f"Proportion of Y|X-shift:                  {(p2s - s2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift:                  {(p2p - p2s + s2q - q2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift Q side:                  {(p2p - p2s) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift P side:                  {(s2q - q2q) / (p2p - q2q):.4f}")
print(f"p2p: accuracy on source domain is:{p2p:.4f}")
print(f"q2q: accuracy on target domain is:{q2q:.4f}")
print(f"p2s: expected Source accuracy on Sx (Reweighted Source Accuracy)  : {p2s:.4f}")
print(f"s2q: expected Target accuracy on Sx (Reweighted Target Accuracy)  : {s2q:.4f}")
print(f"Y|X shift is p2s-s2q :{(p2s - s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p - p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q - q2q):.4f}")
print("-" * 45)

# Project density ratios to induce balanced covariate alignment
wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

source_model = LGBMRegressor(**lgb_params)
source_model.fit(X_source_train, y_source_train, sample_weight=wA)

target_model = LGBMRegressor(**lgb_params)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)

pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

# Calculate model disagreement indicative of epistemic uncertainty
new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")

# Model trained on Synthetic data
print(
    f"Model trained on Synthetic data: MAE on Synthetic data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(
    f"Model trained on Synthetic data MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(
    f"Target model MAE on Synthetic data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(
    f"Target model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

# Fit regression tree on disagreements to extract explicit interpretable risk regions
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")

plt.savefig("TARGET - SYNTHETIC TYPE REGION ANALYSIS.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# REGION ANALYSIS 2: REAL SOURCE vs SYNTHETIC COVARIATE GEOMETRY

real_df = source_df.copy()
target_df = synthetic_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc = combined_encoded.iloc[len(X):]

feature_names = X_enc.columns.tolist()

X_source_full = X_enc.values
X_target = X_target_enc.values

y_source_full = real_df['Weekly_Sales'].values
y_target = target_df['Weekly_Sales'].values

np.random.seed(42)
perm = np.random.permutation(len(X_source_full))
train_idx = perm[:int(0.7 * len(X_source_full))]
id_test_idx = perm[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]

X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"Features: {len(feature_names)}")
np.random.seed(42)
perm = np.random.permutation(len(X_target))
train_target = perm[:int(0.7 * len(X_target))]
id_test_target = perm[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]

X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")
print(f"Features: {len(feature_names)}")


model_disde = LGBMRegressor(**lgb_params)
model_disde.fit(X_source_train, y_source_train)

print("\n>>> Running Degradation Decomposition...")
reg_domain_classifier = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    use_label_encoder=False,
    eval_metric='logloss'
)

# Analyze structural divergence imposed by the NP-SEM synthetic generating process
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_full, y_source_full,
    X_target, y_target,
    model_disde,
    data_sum=20000,
    K=8,
    domain_classifier=reg_domain_classifier,
    draw_calibration=True,
)

print("-" * 45)
print(f"Total Performance Degradation (p2p - q2q): {p2p - q2q:.4f}")
print(f"Proportion of Y|X-shift:                  {(p2s - s2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift:                  {(p2p - p2s + s2q - q2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift Q side:                  {(p2p - p2s) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift P side:                  {(s2q - q2q) / (p2p - q2q):.4f}")
print(f"p2p: accuracy on source domain is:{p2p:.4f}")
print(f"q2q: accuracy on target domain is:{q2q:.4f}")
print(f"p2s: expected Source accuracy on Sx (Reweighted Source Accuracy)  : {p2s:.4f}")
print(f"s2q: expected Target accuracy on Sx (Reweighted Target Accuracy)  : {s2q:.4f}")
print(f"Y|X shift is p2s-s2q :{(p2s - s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p - p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q - q2q):.4f}")
print("-" * 45)
wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

source_model = LGBMRegressor(**lgb_params)

source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model = LGBMRegressor(**lgb_params)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")

# Model trained on real training data
print(
    f"Model trained on real training data MAE on ID real data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(
    f"Model trained on real training data MAE on OOD synthetic data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(
    f"Model trained on Synthetic data MAE on ID real data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(
    f"Model trained on Synthetic data MAE on OOD synthetic data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("TYPE REAL training - SYNTHETIC  REGION ANALYSIS.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# REGION ANALYSIS 3: REAL SOURCE vs HYBRID COVARIATE GEOMETRY

real_df = source_df.copy()
target_df = hybrid_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc = combined_encoded.iloc[len(X):]

feature_names = X_enc.columns.tolist()

X_source_full = X_enc.values
X_target = X_target_enc.values

y_source_full = real_df['Weekly_Sales'].values
y_target = target_df['Weekly_Sales'].values

np.random.seed(42)
perm = np.random.permutation(len(X_source_full))
train_idx = perm[:int(0.7 * len(X_source_full))]
id_test_idx = perm[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]

X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"Features: {len(feature_names)}")
np.random.seed(42)
perm = np.random.permutation(len(X_target))
train_target = perm[:int(0.7 * len(X_target))]
id_test_target = perm[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]

X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")
print(f"Features: {len(feature_names)}")


model_disde = LGBMRegressor(**lgb_params)
model_disde.fit(X_source_train, y_source_train)

print("\n>>> Running Degradation Decomposition...")
reg_domain_classifier = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    use_label_encoder=False,
    eval_metric='logloss'
)

# Quantify regularization provided by the hybrid support expansion strategy
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_full, y_source_full,
    X_target, y_target,
    model_disde,
    data_sum=20000,
    K=8,
    domain_classifier=reg_domain_classifier,
    draw_calibration=True,
)

print("-" * 45)
print(f"Total Performance Degradation (p2p - q2q): {p2p - q2q:.4f}")
print(f"Proportion of Y|X-shift:                  {(p2s - s2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift:                  {(p2p - p2s + s2q - q2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift Q side:                  {(p2p - p2s) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift P side:                  {(s2q - q2q) / (p2p - q2q):.4f}")
print(f"p2p: accuracy on source domain is:{p2p:.4f}")
print(f"q2q: accuracy on target domain is:{q2q:.4f}")
print(f"p2s: expected Source accuracy on Sx (Reweighted Source Accuracy)  : {p2s:.4f}")
print(f"s2q: expected Target accuracy on Sx (Reweighted Target Accuracy)  : {s2q:.4f}")
print(f"Y|X shift is p2s-s2q :{(p2s - s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p - p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q - q2q):.4f}")
print("-" * 45)
wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

source_model = LGBMRegressor(**lgb_params)

source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model = LGBMRegressor(**lgb_params)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")

# Source model
print(
    f"Model trained on real data: MAE on real data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(
    f"Model trained on real data: MAE on hybrid data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(
    f"Hybrid model:  MAE on real data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(
    f"Hybrid model:  MAE on hybrid data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("ALL_REAL_HYBRID_TYPE_tree.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# REGION ANALYSIS 4: HYBRID vs TARGET COVARIATE GEOMETRY

hybrid_df = hybrid_dataset.copy()
target_df = holdout_test_df.copy()

drop_cols = ['Weekly_Sales']

X = hybrid_dataset.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc = combined_encoded.iloc[len(X):]

feature_names = X_enc.columns.tolist()

X_source_full = X_enc.values
X_target = X_target_enc.values

y_source_full = hybrid_df['Weekly_Sales'].values
y_target = target_df['Weekly_Sales'].values

np.random.seed(42)
perm = np.random.permutation(len(X_source_full))
train_idx = perm[:int(0.7 * len(X_source_full))]
id_test_idx = perm[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]

X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"Features: {len(feature_names)}")
np.random.seed(42)
perm = np.random.permutation(len(X_target))
train_target = perm[:int(0.7 * len(X_target))]
id_test_target = perm[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]

X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")
print(f"Features: {len(feature_names)}")
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")
model_disde = LGBMRegressor(**lgb_params)
model_disde.fit(X_source_train, y_source_train)

print("\n>>> Running Degradation Decomposition...")
reg_domain_classifier = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    use_label_encoder=False,
    eval_metric='logloss'
)

# Assess final generalizability improvements across domains
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_full, y_source_full,
    X_target, y_target,
    model_disde,
    data_sum=20000,
    K=8,
    domain_classifier=reg_domain_classifier,
    draw_calibration=True,
)

print("-" * 45)
print(f"Total Performance Degradation (p2p - q2q): {p2p - q2q:.4f}")
print(f"Proportion of Y|X-shift:                  {(p2s - s2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift:                  {(p2p - p2s + s2q - q2q) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift Q side:                  {(p2p - p2s) / (p2p - q2q):.4f}")
print(f"Proportion of X-shift P side:                  {(s2q - q2q) / (p2p - q2q):.4f}")
print(f"p2p: accuracy on source domain is:{p2p:.4f}")
print(f"q2q: accuracy on target domain is:{q2q:.4f}")
print(f"p2s: expected Source accuracy on Sx (Reweighted Source Accuracy)  : {p2s:.4f}")
print(f"s2q: expected Target accuracy on Sx (Reweighted Target Accuracy)  : {s2q:.4f}")
print(f"Y|X shift is p2s-s2q :{(p2s - s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p - p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q - q2q):.4f}")
print("-" * 45)

wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

source_model = LGBMRegressor(**lgb_params)

source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model = LGBMRegressor(**lgb_params)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")

# Source model
print(
    f"Hybrid model MAE on hybrid data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(
    f"Hybrid model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(
    f"Target model MAE on hybrid data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(
    f"Target model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

# Produce final localized risk estimates
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("hybrid_target_TYPE.png")
plt.show()