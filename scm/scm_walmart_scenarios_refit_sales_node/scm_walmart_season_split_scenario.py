#%%
#!/usr/bin/env python3

"""
Implementation of a Structural Causal Model (SCM) for Domain Adaptation under Covariate and Concept Shift.
This script formulates a causal graph for retail sales forecasting,
fits an SCM using the DoWhy library, performs targeted interventions (do-calculus)
to simulate domain shifts across seasonal distributions, and evaluates
out-of-distribution (OOD) performance degradation and empirical risk bounds.
"""

import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

# Suppress warnings to maintain clean stdout during iterative execution
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
from dowhy.gcm.ml import SklearnClassificationModel, SklearnRegressionModel
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import yaml
import sys
sys.path.append("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm")
from custom_mape import custom_retail_mape

# Load hyperparameter configurations from external YAML definition
with open("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

# Define experimental domain configurations.
# Source distribution P(X,Y) and Target distribution Q(X,Y) are delineated by the 'Season' variable.
CONFIG = {
    'source_domain': 'Normal  (Season = 1,2,4)',
    'target_domain': 'Summer  (Season = 3)',
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_Season',
}

# Create checkpoint directory for SCM state preservation
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)


def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Persists the fitted Structural Causal Model (SCM) and topological metadata to disk.
    This facilitates reproducibility and ablation studies without computationally expensive refitting.

    Args:
        scm: Fitted dowhy.gcm.StructuralCausalModel instance.
        feature_graph: networkx DiGraph representing the causal structure.
        parents (list): Direct causal ancestors of the target variable (Weekly_Sales).
        metrics (dict): Dictionary mapping validation metrics.
        stage_name (str): Identifier for the current experimental phase.
    """
    try:
        checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Serialize SCM architecture and learned functional causal mechanisms
        scm_path = checkpoint_dir / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

        # Serialize architectural graph metadata
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
    """
    Deserializes the SCM and corresponding graph metadata.

    Args:
        scm_path (str): Path to SCM pickle artifact.
        metadata_path (str): Path to metadata pickle artifact.

    Returns:
        tuple: (scm, metadata)
    """
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


#%%

print("PART 1: LOADING DATA & SETUP")

# Data ingestion and initial sanitization
try:
    df = pd.read_csv("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv")
    # Discard non-causal structural indices and potential leakage variables
    df = df.drop(
        columns=["Date","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0) # Baseline imputation strategy

    print(f"Dataset shape: {df.shape}")

except FileNotFoundError:
    print("ERROR: Data file '/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv' not found!")
    raise


print("\n" + "=" * 70)
print("PART 2: DOMAIN SHIFT SPLIT")

# Define OOD Split Formulation
# Source Domain (P): Seasons 1, 2, and 4. Target Domain (Q): Season 3.
# This reflects an environmental covariate shift driven by temporal dynamics.
df['Season'] = df['Season'].astype(int)

source_df = df[df['Season'].isin([1, 2, 4])].copy()
target_df = df[df['Season'].isin([3])].copy()

# Partition the Target Domain into an Adaptation set (for supervised causal mechanism updates)
# and a strict Holdout set for unbiased evaluation of domain generalization.
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source domain: {CONFIG['source_domain']}")
print(f"Source shape: {source_df.shape}")
print(f"\nTarget domain: {CONFIG['target_domain']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Rigorous assertions to guarantee absolute domain separation and prevent observational leakage
assert (source_df['Season'].isin([1, 2,4])).all(), "Train set contains Season = 3!"
assert (adaptation_df['Season'] ==3).all(), "Adaptation set contains Season = 1 2 4!"
assert (holdout_test_df['Season'] == 3).all(), "Holdout set contains Season = 1 2 4!"


#%%

print("PART 3: PREPROCESSING")

try:
    # Explicit type.sh casting separates covariates into discrete (categorical/binary)
    # and continuous spaces to properly govern SCM mechanism assignment logic.
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Transform boolean matrices into identifiable discrete categorical states
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

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

#%%


print("PART 4: CAUSAL GRAPH ANALYSIS")

# Formulate the hypothesized causal graph as a Directed Acyclic Graph (DAG)
# combining expert domain knowledge into macroscopic group nodes.
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
    "Season": ["Season_sin","Season", "Season_cos"],
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

# Define macroscopic causal edges representing structural causal dependencies
causal_edges_groups = [
    ("Date_Features", "Season"),
    ("Date_Features", "Holidays"),
    ("Date_Features", "Sales"),
    ("Date_Features", "Economy"),
    ("Season", "Weather"),
   # ("Season", "Sales"), # Assumed conditionally independent given Weather/Holidays
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

# Instantiate the macroscopic DAG
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

def get_all_descendants(graph, source_node):
    """
    Identifies the transitive closure of descendants from a given source node.
    Critical for determining which downstream mechanisms are affected under interventional scenarios.
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

# Isolate all feature groups subjected to causal influence by the 'Season' domain variable.
affected_groups = get_all_descendants(macro_graph, "Season")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Season: {set(macro_graph.successors('Season'))}")
print(f"All affected groups (transitive): {affected_groups}")


print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")

# Unroll the macroscopic group graph into a high-dimensional feature-level DAG
feature_graph = nx.DiGraph()
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

print(f"Direct children of Season: {set(macro_graph.successors('Season'))}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")
for edge in feature_graph.edges:
    print(edge)

print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Initialize the Structural Causal Model given the established prior topological structure
scm = gcm.StructuralCausalModel(feature_graph)

def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assigns localized Functional Causal Models (FCMs) to each node within the DAG.
    - Root nodes are modeled via empirical marginal distributions.
    - Categorical nodes are fitted using non-linear Classifiers (LightGBM).
    - Continuous nodes rely on Additive Noise Models (ANM) with LightGBM regressors.
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        if len(parents) == 0:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif node in classifier_nodes:
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(LGBMClassifier(**lgb_params))
                )
            )
        else:
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    SklearnRegressionModel(LGBMRegressor(**lgb_params))
                )
            )

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on training data...")
    # Estimate causal mechanisms strictly based on observational Source Domain data
    gcm.fit(scm, source_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'initial_fit_complete'},
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save checkpoint: {e}")

#%%


print("PART 8: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Determine nodes necessitating mechanism refitting due to structural shift.
# We traverse topologically and isolate nodes strictly in the descendant causal path of the domain variable.
ordered_nodes = list(nx.topological_sort(feature_graph))

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
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            # Perform targeted structural domain adaptation using the limited Adaptation sample bounds
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n✓ Concept shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

#%%

print("PART 9: SYNTHETIC DATA GENERATION & ML TRAINING")


print("\nExecuting Intervention: Changing days to Season = 3 (Season = 3)...")

def season_intervention_fn(x):
    """Defines the deterministic hard intervention do(Season = 3)"""
    return 3


num_synthetic_samples = len(target_df)

# Propagate the do() intervention to generate synthetic counterfactual trajectories
synthetic_dataset = gcm.interventional_samples(
    scm,
    interventions={'Season': season_intervention_fn},
    num_samples_to_draw=num_synthetic_samples
)
print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")

# Truncate theoretically non-physical trajectories resulting from additive noise over-dispersion
cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

synthetic_dataset[cols] = synthetic_dataset[cols].clip(lower=0)

# Re-align generated categorical features with observational space dictionaries
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

print(f"Synthetic Dataset Shape: {synthetic_dataset.shape}")

print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate predictive performance by training an empirical risk minimizer exclusively on synthesized samples
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)
#%%

# Statistical Validation Checks on generated vs actual density characteristics
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
#%%

print("PART 10: EVALUATION ON TEST SET")


print("\nEvaluation on Test Set ...")

X_test = holdout_test_df.drop(columns=['Weekly_Sales'])

# Project test variables strictly to synthetic alignment to mitigate feature space mismatch
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

y_pred = ml_model_syn.predict(X_test)

# Report point-estimate evaluation metrics demonstrating generalization on the OOD Target domain
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")



#%%
# Construct Hybrid Corpus to induce domain robustness via SCM Data Augmentation
print("\nHybrid Dataset (Source + Synthetic Target)...")


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


ml_model_hbr.fit(X_hybrid, y_hybrid)
#%%

# Verify invariant distributional moments across the combined corpora
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


#%%
# Evaluate Hybrid Model predictive efficacy against pure holdout distributions
print("\nEvaluation on Test Set ...")

X_test = holdout_test_df.drop(columns=['Weekly_Sales'])





X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

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

#%%
# Re-fit the generative mechanism mapping locally onto the synthesized Hybrid distribution
fit_causal_model_of_target(scm, "Weekly_Sales", hybrid_dataset)
def evaluate_sales_on_holdout(model_scm, eval_df, label):
    """
    Evaluates isolated structural mechanism capacity strictly on validation boundaries.
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


#%%
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


def degradation_decomp_regression(
    source_X,
    source_y,
    target_X_raw,
    target_y_raw,
    model,
    data_sum=20000,
    K=8,
    domain_classifier=None,draw_calibration=False, save_calibration_png='calibration.png'
):
    """
    Implements theoretical decomposition of predictive error degradation across domains.
    Disentangles marginal covariate shift (X-shift) and conditional concept shift (Y|X-shift)
    utilizing Propensity Score formulation and Inverse Probability Weighting (IPW), as formalized
    in WhyShift and rigorous Out-of-Distribution (OOD) analysis literature.
    """

    perm1 = np.random.permutation(target_X_raw.shape[0])
    target_X = target_X_raw[perm1[:data_sum], :]
    target_y = target_y_raw[perm1[:data_sum]]


    piA = np.zeros(source_X.shape[0])
    piB = np.zeros(target_X.shape[0])

    permA = np.random.permutation(source_X.shape[0])
    permB = np.random.permutation(target_X.shape[0])

    kf = KFold(n_splits=K, shuffle=False)

    A_train_index_list, A_test_index_list = [], []
    B_train_index_list, B_test_index_list = [], []

    for train_idx, test_idx in kf.split(source_X):
        A_train_index_list.append(train_idx)
        A_test_index_list.append(test_idx)

    for train_idx, test_idx in kf.split(target_X):
        B_train_index_list.append(train_idx)
        B_test_index_list.append(test_idx)

    # Train structural domain classifiers to approximate the underlying Density Ratio
    for i in range(K):
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

        piA[permA[A_test_index_list[i]]] = clf.predict_proba(
            source_X[permA[A_test_index_list[i]]]
        )[:, 1]

        piB[permB[B_test_index_list[i]]] = clf.predict_proba(
            target_X[permB[B_test_index_list[i]]]
        )[:, 1]

    if draw_calibration:
        plot_calibration(piA, piB, save_dir=save_calibration_png)
    alpha = target_X.shape[0] / (source_X.shape[0] + target_X.shape[0])

    # Compute Inverse Probability Weighting (IPW) scalars mapped to the shared region
    wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
    wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))

    wA /= np.sum(wA)
    wB /= np.sum(wB)


    pred_source = model.predict(source_X)
    pred_target = model.predict(target_X)

    loss_source = np.abs(pred_source - source_y)
    loss_target = np.abs(pred_target - target_y)

    # Compute Empirical and Reweighted Structural Risks
    errorA = np.mean(loss_source)
    errorB = np.mean(loss_target)

    # Compute weighted risks indicative of domain divergence
    sx_A = np.dot(wA, loss_source)
    sx_B = np.dot(wB, loss_target)

    return errorA,errorB,sx_A,sx_B

def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None,
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir='.', balance=False):
    """
    Generates reliability diagrams mapping density ratios from domain classification bounds.
    Validates adherence to the Overlapping Support assumption critical for IPW.
    """
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    for i in range(3):
        ax[i].set_box_aspect(1)
        ax[i].set_xlim(0,1)
    fig.suptitle("Calibration: {}".format(name), fontsize="x-large")

    if p_weights is None: p_weights = np.ones_like(prop_p)
    if q_weights is None: q_weights = np.ones_like(prop_q)

    p_sample_weights = p_weights.copy()
    q_sample_weights = q_weights.copy()
    if balance:
        p_sample_weights = p_sample_weights / p_sample_weights.sum()
        q_sample_weights = q_sample_weights / q_sample_weights.sum()

    conf_scores, bin_edges = np.histogram(np.concatenate([1-prop_p, prop_q]),bins=nbins, density=True,
                                          weights=np.concatenate([p_sample_weights,
                                                                 q_sample_weights]),
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2

    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
    # print(bin_mids)
    # print(nanmask * conf_scores / (conf_scores + conf_scores[::-1]))
    ax[0].plot(bin_mids, nanmask * conf_scores / (conf_scores + conf_scores[::-1]), color='green')
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Proportion correct')
    ax[0].set_xlabel('Predicted probability')
    ax[0].set_title('Prop calibration: combined')

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_p]),bins=nbins, weights=p_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[1].plot(bin_mids, conf_scores, color='green')
    ax[1].set_title('Density: P')
    ax[1].set_xlabel('Predicted probability of Q')
    ax[1].set_ylim(bottom=0)

    conf_scores, bin_edges = np.histogram(np.concatenate([prop_q]),bins=nbins, weights=q_sample_weights,
                                         range=(0,1))
    bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
    nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)

    ax[2].plot(bin_mids, conf_scores, color='green')
    ax[2].set_title('Density: Q')
    ax[2].set_xlabel('Predicted probability of Q')
    ax[2].set_ylim(bottom=0)

    fig.tight_layout()

#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight

# ----------------------------------------------------------------------
# OOD REGION ANALYSIS: TARGET VS SYNTHETIC DOMAIN
# Analyzes the structural discrepancy between purely synthesized interventional
# targets and true-world real target observations.
# ----------------------------------------------------------------------

synth_df = synthetic_dataset.copy()
target_df = holdout_test_df.copy()

drop_cols = ['Weekly_Sales']

X_syn = synth_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# Apply One-Hot Encoding to enforce unified affine topological spaces
combined = pd.concat([X_syn, X_target], axis=0)
combined_encoded = pd.get_dummies(combined,columns=categorical_nodes)

X_syn_enc = combined_encoded.iloc[:len(X_syn)]
X_target_enc  = combined_encoded.iloc[len(X_syn):]

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

p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_full, y_source_full,
    X_target, y_target,
    model_disde,
    data_sum=20000,
    K=8,
    domain_classifier=reg_domain_classifier,
    draw_calibration=True,
)

# Formal shift analysis outputs parsing Covariate (X) vs Conditional (Y|X) bounds
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
print(f"Y|X shift is p2s-s2q :{(p2s-s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p-p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q-q2q):.4f}")
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

# Model trained on Synthetic data
print(f"Model trained on Synthetic data: MAE on Synthetic data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Model trained on Synthetic data MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on Synthetic data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

# Train Surrogate Interpretable Decision Tree to delineate high-disagreement OOD regions
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")

plt.savefig("TARGET - SYNTHETIC SEASON REGION ANALYSIS.png")
plt.show()


#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight

# ----------------------------------------------------------------------
# OOD REGION ANALYSIS: REAL SOURCE VS SYNTHETIC
# Probes the discrepancy boundaries between factual historical source data
# and generated counterfactual manifolds post-intervention.
# ----------------------------------------------------------------------

real_df = source_df.copy()
target_df = synthetic_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# Apply One-Hot Encoding strictly
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined,columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc  = combined_encoded.iloc[len(X):]

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
print(f"Y|X shift is p2s-s2q :{(p2s-s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p-p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q-q2q):.4f}")
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
print(f"Model trained on real training data MAE on ID real data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Model trained on real training data MAE on OOD synthetic data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Model trained on Synthetic data MAE on ID real data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Model trained on Synthetic data MAE on OOD synthetic data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("SEASON REAL training - SYNTHETIC  REGION ANALYSIS.png")
plt.show()


#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight

# ----------------------------------------------------------------------
# OOD REGION ANALYSIS: REAL SOURCE VS HYBRID
# Investigates whether utilizing a fused Hybrid data proxy functions as an
# effective regularizer against domain shift variance.
# ----------------------------------------------------------------------

real_df = source_df.copy()
target_df = hybrid_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# Apply One-Hot Encoding (CRITICAL for matrix alignment)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined,columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc  = combined_encoded.iloc[len(X):]

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
print(f"Y|X shift is p2s-s2q :{(p2s-s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p-p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q-q2q):.4f}")
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
print(f"Model trained on real training data: MAE on real data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Model trained on real training data: MAE on hybrid data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Hybrid model:  MAE on real data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Hybrid model:  MAE on hybrid data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("ALL_REAL_HYBRID_SEASON_tree.png")
plt.show()


#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight

# ----------------------------------------------------------------------
# OOD REGION ANALYSIS: HYBRID VS TARGET
# Evaluates whether the Hybrid data geometry authentically envelops the
# true underlying Out-of-Distribution holdout target structure.
# ----------------------------------------------------------------------

hybrid_df = hybrid_dataset.copy()
target_df = holdout_test_df.copy()

drop_cols = ['Weekly_Sales']

X = hybrid_dataset.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# Apply One-Hot Encoding (CRITICAL)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined,columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc  = combined_encoded.iloc[len(X):]

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
print(f"Y|X shift is p2s-s2q :{(p2s-s2q):.4f}")
print(f"X shift (P) is p2p-p2s :{(p2p-p2s):.4f}")
print(f"X shift (Q) is s2q-q2q :{(s2q-q2q):.4f}")
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
print(f"Hybrid model MAE on hybrid data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Hybrid model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on hybrid data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("hybrid_target_SEASON.png")
plt.show()