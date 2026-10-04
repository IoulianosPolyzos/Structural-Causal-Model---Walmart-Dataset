# %%
# !/usr/bin/env python3


# Methodological Setup and Library Importation

# Suppress routine warnings to maintain clean execution outputs during model training.
import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Import core machine learning and causal inference libraries.
# LightGBM is utilized as the primary functional mechanism within the SCM.
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


# Configuration and Hyperparameter Initialization

# Load pre-defined hyperparameter configurations for the LightGBM models.
with open("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/scm/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

# Define experimental domain configurations and directory paths for state preservation.
CONFIG = {
    'source_domain': 'Normal Days (IsChristmas_Season=0)',
    'target_domain': 'Holiday Days (Is_Christmas_Season=1)',
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_Is_Christmas_Season',
}

# Create checkpoint directory for intermediate model persistence
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)



# State Preservation Mechanisms for the Structural Causal Model (SCM)

def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Serializes and persists the fitted Structural Causal Model and its topological
    metadata to disk for reproducible experimentation and subsequent analyses.

    Args:
        scm: Fitted StructuralCausalModel framework.
        feature_graph: networkx DiGraph representing the causal topology.
        parents (list): Direct causal ancestors of the target variable (Weekly_Sales).
        metrics (dict): Dictionary capturing empirical performance metrics.
        stage_name (str): Identifier for the experimental lifecycle stage.
    """
    try:
        checkpoint_dir = Path(CONFIG['checkpoint_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Serialize the Structural Causal Model object
        scm_path = checkpoint_dir / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

        # Serialize the topological and experimental metadata
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
    Restores the SCM and its topological metadata from secondary storage.

    Args:
        scm_path (str): Filepath to the serialized SCM.
        metadata_path (str): Filepath to the serialized metadata.

    Returns:
        tuple: (restored_scm, restored_metadata)
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


# %%


# Data Ingestion and Preliminary Processing

print("PART 1: LOADING DATA & SETUP")

try:
    # Ingest the comprehensive Walmart dataset and eliminate highly collinear
    # or strictly deterministic temporal proxies to constrain the causal graph complexity.
    df = pd.read_csv("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv")
    df = df.drop(
        columns=["Date", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    # Perform naive median/zero imputation for structural missingness
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Is_Christmas_Season distribution:")
    print(f"  Normal Days (0): {(df['Is_Christmas_Season'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['Is_Christmas_Season'] == 1).sum()} records")

except FileNotFoundError:
    print(
        "ERROR: Data file '/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv' not found!")
    raise


# Domain Stratification: Segregation of Source and Target Distributions

print("\n" + "=" * 70)
print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs CHRISTMAS SEASON)")

# Ensure binary intervention variable maintains discrete scalar typing
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

# Extract Source Domain: Observational data representing standard operational days
source_df = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)

# Extract Target Domain: Out-of-distribution (OOD) data representing seasonal events
target_df = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)

# Partition the Target Domain into an Adaptation set (for covariate shift recalibration)
# and a strictly disjoint Holdout set (for unbiased empirical validation).
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

# Verify integrity of the inter-domain boundaries
assert (source_df['Is_Christmas_Season'] == 0).all(), "Train set contains non-zero Is_Christmas_Season!"
assert (adaptation_df['Is_Christmas_Season'] == 1).all(), "Adaptation set contains non-CHRISTMAS days!"
assert (holdout_test_df['Is_Christmas_Season'] == 1).all(), "Holdout set contains non-CHRISTMAS days!"
print("✓ Domain separation verified: No overlap between Normal/CHRISTMAS days")

# %%


# Feature Engineering and Typological Alignment

print("PART 3: PREPROCESSING")

try:
    # Isolate non-ordinal categorical features to prevent improper numerical casting
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Cast continuous covariates strictly to standard numeric datatypes
    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Standardize string representations for discrete binary nodes
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start",
                    "Is_Month_End"]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Standardize string representations for multinomial categorical nodes
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


# Structural Causal Graph Definition (Macro and Micro/Feature Levels)

print("PART 4: CAUSAL GRAPH ANALYSIS")

# Define macro-level latent conceptual categories to guide the DAG generation
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

# Define hypothesized directed dependencies across macro-level categories
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

# Instantiate macro-level Directed Acyclic Graph (DAG)
macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)


def get_all_descendants(graph, source_node):
    """Recursively identify all downstream dependents within the specified DAG structure."""
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


# Isolate causal descendants subject to the proposed interventional shift
affected_groups = get_all_descendants(macro_graph, "Holidays")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Is_Christmas_Season: {set(macro_graph.successors('Holidays'))}")
print(f"All affected groups (transitive): {affected_groups}")

print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")

# Translate macro-level topological relationships to a granular, feature-specific DAG
feature_graph = nx.DiGraph()
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Enforce explicit d-separation: Ensure the causal effect of Christmas Season on Sales
# is strictly mediated through defined intermediate covariates, removing the direct edge.
feature_graph.remove_edge('Is_Christmas_Season', 'Weekly_Sales')
for edge in feature_graph.edges:
    print(edge)

print("Is_Christmas_Season successors : ", list(feature_graph.successors("Is_Christmas_Season")))
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")


# Empirical Estimation of the Structural Causal Model (Source Domain)

print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Initialize DoWhy Graphical Causal Model (GCM) instance
scm = gcm.StructuralCausalModel(feature_graph)


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assign appropriate functional causal mechanisms to each node based on
    topological degree and typological constraints (e.g., discrete vs continuous).
    """
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))

        if len(parents) == 0:
            # Exogenous root nodes utilize their empirical marginal distributions
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif node in classifier_nodes:
            # Categorical endogenous nodes are modeled via Classification mechanisms
            scm.set_causal_mechanism(
                node,
                gcm.ClassifierFCM(
                    SklearnClassificationModel(LGBMClassifier(**lgb_params))
                )
            )
        else:
            # Continuous endogenous nodes are modeled via Additive Noise Models (ANM)
            scm.set_causal_mechanism(
                node,
                gcm.AdditiveNoiseModel(
                    SklearnRegressionModel(LGBMRegressor(**lgb_params))
                )
            )


try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on training data...")
    # Estimate causal mechanisms utilizing the Source Domain (Normal Days)
    gcm.fit(scm, source_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

try:
    # Persist the baseline empirical SCM trained on Source domain
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'initial_fit_complete'},
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save checkpoint: {e}")

# %%


# Supervised Domain Adaptation (Concept & Covariate Shift Correction)

print("PART 8: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Derive the topologically sorted order of the SCM DAG
ordered_nodes = list(nx.topological_sort(feature_graph))

# Isolate structurally dependent nodes necessitating distributional calibration
# due to the targeted intervention domain shift.
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
    # Re-estimate conditional mechanisms over the Adaptation Subset to correct for concept shift
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


# Counterfactual Intervention and Synthetic Target Data Generation

print("PART 9: SYNTHETIC DATA GENERATION & ML TRAINING")

print("\nExecuting Intervention: Changing days to Christmas (Is_Christmas_Season = 1)...")


def christmas_intervention_fn(x):
    """Define deterministic structural intervention (do-operator)."""
    return 1


num_synthetic_samples = len(target_df)

# Perform ancestral sampling through the adapted SCM to simulate Target Domain distributions
synthetic_dataset = gcm.interventional_samples(
    scm,
    interventions={'Is_Christmas_Season': christmas_intervention_fn},
    num_samples_to_draw=num_synthetic_samples
)
print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")

# Post-processing filtration: Constrain inherently non-negative variables
# to structurally valid support boundaries.
cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

synthetic_dataset[cols] = synthetic_dataset[cols].clip(lower=0)

# Re-align categorical factor levels matching the global vocabulary
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

print(f"Synthetic Dataset Shape: {synthetic_dataset.shape}")

print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)
# %%

# Distributional verification statistics prior to model inference
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


# Empirical Validation on the OOD Target Set

print("PART 10: EVALUATION ON TEST SET")

print("\nEvaluation on Test Set ...")

# Format out-of-distribution Target set features for unbiased evaluation
X_test = holdout_test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

# Generate predictions utilizing the SCM-synthesized target model
y_pred = ml_model_syn.predict(X_test)

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
    'Christmas Season': holdout_test_df['Is_Christmas_Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%


# Formulation and Training of the Hybrid Model Architecture

print("\nHybrid Dataset (Source + Synthetic Target)...")

# Construct composite manifold amalgamating observational Source data
# with generated Interventional counterfactual Target data.
train_aligned = source_df[synthetic_dataset.columns].copy()

hybrid_dataset = pd.concat([train_aligned, synthetic_dataset], ignore_index=True)

# Maintain factor integrity across concatenated feature space
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    hybrid_dataset[col] = pd.Categorical(hybrid_dataset[col].astype(str), categories=all_categories)

print(f"Hybrid Dataset Shape: {hybrid_dataset.shape} (Real: {len(train_aligned)}, Synthetic: {len(synthetic_dataset)})")

X_hybrid = hybrid_dataset.drop(columns=['Weekly_Sales'])
y_hybrid = hybrid_dataset['Weekly_Sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

ml_model_hbr.fit(X_hybrid, y_hybrid)
# %%

# Provide comparative statistical diagnostics over the generated Hybrid structure
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

# Inference & Performance Assessment of the Hybrid Model Baseline

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
results_df = pd.DataFrame({
    'Christmas Season': holdout_test_df['Is_Christmas_Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%

# Directly estimate and evaluate the causal mechanism of target Sales
# adapting directly to the composite Hybrid Dataset.
fit_causal_model_of_target(scm, "Weekly_Sales", hybrid_dataset)


def evaluate_sales_on_holdout(model_scm, eval_df, label):
    """Evaluate Weekly_Sales mechanism on the strict temporal Holdout set to prevent data leakage."""
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
from sklearn.model_selection import KFold
from xgboost import XGBClassifier



# Performance Degradation Decomposition Methodology

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
    Decomposes predictive performance degradation under out-of-distribution (OOD) shift.
    Estimates the proportions attributed to Marginal Feature Shift (X-shift) versus
    Conditional Label Shift (Y|X-shift) leveraging density ratio estimation via
    K-Fold trained Domain Classifiers.
    """
    # Truncate Target arrays to equalize volumetric discrepancy for balanced classification
    perm1 = np.random.permutation(target_X_raw.shape[0])
    target_X = target_X_raw[perm1[:data_sum], :]
    target_y = target_y_raw[perm1[:data_sum]]

    piA = np.zeros(source_X.shape[0])
    piB = np.zeros(target_X.shape[0])

    permA = np.random.permutation(source_X.shape[0])
    permB = np.random.permutation(target_X.shape[0])

    # K-Fold Stratification to mitigate overfitting in density ratio probability estimates
    kf = KFold(n_splits=K, shuffle=False)

    A_train_index_list, A_test_index_list = [], []
    B_train_index_list, B_test_index_list = [], []

    for train_idx, test_idx in kf.split(source_X):
        A_train_index_list.append(train_idx)
        A_test_index_list.append(test_idx)

    for train_idx, test_idx in kf.split(target_X):
        B_train_index_list.append(train_idx)
        B_test_index_list.append(test_idx)

    # Train discriminator to separate Source (label 0) and Target (label 1) distributions
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

    # Generate calibration plots demonstrating structural alignment overlap
    if draw_calibration:
        plot_calibration(piA, piB, save_dir=save_calibration_png)

    alpha = target_X.shape[0] / (source_X.shape[0] + target_X.shape[0])

    # Compute importance weights mapping Source marginals to Target marginals
    wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
    wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))

    # Normalize computed weights
    wA /= np.sum(wA)
    wB /= np.sum(wB)

    pred_source = model.predict(source_X)
    pred_target = model.predict(target_X)

    loss_source = np.abs(pred_source - source_y)
    loss_target = np.abs(pred_target - target_y)

    # Calculate Empirical Risk components
    errorA = np.mean(loss_source)
    errorB = np.mean(loss_target)

    # Calculate Importance-Weighted Risk components estimating theoretical bounds
    sx_A = np.dot(wA, loss_source)
    sx_B = np.dot(wB, loss_target)

    return errorA, errorB, sx_A, sx_B


def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None,
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir='.', balance=False):
    """
    Renders diagnostic histograms identifying the calibration density overlap
    associated with propensity score calculations.
    """
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


# Heterogeneous Treatment Effect / Risk Region Analysis: Target vs Synthetic

# TARGET - SYNTHETIC REGION ANALYSIS
synth_df = synthetic_dataset.copy()
target_df = holdout_test_df.copy()

drop_cols = ['Weekly_Sales']

X_syn = synth_df.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
# Employs global categorical one-hot encoding to resolve topological dimension variance
# prior to spatial subpopulation segmentation via Decision Trees.
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


# Base Regressor initialization for establishing theoretical degradation benchmarks
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

# Conduct degradation analysis yielding p2p (Source Error), q2q (Target Error),
# p2s (Reweighted Source), and s2q (Reweighted Target) matrices.
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

# Calculate Importance Sampling weights mitigating underlying covariate imbalances
wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

source_model = LGBMRegressor(**lgb_params)

# Estimate regressors bounded explicitly by domain-conditional Importance Sampling weights
source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model = LGBMRegressor(**lgb_params)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

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

# Deploy surrogate Decision Tree to interpret regions of maximum predictive divergence
# bounded by specific feature thresholds (identifying mechanistic risk regions).
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")

plt.savefig("TARGET - SYNTHETIC CHRISTMAS REGION ANALYSIS.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# Heterogeneous Treatment Effect / Risk Region Analysis: Real vs Synthetic

# REAL - SYNTHETIC REGION ANALYSIS
real_df = source_df.copy()
target_df = synthetic_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
# Employs global categorical one-hot encoding to resolve topological dimension variance.
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

# Execute identical Degradation Decomposition pipeline comparing the Observational (Real)
# versus Counterfactual (Synthetic) operational manifolds.
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

# Re-estimate surrogate interpretability model across the new domain juxtaposition
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("CHRISTMAS REAL training - SYNTHETIC  REGION ANALYSIS.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# Heterogeneous Treatment Effect / Risk Region Analysis: Real vs Hybrid

# REAL - Hybrid REGION ANALYSIS
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

# Execute Decomposition comparing standard Real observational samples vs Composite Hybrid data
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
plt.savefig("ALL_REAL_HYBRID_CHRISTMAS_tree.png")
plt.show()

# %%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight


# Heterogeneous Treatment Effect / Risk Region Analysis: Hybrid vs Target

# Hybrid - Target REGION ANALYSIS
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

# Execution profiling final juxtaposition of Composite Hybrid distributions
# against Ground Truth out-of-distribution manifestations.
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

# Export spatial risk delineation to final plot.
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("hybrid_target_CHRISTMAS.png")
plt.show()

# %%