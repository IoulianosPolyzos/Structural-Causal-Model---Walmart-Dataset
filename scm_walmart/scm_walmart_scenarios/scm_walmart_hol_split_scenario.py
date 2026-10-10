#%%
#!/usr/bin/env python3


# Importation of requisite libraries for causal inference, machine learning,
## and graphical modeling.

import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime

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
sys.path.append("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart")
from custom_mape import custom_retail_mape

# Load experimental configurations and hyperparameters via an external YAML file.
with open("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart/config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

# Define foundational configuration mapping for domain adaptation parameters.
CONFIG = {
    'source_domain': 'Normal Days (IsHoliday=0)',
    'target_domain': 'Holiday Days (IsHoliday=1)',
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_holidays',

}

# Create checkpoint directory for model serialization.
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)


def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """
    Utility function for state serialization.
    Ensures the persistence of the fitted Structural Causal Model (SCM)
    and the associated topological parameters for reproducible experiments.

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

        # Save SCM architecture and learned weights
        scm_path = checkpoint_dir / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

        # Retain structural metadata and performance tracking metrics
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
    Deserializes and loads a previously computed SCM and its topological metadata
    from disk to circumvent redundant computational overhead.

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

        print(f"✓ Checkpoint loaded from {Path(scm_path).name}")
        return scm, metadata
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise


#%%


# PHASE 1: Data Ingestion and Initialization

print("PART 1: LOADING DATA & SETUP")


try:
    # Load the primary observational dataset and drop temporally redundant
    # or structurally uninformative variables to reduce dimensionality.
    df = pd.read_csv("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv")
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
    print("ERROR: Data file '/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv' not found!")
    raise



# PHASE 2: Domain Stratification

print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs HOLIDAY)")


# Ensure the intervention variable (IsHoliday) is strictly numeric.
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Bifurcate the dataset into a Source Domain (observational/normal dynamics)
# and a Target Domain (shifted distribution/holidays).
source_df = df[df['IsHoliday'] == 0].copy().reset_index(drop=True)
target_df = df[df['IsHoliday'] == 1].copy().reset_index(drop=True)


adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.20,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source domain: {CONFIG['source_domain']}")
print(f"Source shape: {source_df.shape}")
print(f"\nTarget domain: {CONFIG['target_domain']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Empirical verification of domain orthogonality.
assert (source_df['IsHoliday'] == 0).all(), "Train set contains non-zero IsHoliday!"
assert (adaptation_df['IsHoliday'] == 1).all(), "Adaptation set contains non-holiday days!"
assert (holdout_test_df['IsHoliday'] == 1).all(), "Holdout set contains non-holiday days!"
print("✓ Domain separation verified: No overlap between Normal/Holiday days")

#%%


# PHASE 3: Data Transformation and Preprocessing

print("PART 3: PREPROCESSING")


try:
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Type coercion: Enforce continuous covariates to maintain numeric dtype formats.
    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Type coercion: Transform binary and nominal variables into categorical representations
    # requisite for appropriate classification mechanism fitting within the SCM.
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


# PHASE 4: Macro-Level Causal Graph Specification

print("PART 4: CAUSAL GRAPH ANALYSIS")


# A priori mapping of covariate subsets to overarching causal constructs.
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

# Specification of directed edges between macroscopic causal groups.
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

macro_graph = nx.DiGraph()
macro_graph.add_nodes_from(causal_groups.keys())
macro_graph.add_edges_from(causal_edges_groups)

def get_all_descendants(graph, source_node):
    """
    Traverses the macroscopic DAG to identify all transitive descendants of a given source node.
    Critical for isolating the subgraph vulnerable to distributional shifts originating at the source.
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

# Isolate causal paths emanating from the intervention node ('Holidays').
affected_groups = get_all_descendants(macro_graph, "Holidays")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Holidays: {set(macro_graph.successors('Holidays'))}")
print(f"All affected groups (transitive): {affected_groups}")


# PHASE 5: Micro-Level (Feature) Causal Graph Construction

print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


# Translate the macroscopic DAG into an atomic, feature-level structural graph.
feature_graph = nx.DiGraph()
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)

for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)

# Interventional structural modification: Pruning direct influence to model mediated effects strictly.
feature_graph.remove_edge('IsHoliday','Weekly_Sales')
for edge in feature_graph.edges:
    print(edge)
print("IsHoliday successors : ",list(feature_graph.successors("IsHoliday")))
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")


# PHASE 6: Structural Causal Model (SCM) Estimation

print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")


scm = gcm.StructuralCausalModel(feature_graph)

def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    """
    Assigns generative mechanisms to the nodes of the SCM.
    - Root nodes: Modeled via empirical marginal distributions.
    - Categorical nodes: Modeled utilizing gradient boosted classifiers (LGBM).
    - Continuous descendent nodes: Modeled via Additive Noise Models (ANMs) utilizing LGBM regression.
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
    # Formulate the baseline generative mechanisms exclusively on the Source Domain.
    gcm.fit(scm, source_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

try:
    # Serialize baseline state geometry.
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {'status': 'initial_fit_complete'},
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save checkpoint: {e}")

#%%


# PHASE 8: Supervised Domain Adaptation via Structural Concept Drift

print("PART 8: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")



# Execute topological sort to ensure causal mechanism refitting adheres to the causal hierarchy.
ordered_nodes = list(nx.topological_sort(feature_graph))

# Isolate mechanisms requiring updating: strictly the causal descendants of the shift indicator
# (excluding the final target proxy to prevent label leakage during adaptation).
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes and n != "Weekly_Sales"
]

nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features ONLY - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"  ✓ {node}")

print(f"\nAdapting mechanisms to adaptation set (learning new behavior for features)...")

try:
    refit_count = 0
    # Iteratively update causal mechanism parameters utilizing the adaptation set,
    # mapping Target Domain dynamics onto the previously Source-learned graph.
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

#%%

# PHASE 9: Counterfactual Inference & Surrogate Model Training

print("PART 9: SYNTHETIC DATA GENERATION & ML TRAINING")


print("\nExecuting Intervention: Changing days to Holidays (IsHoliday = 1)...")


def holiday_intervention_fn(x):
    """Atomic interventional assignment implementing do(IsHoliday=1)."""
    return 1


num_synthetic_samples = len(target_df)

# Synthesize target domain observations by executing the defined do-calculus intervention
# on the adapted SCM structural parameters.
synthetic_dataset = gcm.interventional_samples(
    scm,
    interventions={'IsHoliday': holiday_intervention_fn},
    num_samples_to_draw=num_synthetic_samples
)
print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")
cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

# Filtering structurally anomalous negative generation artifacts to maintain econometric validity.
synthetic_dataset[cols] = synthetic_dataset[cols].clip(lower=0)
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

print(f"Synthetic Dataset Shape: {synthetic_dataset.shape}")

print("\nTrain ML model on Synthetic Dataset...")

# Train a discriminative machine learning surrogate using exclusively synthetic counterfactual data.
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)
#%%

# Diagnostic evaluation of synthetic distribution moments against the observational baselines.
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
print(f"  Mean: {target_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {target_df['Weekly_Sales'].std():.4f}")
#%%


# PHASE 10: Empirical Holdout Evaluation

print("PART 10: EVALUATION ON TEST SET (FULL)")


print("\nEvaluation on Test Set ...")

X_test = target_df.drop(columns=['Weekly_Sales'])

# Ensure strict feature space consistency for categorical evaluation.
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = target_df['Weekly_Sales']

# Inference phase evaluating zero-shot Target domain capabilities of the synthetic surrogate.
y_pred = ml_model_syn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on target_df:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'IsHoliday': target_df['IsHoliday'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
#%%

# Hybrid Domain Augmentation Methodology

print("\nHybrid Dataset (Source + Synthetic Target)...")


# Concatenation of the observed Source domain and synthetic Target counterfactuals
# to establish a generalized, cross-domain training manifold.
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

# Assessment of the composite hybrid distributions moments.
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
print(f"  Mean: {target_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {target_df['Weekly_Sales'].std():.4f}")


#%%

# Quantitative inference of the Hybrid model against the Target holdout set.
print("\nEvaluation on Test Set ...")

X_test = target_df.drop(columns=['Weekly_Sales'])





X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = target_df['Weekly_Sales']

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
    'IsHoliday': target_df['IsHoliday'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
#%%
fit_causal_model_of_target(scm, "Weekly_Sales", hybrid_dataset)
def evaluate_sales_on_test_full(model_scm, eval_df, label):
    """Evaluate Weekly_Sales mechanism on the exact same test set."""
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

adapted_metrics_holdout = evaluate_sales_on_test_full(
        scm,
        target_df,
        "ADAPTED SCM (after adaptation)"
    )

# Risk Region Analysis: Target vs. Synthetic

from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *

#TARGET - SYNTHETIC REGION ANALYSIS
synth_df = synthetic_dataset.copy()
target_df = target_df.copy()

drop_cols = ['Weekly_Sales']

X_syn = synth_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL): Align sparse manifolds across domain definitions.
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

# Execute shift attribution formulation on Synthetic -> Target discrepancy.
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
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

# Calculate importance weighted strata to map localized model discrepancies.
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

# Train shallow Decision Tree mapped over residual divergence (new_Y) to interpretably
# surface heterogeneous covariate sub-spaces experiencing high failure rates.
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")

plt.savefig("TARGET - SYNTHETIC holiday REGION ANALYSIS.png")
plt.show()


#%%

# Risk Region Analysis: Real Source vs. Synthetic Generation

from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
#REAL - SYNTHETIC REGION ANALYSIS
real_df = source_df.copy()
target_df = synthetic_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
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

# Execute shift attribution formulation on Source Real -> Synthetic discrepancy.
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
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

# Risk Stratification via Decision Tree representation of residual divergence.
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("HOLIDAY REAL training - SYNTHETIC  REGION ANALYSIS.png")
plt.show()


#%%

# Risk Region Analysis: Real Domain vs. Hybrid Domain

from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *

#REAL - Hybrid REGION ANALYSIS
real_df = source_df.copy()
target_df = hybrid_dataset.copy()

drop_cols = ['Weekly_Sales']

X = real_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
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

# Execute shift attribution formulation on Real Domain -> Hybrid Dataset discrepancy.
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
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
print(f"Model trained on real data: MAE on real data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Model trained on real data: MAE on hybrid data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Hybrid model:  MAE on real data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Hybrid model:  MAE on hybrid data test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

# Risk Stratification via Decision Tree.
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("ALL_REAL_HYBRID_HOLIDAY_tree.png")
plt.show()


#%%

# Risk Region Analysis: Hybrid Domain vs. Target Domain

from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
#Hybrid - Target REGION ANALYSIS
hybrid_df = hybrid_dataset.copy()
target_df = target_df.copy()

drop_cols = ['Weekly_Sales']

X = hybrid_dataset.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
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

# Execute shift attribution formulation on Hybrid Domain -> Final Target discrepancy.
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
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

# Risk Stratification via Decision Tree.
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("hybrid_target_HOLIDAY.png")
plt.show()