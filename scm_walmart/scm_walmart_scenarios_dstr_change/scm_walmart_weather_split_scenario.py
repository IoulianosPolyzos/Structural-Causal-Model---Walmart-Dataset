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
sys.path.append("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart")
from custom_mape import custom_retail_mape


try:
    with open("/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    lgb_params = config["lgb_params"]
except FileNotFoundError:
    print("No /home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/scm_walmart/config.yaml found, using default LGBM parameters.")
    lgb_params = {'n_estimators': 100, 'random_state': 42}


CONFIG = {
    'source_weather': ["Rain","Snow","Clouds"],
    'target_weather': ["Clear"],
    'random_seed': 42,

    'checkpoint_dir': './checkpoints_weather',

}
# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

def save_checkpoint(scm, feature_graph, parents, metrics, stage_name="checkpoint"):
    """Save SCM model and metadata to disk."""
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
    """Load SCM model and metadata from disk."""
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
    print("ERROR: Data file '/home/it2022091/Sales-Prediction---Distribution-Shifts---SCMs/walmart_dataset/final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY CITIES (WITH VALIDATION SET)



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY weather_conditions)")


source_weather = CONFIG['source_weather']
target_weather = CONFIG['target_weather']

source_df = df[df['weather_condition'].isin(source_weather)].copy().reset_index(drop=True)
target_df = df[df['weather_condition'].isin(target_weather)].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)


print(f"Train weather conditions: {source_weather}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest weather conditions: {target_weather}")


# Verify no overlap
assert set(source_weather).isdisjoint(set(target_weather)), "Train/Test weather conditions overlap!"
print("✓ No overlap between train/val/test weather_conditions")

#%%


print("PART 3: PREPROCESSING")


try:
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError as e:
                pass

    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

#%%


print("PART 4: CAUSAL GRAPH ANALYSIS")


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

affected_groups = get_all_descendants(macro_graph, "Weather")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Weather: {set(macro_graph.successors('Weather'))}")
print(f"All affected groups (transitive): {affected_groups}")


print("PART 5: FEATURE-LEVEL CAUSAL GRAPH")


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


scm = gcm.StructuralCausalModel(feature_graph)

def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
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
    gcm.fit(scm, train_df)
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

print("PART 7: SYNTHETIC DATA GENERATION & ML TRAINING")



def weather_intervention_fn(x):
    return 'Clear'


num_synthetic_samples = len(test_df)

synthetic_dataset = gcm.interventional_samples(
    scm,
    interventions={'weather_condition': weather_intervention_fn},
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

synthetic_dataset[cols] = synthetic_dataset[cols].clip(lower=0)


for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

print(f"Synthetic Dataset Shape: {synthetic_dataset.shape}")
print("\nTrain ML model on Synthetic Dataset...")

X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)
#%%

print("\n>>> RAW SYNTHETIC DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {synthetic_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {synthetic_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {synthetic_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {synthetic_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(synthetic_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")


print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {train_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {train_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {test_df['Weekly_Sales'].std():.4f}")
#%%
for col in classifier_nodes:
    all_categories = df[col].astype(str).unique()
    synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

# ---------------------------
#%%

print("PART 8: EVALUATION ON TEST SET")


print("\nEvaluation on Test Set ...")

X_test = test_df.drop(columns=['Weekly_Sales'])

X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = test_df['Weekly_Sales']

y_pred = ml_model_syn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Weather': test_df['weather_condition'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
#%%
print("\nHybrid Dataset (Source + Synthetic Target)...")


train_aligned = train_df[synthetic_dataset.columns].copy()

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

print("\n>>> RAW HYBRID DATA :")
print(f"Weekly_Sales stats:")
print(f"  Mean: {hybrid_dataset['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {hybrid_dataset['Weekly_Sales'].std():.4f}")
print(f"  Min:  {hybrid_dataset['Weekly_Sales'].min():.4f}")
print(f"  Max:  {hybrid_dataset['Weekly_Sales'].max():.4f}")
print(f"  % negative: {(hybrid_dataset['Weekly_Sales'] < 0).mean() * 100:.2f}%")


print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {train_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {train_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {test_df['Weekly_Sales'].std():.4f}")


#%%
print("\nEvaluation on Test Set ...")

X_test = test_df.drop(columns=['Weekly_Sales'])





X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = test_df['Weekly_Sales']

y_pred = ml_model_hbr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")
results_df = pd.DataFrame({
    'Weather': test_df['weather_condition'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
#%%
fit_causal_model_of_target(scm, "Weekly_Sales", hybrid_dataset)
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

adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        test_df,
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

    wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
    wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))

    wA /= np.sum(wA)
    wB /= np.sum(wB)


    pred_source = model.predict(source_X)
    pred_target = model.predict(target_X)

    loss_source = np.abs(pred_source - source_y)
    loss_target = np.abs(pred_target - target_y)

    # empirical risks
    errorA = np.mean(loss_source)
    errorB = np.mean(loss_target)

    # weighted risks
    sx_A = np.dot(wA, loss_source)
    sx_B = np.dot(wB, loss_target)

    return errorA,errorB,sx_A,sx_B

def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None,
                     nanmask_threshold=0.01, name='Prop Score',
                     save_dir='.', balance=False):

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
from whyshift_reg.disde_reg import *
#TARGET - SYNTHETIC REGION ANALYSIS
synth_df = synthetic_dataset.copy()
target_df = test_df.copy()

drop_cols = ['Weekly_Sales']

X_syn = synth_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
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

# Model trained on Synthetic data
print(f"Model trained on Synthetic data: MAE on Synthetic data test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Model trained on Synthetic data MAE on Real (OOD) Target Domain data test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on Synthetic data test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
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
plt.show()

#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
#REAL - SYNTHETIC REGION ANALYSIS
real_df = train_df.copy()
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
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.show()

#%%
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
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("hybrid_synthetic_tree.png")
plt.show()
#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
#Hybrid - Target REGION ANALYSIS
hybrid_df = hybrid_dataset.copy()
target_df = test_df.copy()

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
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.show()

#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
from whyshift.disde import degradation_decomp
# HYBRID ID OOD REGION ANALYSIS
df_hyb = hybrid_dataset.copy()


drop_cols = ['Weekly_Sales']
source_weather = CONFIG['source_weather']
target_weather = CONFIG['target_weather']
source_df = df_hyb[df_hyb['weather_condition'].isin(source_weather)].copy()
target_df = df_hyb[df_hyb['weather_condition'].isin(target_weather)].copy()

X = source_df.drop(columns=drop_cols)
X_target  = target_df.drop(columns=drop_cols)

# ONE HOT ENCODING (CRITICAL)
combined = pd.concat([X, X_target], axis=0)
combined_encoded = pd.get_dummies(combined,columns=categorical_nodes)

X_enc = combined_encoded.iloc[:len(X)]
X_target_enc  = combined_encoded.iloc[len(X):]

feature_names = X_enc.columns.tolist()

X_source_full = X_enc.values
X_target = X_target_enc.values

y_source_full = source_df['Weekly_Sales'].values
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
print(f"Source model MAE on ID test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Source model MAE on OOD test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on ID test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on OOD test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.show()

#%%
from sklearn.tree import plot_tree, DecisionTreeRegressor
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import *
#REGION ANALYSIS STANDARD WALMART DF ID OOD TEST
real_df = train_df.copy()
target_df = test_df.copy()

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
reg_domain_classifier = LGBMClassifier(**lgb_params)

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
print(f"Source model MAE on ID test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Source model MAE on OOD test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on ID test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on OOD test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.show()

