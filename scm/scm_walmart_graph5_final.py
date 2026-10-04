#%%
import os
import warnings

from sklearn.metrics import mean_squared_error

# Suppress warning messages to ensure cleaner experimental logs.
# This is common in empirical studies where excessive warnings may obscure key outputs.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# Import machine learning models and utilities.
# LightGBM is selected due to its efficiency and strong performance on tabular data.
from lightgbm import LGBMRegressor, LGBMClassifier

# Import data splitting utility for constructing training and evaluation sets.
from sklearn.model_selection import train_test_split

# Core data manipulation and graph libraries.
import pandas as pd
import networkx as nx

# Import DoWhy's graphical causal modeling framework.
# This enables structural causal model (SCM) specification and inference.
import dowhy.gcm as gcm

# Numerical and plotting libraries for computation and visualization.
import numpy as np
from matplotlib import pyplot as plt

# Wrappers to integrate sklearn models into DoWhy's causal mechanisms.
from dowhy.gcm.ml import SklearnRegressionModel, SklearnClassificationModel


# Load the dataset from disk.
# The dataset represents a preprocessed version of the Walmart sales data.
df = pd.read_csv("/home/it2022091/Structural-Causal-Model---Walmart-Dataset/walmart_dataset/final_data_walmart.csv")

# Drop redundant or already-encoded temporal variables.
# These variables are removed either due to feature engineering replacements
# (e.g., cyclic encodings) or to prevent multicollinearity.
df = df.drop(columns=["Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')

# Handle missing values via zero imputation.
# While simplistic, this approach ensures model compatibility;
# however, it implicitly assumes missingness is non-informative.
df = df.fillna(0)

# Report dataset dimensionality.
# This provides a quick sanity check before proceeding to modeling.
print(f"Dataset shape: {df.shape}")

# Split dataset into training and testing subsets.
# This enables unbiased evaluation of model generalization performance.
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Report resulting split sizes.
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
#%%
# Define categorical variables that should be excluded from numeric coercion.
# These variables represent nominal or identifier-based information and must
# retain their categorical nature for downstream causal modeling.
categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

# Attempt to convert all remaining columns to numeric types.
# This enforces consistency for continuous variables and ensures compatibility
# with regression-based structural mechanisms.
for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
            except ValueError as e:
                # Log conversion failures for diagnostic purposes without interrupting execution.
                print(f"WARNING: Could not convert {col} to numeric: {e}")

# Define binary variables.
# These variables are explicitly cast to string type.sh to ensure they are treated
# as categorical variables rather than ordinal numerical values.
binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]

# Convert binary variables into categorical string representation.
# This aligns with classifier-based causal mechanisms in the SCM.
for col in binary_nodes:
    train_df[col] = train_df[col].astype(int).astype(str)
    test_df[col] = test_df[col].astype(int).astype(str)

# Define multi-class categorical variables.
# These represent nominal features with no inherent ordering.
categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]

# Ensure consistent string typing across training and test sets.
# This prevents type.sh mismatches during model fitting and inference.
for col in categorical_nodes:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Aggregate all categorical variables (binary + multi-class) that will be
# modeled using classification-based mechanisms in the SCM.
classifier_nodes = binary_nodes + categorical_nodes

# Inspect data types and sample unique values for classifier nodes.
# This serves as a validation step to confirm correct preprocessing.
print("\nDtypes classifier nodes (Train):")
for col in classifier_nodes:
    print(f"  {col}: dtype={train_df[col].dtype}, unique={sorted(train_df[col].unique())[:6]}")
#%%

# Definition of high-level causal abstractions (macro-level variables)

# This dictionary defines a hierarchical grouping of observed variables into
# semantically meaningful latent constructs. Such grouping is commonly used
# in causal modeling to reduce dimensionality and to enforce interpretability
# at a higher level of abstraction (macro-causal graph).
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


# Specification of causal structure between macro-level variables

# These directed edges encode domain-informed assumptions about the data-
# generating process. They define a Directed Acyclic Graph (DAG) over
# latent groups, which is later expanded into a fine-grained feature-level
# causal graph.
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
#%%

# Construction of macro-level causal graph (Directed Acyclic Graph - DAG)

# The graph encodes causal relationships between high-level feature groups.
# Each node represents a latent conceptual variable, while edges encode
# assumed causal influence directions based on domain knowledge.

macro_graph = nx.DiGraph()

# Add nodes corresponding to causal abstractions defined in causal_groups.
macro_graph.add_nodes_from(causal_groups.keys())

# Add directed edges encoding hypothesized causal mechanisms.
macro_graph.add_edges_from(causal_edges_groups)


# Graph layout specification for visualization

# The manual layout enforces interpretability by positioning variables
# according to assumed temporal/causal hierarchy (top → bottom flow).
pos = {
    "City": (0, 3), "Date_Features": (2, 3),
    "Season": (4, 2.5), "Holidays": (6, 2.5),
    "Store_Features": (1, 2),
    "Economy": (2, 2),
    "Weather": (4.5, 1.5),
    "MarkDowns": (6, 1),
    "Sales": (4, 0)
}


# Visualization of the macro-level causal DAG

# This visualization serves as a qualitative validation tool for the assumed
# causal structure before expanding to feature-level modeling.
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

plt.title("Causal Graph", fontsize=16)
plt.tight_layout()

# Save figure for reproducibility and inclusion in reports or publications.
plt.savefig("causal_graph.png", dpi=150)

plt.show()
#%%

# Construction of feature-level causal graph

# This step refines the macro-level causal DAG into a fine-grained graph
# where each node corresponds to an observed feature rather than a latent group.
# The resulting graph represents a full expansion of group-level assumptions
# into pairwise feature-level causal dependencies.

feature_graph = nx.DiGraph()


# Step 1: Add all observed variables as nodes

# Each feature from each causal group is explicitly added as a node in the graph,
# ensuring full representation of the observational space.
for group_name, columns in causal_groups.items():
    for col in columns:
        feature_graph.add_node(col)


# Step 2: Expand group-level edges into feature-level causal links

# Each edge between groups is translated into a complete bipartite structure
# between all features in the source group and all features in the target group.
# This assumes homogeneous causal influence within group interactions.
for source_group, target_group in causal_edges_groups:
    for src_col in causal_groups[source_group]:
        for tgt_col in causal_groups[target_group]:
            feature_graph.add_edge(src_col, tgt_col)


# Graph diagnostics

# Reporting the size of the induced causal graph provides a sanity check
# for combinatorial expansion and helps assess computational complexity.
print(f"\nNodes: {feature_graph.number_of_nodes()}")
print(f"Edges: {feature_graph.number_of_edges()}")
#%%

# Structural Causal Model (SCM) specification over feature-level DAG

# The SCM formalizes the data-generating process as a collection of structural
# equations, one per node in the causal graph. Each equation defines how a
# variable is generated from its direct parents plus a noise term.

scm = gcm.StructuralCausalModel(feature_graph)


# Mechanism assignment for each node in the causal graph

# We assign different functional forms depending on the statistical nature
# of each variable:
#   - Root nodes: empirical distributions (non-parametric modeling)
#   - Categorical nodes: classification-based functional causal models (FCM)
#   - Continuous nodes: additive noise models (ANM) with regression learners

for node in feature_graph.nodes:
    parents = list(feature_graph.predecessors(node))


    # Case 1: Exogenous variables (no parents)

    # These variables are modeled directly from their empirical distribution,
    # assuming no structural dependencies in the graph.
    if len(parents) == 0:
        scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())


    # Case 2: Discrete / categorical variables

    # These variables are modeled using a probabilistic classifier embedded
    # within a Functional Causal Model (FCM). This captures conditional
    # distributions P(X | Parents(X)).
    elif node in classifier_nodes:
        scm.set_causal_mechanism(
            node,
            gcm.ClassifierFCM(
                SklearnClassificationModel(
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=62,
                        max_depth=20,
                        verbose=-1,
                        n_jobs=1
                    )
                )
            )
        )


    # Case 3: Continuous variables

    # Continuous outcomes are modeled using an Additive Noise Model (ANM),
    # where the structural equation is learned via regression.
    else:
        scm.set_causal_mechanism(
            node,
            gcm.AdditiveNoiseModel(
                SklearnRegressionModel(
                    LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=62,
                        max_depth=20,
                        verbose=-1,
                        n_jobs=1
                    )
                )
            )
        )
#%%

# Fitting the Structural Causal Model to observational data

# This step estimates the parameters of all structural mechanisms defined
# in the SCM using observational training data. Under the assumption that
# the causal graph is correctly specified, this corresponds to learning
# the conditional distributions P(X_i | Parents(X_i)) for all nodes.

print("\nFitting SCM on training data...")

# Perform maximum likelihood / supervised fitting of all node mechanisms.
# Each structural equation is trained independently using its parent set.
gcm.fit(scm, train_df)
#%%

# Evaluation of the Structural Causal Model (SCM)

# This step assesses the quality of the learned causal model both in terms
# of:
#   (1) Causal mechanisms (i.e., goodness-of-fit of structural equations)
#   (2) Causal structure (i.e., consistency of the DAG with data)
#
# Such evaluation provides diagnostic insight into whether the SCM is
# capable of generating realistic interventional and observational behavior.

print("Evaluating SCM on test data...")


# Subsampling test data for computational efficiency

# A random subset of the test set is selected to reduce computational cost
# while maintaining statistical representativeness.
test_sample = test_df.sample(
    n=min(15000, len(test_df)),
    random_state=42
).reset_index(drop=True)


# SCM evaluation

# The evaluation function quantifies:
#   - Predictive accuracy of each causal mechanism
#   - Structural validity of the causal graph assumptions
#   - Overall model consistency under observational data
evaluation_metrics = gcm.evaluate_causal_model(
    scm,
    test_df,
    evaluate_causal_mechanisms=True,
    evaluate_causal_structure=True,
    max_num_samples=-1
)

# Output evaluation summary
print(evaluation_metrics)
#%%

# Direct evaluation of the structural mechanism for Weekly_Sales

# In this step, we isolate the learned structural equation corresponding to
# the target variable (Weekly_Sales) and evaluate its predictive performance
# on held-out test data. This provides a localized assessment of the SCM's
# accuracy for the primary outcome variable.

from sklearn.metrics import mean_absolute_error, r2_score

# Extract the learned causal mechanism for the target variable.
# This mechanism encodes the conditional distribution:
#   P(Weekly_Sales | Parents(Weekly_Sales))
sales_mechanism = scm.causal_mechanism("Weekly_Sales")

# Retrieve the causal parents of Weekly_Sales from the learned graph.
parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))


# Construct test design matrix based on causal parents

# Only variables identified as direct causes are used for prediction,
# ensuring consistency with the structural causal formulation.
X_test = test_df[parents].to_numpy()

# Generate predictions using the learned structural model.
# The prediction model corresponds to the regression component of the ANM.
test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()


# Compute standard regression evaluation metrics

# These metrics quantify predictive accuracy but do not capture causal validity.
mae = mean_absolute_error(test_df['Weekly_Sales'].values, test_predictions)
mse = mean_squared_error(test_df['Weekly_Sales'].values, test_predictions)
rmse = np.sqrt(mean_squared_error(test_df['Weekly_Sales'].values, test_predictions))
r2 = r2_score(test_df['Weekly_Sales'].values, test_predictions)
#%%

# Graph falsification analysis and visualization

# This step evaluates whether the assumed causal graph is consistent with
# observed data by testing for statistical violations of implied causal
# assumptions (e.g., conditional independencies or distributional mismatch).
#
# Graph falsification provides a diagnostic signal for potential misspecification
# of the causal structure.

if evaluation_metrics.graph_falsification is not None:


    # Visualization of falsification results

    # The plot summarizes which parts of the causal graph are most likely
    # to violate structural assumptions under the observed data distribution.
    gcm.falsify.plot_evaluation_results(evaluation_metrics.graph_falsification)

    # Save figure for reproducibility and reporting purposes.
    plt.savefig("graph_falsification.png", dpi=150, bbox_inches='tight')

    # Close figure to free memory and avoid overlap in subsequent plots.
    plt.close()