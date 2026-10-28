
# W3D3: Decision Trees and Random Forests
# Cynaris Solutions - AI/ML Internship

import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# LangGraph workflow
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

EXPERIMENT_NAME = "Cynaris-W3D3-Decision-Trees"

# Use paths relative to this Python file, not the terminal folder.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_PLOT_PATH = os.path.join(OUTPUT_DIR, "decision_tree.png")
IMPORTANCE_PLOT_PATH = os.path.join(
    OUTPUT_DIR, "feature_importance.png"
)

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 2. LOAD AND PREPARE THE DATASET
# --------------------------------------------------

data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
classes = data.target_names

print("=== Iris Dataset Overview ===")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Flower classes: {list(classes)}")

print("\n=== First Five Rows ===")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== Dataset Split ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# 3. DECISION TREE DEPTH EXPERIMENTS + MLFLOW
# --------------------------------------------------

print("\n=== Effect of Tree Depth ===")

for depth in [1, 2, 3, 5, None]:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    tree.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, tree.predict(X_train)
    )
    test_accuracy = accuracy_score(
        y_test, tree.predict(X_test)
    )

    cv_scores = cross_val_score(
        tree,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    with mlflow.start_run(
        run_name=f"Decision-Tree-Depth-{depth}"
    ):
        mlflow.log_param(
            "max_depth",
            str(depth) if depth is None else depth
        )
        mlflow.log_param("criterion", "gini")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

    print(
        f"Depth={str(depth):5s} | "
        f"Train={train_accuracy:.3f} | "
        f"Test={test_accuracy:.3f} | "
        f"CV={cv_scores.mean():.3f} "
        f"+/- {cv_scores.std():.3f}"
    )


# --------------------------------------------------
# 4. GINI VS ENTROPY + MLFLOW
# --------------------------------------------------

print("\n=== Gini vs Entropy ===")

for criterion in ["gini", "entropy"]:

    tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=3,
        random_state=42
    )
    tree.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, tree.predict(X_train)
    )
    test_accuracy = accuracy_score(
        y_test, tree.predict(X_test)
    )

    cv_scores = cross_val_score(
        tree, X_train, y_train, cv=5, scoring="accuracy"
    )

    with mlflow.start_run(
        run_name=f"Decision-Tree-{criterion}"
    ):
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())

    print(
        f"Criterion={criterion:7s} | "
        f"Train={train_accuracy:.3f} | "
        f"Test={test_accuracy:.3f} | "
        f"CV={cv_scores.mean():.3f}"
    )


# --------------------------------------------------
# 5. OVERFITTING CONTROLS + MLFLOW
# --------------------------------------------------

print("\n=== Testing Overfitting Controls ===")

settings = [
    {"min_samples_leaf": 1, "min_samples_split": 2},
    {"min_samples_leaf": 5, "min_samples_split": 2},
    {"min_samples_leaf": 1, "min_samples_split": 10}
]

for setting in settings:

    tree = DecisionTreeClassifier(
        max_depth=None,
        random_state=42,
        **setting
    )
    tree.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, tree.predict(X_train)
    )
    test_accuracy = accuracy_score(
        y_test, tree.predict(X_test)
    )

    with mlflow.start_run(
        run_name=(
            f"Tree-Leaf-{setting['min_samples_leaf']}"
            f"-Split-{setting['min_samples_split']}"
        )
    ):
        mlflow.log_param("max_depth", "None")
        mlflow.log_param(
            "min_samples_leaf", setting["min_samples_leaf"]
        )
        mlflow.log_param(
            "min_samples_split", setting["min_samples_split"]
        )
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

    print(
        f"Leaf={setting['min_samples_leaf']} | "
        f"Split={setting['min_samples_split']} | "
        f"Train={train_accuracy:.3f} | "
        f"Test={test_accuracy:.3f}"
    )


# --------------------------------------------------
# 6. SELECTED DECISION TREE, RULES AND VISUALIZATION
# --------------------------------------------------

print("\n=== Selected Decision Tree ===")

best_tree = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42
)
best_tree.fit(X_train, y_train)

tree_train_accuracy = accuracy_score(
    y_train, best_tree.predict(X_train)
)
tree_test_accuracy = accuracy_score(
    y_test, best_tree.predict(X_test)
)
tree_cv_scores = cross_val_score(
    best_tree, X_train, y_train, cv=5, scoring="accuracy"
)

print("\n=== Decision Tree Rules ===")
rules = export_text(
    best_tree, feature_names=list(data.feature_names)
)
print(rules)

# Save the tree image for reports and MLflow.
plt.figure(figsize=(16, 9))
plot_tree(
    best_tree,
    feature_names=data.feature_names,
    class_names=[str(name) for name in classes],
    filled=True,
    rounded=True
)
plt.title("Iris Decision Tree (Maximum Depth = 3)")
plt.tight_layout()
plt.savefig(TREE_PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n=== Feature Importance ===")

importances = best_tree.feature_importances_

sorted_features = sorted(
    zip(data.feature_names, importances),
    key=lambda item: item[1],
    reverse=True
)

for feature, importance in sorted_features:
    bar = "#" * int(importance * 40)
    print(f"{feature:25s} {importance:.4f} {bar}")

plt.figure(figsize=(8, 5))
plt.bar(
    [item[0] for item in sorted_features],
    [item[1] for item in sorted_features]
)
plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    IMPORTANCE_PLOT_PATH, dpi=150, bbox_inches="tight"
)
plt.close()


# Log the selected model, metrics and visual artifacts.
with mlflow.start_run(run_name="Selected-Decision-Tree-Depth-3"):

    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_param("criterion", "gini")
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)

    mlflow.log_metric("train_accuracy", tree_train_accuracy)
    mlflow.log_metric("test_accuracy", tree_test_accuracy)
    mlflow.log_metric("cv_accuracy_mean", tree_cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", tree_cv_scores.std())

    # Save both plots and the learned decision rules.
    mlflow.log_artifact(TREE_PLOT_PATH)
    mlflow.log_artifact(IMPORTANCE_PLOT_PATH)

    rules_path = os.path.join(OUTPUT_DIR, "decision_tree_rules.txt")
    with open(rules_path, "w", encoding="utf-8") as file:
        file.write(rules)
    mlflow.log_artifact(rules_path)
    # Model serialization is intentionally omitted; metrics and artifacts are tracked.

print(f"\nDecision Tree plot saved: {TREE_PLOT_PATH}")
print(f"Feature importance plot saved: {IMPORTANCE_PLOT_PATH}")


# --------------------------------------------------
# 8. RANDOM FOREST + MLFLOW
# --------------------------------------------------

print("\n=== Random Forest Comparison ===")

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
forest.fit(X_train, y_train)

forest_train_accuracy = accuracy_score(
    y_train, forest.predict(X_train)
)
forest_test_accuracy = accuracy_score(
    y_test, forest.predict(X_test)
)
forest_cv_scores = cross_val_score(
    forest, X_train, y_train, cv=5, scoring="accuracy"
)

print(f"Training accuracy: {forest_train_accuracy:.3f}")
print(f"Testing accuracy:  {forest_test_accuracy:.3f}")
print(
    f"Cross-validation:  {forest_cv_scores.mean():.3f} "
    f"+/- {forest_cv_scores.std():.3f}"
)

print("\n=== Random Forest Classification Report ===")
forest_report = classification_report(
    y_test,
    forest.predict(X_test),
    target_names=[str(name) for name in classes],
    zero_division=0
)
print(forest_report)

with mlflow.start_run(run_name="Random-Forest-100-Trees"):

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)

    mlflow.log_metric("train_accuracy", forest_train_accuracy)
    mlflow.log_metric("test_accuracy", forest_test_accuracy)
    mlflow.log_metric("cv_accuracy_mean", forest_cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", forest_cv_scores.std())

    report_path = os.path.join(
        OUTPUT_DIR, "random_forest_classification_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(forest_report)

    mlflow.log_artifact(report_path)
    # Model serialization is intentionally omitted; metrics and artifacts are tracked.


# --------------------------------------------------
# 9. FINAL MODEL COMPARISON AND REPORT
# --------------------------------------------------

print("\n=== Final Model Comparison ===")
print(f"Decision Tree (depth 3): {tree_test_accuracy:.3f}")
print(f"Random Forest:           {forest_test_accuracy:.3f}")

print("\n=== Decision Tree Classification Report ===")

tree_report = classification_report(
    y_test,
    best_tree.predict(X_test),
    target_names=[str(name) for name in classes],
    zero_division=0
)
print(tree_report)


# --------------------------------------------------
# 10. MINIMAL LANGGRAPH WORKFLOW
# --------------------------------------------------

# The graph coordinates the final evaluation summary.
# The training and MLflow experiments above remain unchanged.

class WorkflowState(TypedDict):
    tree_accuracy: float
    forest_accuracy: float
    summary: str


def compare_models(state: WorkflowState) -> WorkflowState:
    """Compare the final test accuracies of both models."""

    if state["tree_accuracy"] > state["forest_accuracy"]:
        winner = "Decision Tree"
    elif state["forest_accuracy"] > state["tree_accuracy"]:
        winner = "Random Forest"
    else:
        winner = "Both models have equal test accuracy"

    summary = (
        f"Decision Tree test accuracy: {state['tree_accuracy']:.3f}\n"
        f"Random Forest test accuracy: {state['forest_accuracy']:.3f}\n"
        f"Result on this test split: {winner}"
    )

    return {**state, "summary": summary}


workflow = StateGraph(WorkflowState)
workflow.add_node("compare_models", compare_models)
workflow.add_edge(START, "compare_models")
workflow.add_edge("compare_models", END)

app = workflow.compile()

workflow_result = app.invoke({
    "tree_accuracy": tree_test_accuracy,
    "forest_accuracy": forest_test_accuracy,
    "summary": ""
})

print("\n=== LangGraph Model Comparison Workflow ===")
print(workflow_result["summary"])

print("\nW3D3 experiments completed.")
print(f"MLflow experiment: {EXPERIMENT_NAME}")
