import argparse
import joblib
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC curves from GridSearchCV results.")

    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--best_models", type=str, nargs="+", required=True, help="Paths to GridSearchCV objects")
    parser.add_argument("--model_names", type=str, nargs="+", required=True, help="List of model names")
    parser.add_argument("--output_path_roc_curve", type=str, required=True, help="Path to save the ROC curve plot")

    return parser.parse_args()

def load_best_model(file_paths, model_names):
    """Load all grid search objects from the provided file paths."""
    best_models = {}
    for file_path, model_name in zip(file_paths, model_names):
        best_models[model_name] = joblib.load(file_path)
    return best_models

def plot_roc_curve(
    X,
    y,
    best_models,
    model_name,
    n_splits,
    output_path_roc_curve,
):
    """Plot the ROC curve."""
    

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=n_splits, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    X = X.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        best_model.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            best_model,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Compute mean and std deviation for the ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f ± %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # Confidence band (±1 std dev)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Mean ROC Curve with Variability", fontsize=16)
    ax.legend(loc="lower right")
    fig.suptitle(f"{model_name}", fontsize=16)

    os.makedirs(os.path.dirname(output_path_roc_curve), exist_ok=True)
    plt.savefig(output_path_roc_curve, dpi=300, format="png")
    plt.close()

def main():
    args = parse_args()

    # Load config file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load best models
    best_models = load_best_model(args.best_models, args.model_names)

    # print
    print("Loaded best models:")
    for model_name, model in best_models.items():
        print(f"Model: {model_name}")

    # TODO: Implement ROC curve plotting using the loaded GridSearchCV results

    print(f"ROC curves will be saved to {args.output_path_roc_curve}")

if __name__ == "__main__":
    main()