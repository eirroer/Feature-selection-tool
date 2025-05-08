import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from NormalizerVST import NormalizerVST
from NormalizerTMM import NormalizerTMM
from NormalizerDESEQ2 import NormalizerDESEQ2
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay


def write_hyperparameters_to_file(best_params: dict, output_path_hyperparams):
    """Write the best hyperparameters to a file."""
    # save the best hyperparameters
    df_best = pd.DataFrame(
        list(best_params.items()), columns=["parameter", "value"]
    )
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=",", index=False, header=True)

def save_model(model, output_path_model):
    """Save the model to a file."""
    os.makedirs(os.path.dirname(output_path_model), exist_ok=True)
    with open(output_path_model, "wb") as file:
        pickle.dump(model, file)

def write_feature_importance_to_file(X, feature_importances, output_path_file):
    """Write the feature importance to a file."""

    features = X.columns
    top_features = pd.DataFrame(
        {"Feature": features, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)
    top_features = top_features.head(10)

    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=",", index=False)

def plot_feature_importance(X, feature_importances, output_path_plot):
    """Plot the feature importance."""
    features = X.columns
    top_features = pd.DataFrame(
        {"Feature": features, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)
    top_features = top_features.head(10)
    # reverse the order of the top features for better visualization
    top_features = top_features.iloc[::-1]
    plt.figure()
    plt.barh(range(top_features.shape[0]), top_features["Importance"], align="center")
    plt.yticks(range(top_features.shape[0]), top_features["Feature"])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path_plot), exist_ok=True)
    plt.savefig(output_path_plot, dpi=300, format="png")

def hyperopt(X, y, model, space, max_evals):
    """Perform hyperopt search cross validation."""
    def objective(params):
        model.set_params(**params)
        score = cross_val_score(model, X, y, cv=5).mean()
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best)
    return best_params

def run_feature_selection(
    count_file: str,
    metadata_file: str,
    config_file: str,
    feature_selection_method: str,
    output_path_file: str,
    output_path_plot: str,
    output_path_hyperparams: str,
    output_path_model: str,
    output_path_scores: str,
    output_path_roc_curve: str,
):
    # read the configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # read the data
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)
    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    hyperparameter_optimization_method = config["feature_selection"][
        "hyperparameter_optimization_method"
    ]
    if hyperparameter_optimization_method not in ["gridsearch", "randomsearch"]:
        raise ValueError(f"Hyperparameter tuning method {hyperparameter_optimization_method} not supported.")

    param_grid = config["feature_selection"][feature_selection_method][hyperparameter_optimization_method]["param_grid"]
    param_grid = {f"classifier__{key}": value for key, value in param_grid.items()}  # add classifier__ to the keys

    # if 'none' is in the list, replace it with None
    for key, value in param_grid.items():
        if 'None' in value:
            value = [None if x == 'None' else x for x in value]
            param_grid[key] = value
    # print(f"Parameters: {param_grid}")

    # Create the pipeline
    pipeline_steps = []

    normalization_methods = [
        method
        for method, is_use in config["preprocessing"]["normalization_methods"].items()
        if is_use["use_method"]
    ]

    if normalization_methods:
        for method in normalization_methods:
            if method == "vst":
                pipeline_steps.append(("normalizer_vst", NormalizerVST(metadata=metadata)))
            elif method == "tmm":
                pipeline_steps.append(("normalizer_tmm", NormalizerTMM()))
            elif method == "deseq2":
                pipeline_steps.append(("normalizer_deseq2", NormalizerDESEQ2()))
            else:
                raise ValueError(f"Normalization method {method} not supported.")

    model = None
    if feature_selection_method == "random_forest":
        model_name = "Random Forest"
        model = RandomForestClassifier(random_state=42)
    elif feature_selection_method == "xgboost":
        model_name = "XGBoost"
        model = XGBClassifier(random_state=42)
    # elif feature_selection_method == "rfe_lr":
    #     model = RFE(LogisticRegression('elasticnet'), n_features_to_select=1)
    else:
        raise ValueError(f"Feature selection method {feature_selection_method} not supported.")
    pipeline_steps.append(("classifier", model))

    pipeline = Pipeline(pipeline_steps)

    scoring = {
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1": make_scorer(f1_score, average="macro"),
    }

    refit = config["feature_selection"]["refit"]
    if refit not in scoring.keys():
        raise ValueError(f"Refit metric {refit} not supported.")

    cv = config["feature_selection"]["cv"]
    if cv < 2:
        raise ValueError(f"Cross-validation must be at least 2.")

    verbose = config["feature_selection"]["verbose"]
    if verbose not in range(0, 11):
        raise ValueError(f"Verbose must be between 0 and 10.")

    n_splits = cv
    cv_strategy = StratifiedKFold(n_splits=n_splits)  # Can be any cross-validation strategy

    if hyperparameter_optimization_method == "gridsearch":
        gridsearch = GridSearchCV(pipeline, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, verbose=verbose, scoring=scoring, refit=refit, return_train_score=True)
    elif hyperparameter_optimization_method == "randomsearch":
        gridsearch = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=cv_strategy, n_jobs=-1, verbose=verbose, scoring=scoring, refit=refit)
    else:
        raise ValueError(f"Hyperparameter optimization method {hyperparameter_optimization_method} not supported.")

    gridsearch.fit(X, y) # Fit the model

    write_hyperparameters_to_file(gridsearch.best_params_, output_path_hyperparams)
    save_model(gridsearch.best_estimator_.named_steps["classifier"], output_path_model)
    write_feature_importance_to_file(
        X,
        gridsearch.best_estimator_.named_steps["classifier"].feature_importances_,
        output_path_file,
    )
    plot_feature_importance(
        X,
        gridsearch.best_estimator_.named_steps["classifier"].feature_importances_,
        output_path_plot,
    )

    # Save the scores
    best_index = gridsearch.best_index_
    best_results = {key: values[best_index] for key, values in gridsearch.cv_results_.items()}
    best_model = gridsearch.best_estimator_.named_steps["classifier"]

    y_true = y
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    # Additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    

    train_scores = pd.DataFrame(
        [[roc_auc, balanced_accuracy, accuracy, precision, recall, f1]],
        columns=[
            "mean_train_roc_auc",
            "mean_train_balanced_accuracy",
            "mean_train_accuracy",
            "mean_train_precision",
            "mean_train_recall",
            "mean_train_f1",
        ],
        index=[f"{model_name} (train)"],
    ).round(4)


    cv_scores = pd.DataFrame(
        best_results,
        columns=[
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_precision",
            "std_test_precision",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_f1",
            "std_test_f1",
        ],
        # + [f"split{i}_test_{refit}" for i in range(n_splits)],
        index=[f"{model_name} (CV)"],
    ).round(4)

    all_scores = pd.concat([train_scores, cv_scores], axis=1)

    # change the '_test_' to '_CV_' in the column names
    all_scores.columns = [col.replace("_test_", "_CV_") for col in all_scores.columns]

    os.makedirs(os.path.dirname(output_path_scores), exist_ok=True)
    all_scores.to_csv(output_path_scores, index=True, header=True)

    # Plot the ROC curve
    best_model = gridsearch.best_estimator_.named_steps["classifier"]

    # Cross-validation setup
    cv = gridsearch.cv

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

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC Curve with Variability",
    )
    ax.legend(loc="lower right")
    fig.suptitle(f"{model_name}: cross-validation", fontsize=16)

    os.makedirs(os.path.dirname(output_path_roc_curve), exist_ok=True)
    plt.savefig(output_path_roc_curve, dpi=300, format="png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize hyperparameters for feature selection on the count data.")
    parser.add_argument(
        "--count_file",
        type=str,
        help="The file path to the count data.",
        required=True,
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="The file path to the metadata.",
        required=True,
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="The file path to the configuration file.",
        required=True,
    )
    parser.add_argument(
        "--feature_selection_method",
        type=str,
        help="The method to use for feature selection.",
        required=True,
    )
    parser.add_argument(
        "--output_path_file",
        type=str,
        help="The output path to save the feature importance file.",
        required=True,
    )
    parser.add_argument(
        "--output_path_plot",
        type=str,
        help="The output path to save the feature importance plot.",
        required=True,
    )
    parser.add_argument(
        "--output_path_hyperparams",
        type=str,
        help="The output path to save the best hyperparameters.",
        required=False,
    )
    parser.add_argument(
        "--output_path_model",
        type=str,
        help="The output path to save the best hyperparameters.",
        required=False,
    )
    parser.add_argument(
        "--output_path_scores",
        type=str,
        help="The output path to save the scores.",
        required=False,
    )
    parser.add_argument(
        "--output_path_roc_curve",
        type=str,
        help="The output path to save the ROC curve plot.",
        required=False,
    )

    args = parser.parse_args()

    run_feature_selection(
        args.count_file,
        args.metadata_file,
        args.config_file,
        args.feature_selection_method,
        args.output_path_file,
        args.output_path_plot,
        args.output_path_hyperparams,
        args.output_path_model,
        args.output_path_scores,
        args.output_path_roc_curve,
    )
