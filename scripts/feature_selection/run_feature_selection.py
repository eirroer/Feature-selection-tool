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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from NormalizerVST import NormalizerVST
from NormalizerTMM import NormalizerTMM
from NormalizerDESEQ2 import NormalizerDESEQ2
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

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
        model = RandomForestClassifier(random_state=42)
    elif feature_selection_method == "xgboost":
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

    if hyperparameter_optimization_method == "gridsearch":
        gridsearch = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=verbose, scoring=scoring, refit=refit)
    elif hyperparameter_optimization_method == "randomsearch":
        gridsearch = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv=cv, n_jobs=-1, verbose=verbose, scoring=scoring, refit=refit)
    else:
        raise ValueError(f"Hyperparameter optimization method {hyperparameter_optimization_method} not supported.")

    gridsearch.fit(X, y)

    # for metric in scoring.keys():
    #     mean_scores = gridsearch.cv_results_[f"mean_test_{metric}"]
    #     std_scores = gridsearch.cv_results_[f"std_test_{metric}"]
    #     print(f"\nResults for {metric}:")
    #     for params, mean, std in zip(gridsearch.cv_results_["params"], mean_scores, std_scores):
    #         print(f"params: {params} \n{metric.upper()}: {mean:.4f} ± {std:.4f}\n")

    # print(f"Best hyperparameters: {gridsearch.best_params_}")
    # print(f"Best AUC score: {gridsearch.best_score_:.4f}")

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

    print("cv_results_ columns:")
    print(pd.DataFrame(gridsearch.cv_results_).columns)

    # print("best_score := gridsearch.best_score_")
    # print(gridsearch.best_score_)

    # print all scores of the 
    print("gridsearch.cv_results_['mean_test_roc_auc']")
    print(gridsearch.cv_results_["mean_test_roc_auc"][gridsearch.best_index_])

    # best_mean_scores = {
    #     metric: gridsearch.cv_results_[f"mean_test_{metric}"][gridsearch.best_index_]
    #     for metric in scoring
    # }
    # print("best_mean_scores")
    # print(best_mean_scores)
    best_index = gridsearch.best_index_  # Get the best index (integer)
    best_results = {key: values[best_index] for key, values in gridsearch.cv_results_.items()}

    all_scores = pd.DataFrame(
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
        index=[feature_selection_method],
    ).round(4)
    # change the '_test_' to '_CV_' in the column names
    all_scores.columns = [col.replace("_test_", "_CV_") for col in all_scores.columns]

    os.makedirs(os.path.dirname(output_path_scores), exist_ok=True)
    all_scores.to_csv(output_path_scores, index=True, header=True)


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
    )
