import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel


def write_hyperparameters_to_file(best_params: dict, output_path_hyperparams):
    """Write the best hyperparameters to a file."""
    # save the best hyperparameters
    df_best = pd.DataFrame(
        list(best_params.items()), columns=["parameter", "value"]
    )
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

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
    top_features.to_csv(output_path_file, sep=";", index=False)

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

def gridsearchCV(X, y, model, param_grid, cv):
    """Perform grid search cross validation."""
    clf = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=10, scoring='roc_auc')
    clf.fit(X, y)
    return clf

def randomsearchCV(X, y, model, param_dist, cv):
    """Perform random search cross validation."""
    clf = RandomizedSearchCV(model, param_dist, cv=cv, n_jobs=-1,verbose=10, scoring='roc_auc')
    clf.fit(X, y)
    return clf

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
    hyperparameter_optimization_method: str,
    feature_selection_method: str,
    output_path_file: str,
    output_path_plot: str,
    output_path_hyperparams: str,
    output_path_model: str,
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

    param_grid = config["feature_selection"][feature_selection_method][hyperparameter_optimization_method]["param_grid"]

    # if 'none' is in the list, replace it with None
    for key, value in param_grid.items():
        if 'None' in value:
            value = [None if x == 'None' else x for x in value]
            param_grid[key] = value
    print(f"Parameters: {param_grid}")

    def define_space_from_param_grid(param_grid) -> dict:
        space = {}
        for param_name, param_details in param_grid.items():
            param_type = param_details["parameter_type"]
            values = param_details["values"]

            if param_type == "choice":
                space[param_name] = hp.choice(param_name, values)
            elif param_type == "quniform":
                space[param_name] = hp.quniform(param_name, *values)
            elif param_type == "uniform":
                space[param_name] = hp.uniform(param_name, *values)
            elif param_type == "loguniform":
                space[param_name] = hp.loguniform(param_name, *values)
            else:
                raise ValueError(f"Parameter type {param_type} not supported.")
        return space

    model = None
    if feature_selection_method == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif feature_selection_method == "xgboost":
        model = XGBClassifier(random_state=42)
    else:
        raise ValueError(f"Feature selection method {feature_selection_method} not supported.")

    if hyperparameter_optimization_method == "gridsearch":
        clf = gridsearchCV(X, y, model, param_grid, cv=5)
    elif hyperparameter_optimization_method == "randomsearch":
        clf = randomsearchCV(X, y, model, param_grid, cv=5)
    # elif hyperparameter_optimization_method == "hyperopt":
    #     space = define_space_from_param_grid(param_grid)
    #     clf = hyperopt(X, y, model, space, max_evals=2)
    #     clf = space_eval(space, best_params) # convert the best hyperparameters to the original space
    else:
        raise ValueError(f"Hyperparameter optimization method {hyperparameter_optimization_method} not supported.")

    all_auc_scores = clf.cv_results_["mean_test_score"]
    std_auc_scores = clf.cv_results_["std_test_score"]
    param_grid = clf.cv_results_["params"]
    for params, mean_auc, std_auc in zip(param_grid, all_auc_scores, std_auc_scores):
        print(f"\nparams: {params} \nAUC: {mean_auc:.4f} ± {std_auc:.4f}\n")

    print(f"Best hyperparameters: {clf.best_params_}")
    print(f"Best AUC score: {clf.best_score_:.4f}")


    write_hyperparameters_to_file(clf.best_params_, output_path_hyperparams)
    save_model(clf.best_estimator_, output_path_model)
    write_feature_importance_to_file(X, clf.best_estimator_.feature_importances_, output_path_file)
    plot_feature_importance(
        X, clf.best_estimator_.feature_importances_, output_path_plot
    )

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
        "--hyperparameter_optimization_method",
        type=str,
        help="The method to use for hyperparameter optimization.",
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
    args = parser.parse_args()

    run_feature_selection(
        args.count_file,
        args.metadata_file,
        args.config_file,
        args.hyperparameter_optimization_method,
        args.feature_selection_method,
        args.output_path_file,
        args.output_path_plot,
        args.output_path_hyperparams,
        args.output_path_model,
    )
