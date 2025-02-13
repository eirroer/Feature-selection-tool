import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

sys.stdout.reconfigure(line_buffering=True)

def write_feature_importance_to_file(top_features: pd.DataFrame, output_path_file):
    """Write the feature importance to a file."""
    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=";", index=False)

def plot_feature_importance(top_features: pd.DataFrame, output_path_plot):
    """Plot the feature importance."""
    # Plot the feature importance horizontally
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


def random_forest_gridsearch(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["param_grid_gridsearch"]
    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="roc_auc",  # Metric to optimize
        n_jobs=-1,  # Use all CPUs
        verbose=2,
    )

    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)

    # convert the best dict to a csv file
    df_best = pd.DataFrame(
        list(grid_search.best_params_.items()), columns=["parameter", "value"]
    )
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    best_model = grid_search.best_estimator_

    # show the auc score
    print("Calculating AUC score...")
    auc_score = cross_val_score(best_model, X, y, cv=5, scoring="roc_auc").mean()
    print("AUC Score:", auc_score)

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )
    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("Random Forest completed successfully!")

def random_forest_randomsearch(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["param_grid_randomsearch"]

    # if 'none' is in the list, replace it with None
    for key, value in param_grid.items():
        print(key, value)
        if 'None' in value:
            value = [None if x == 'None' else x for x in value]
            param_grid[key] = value

    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=10,  # Number of random samples
        cv=5,  # 5-fold cross-validation
        scoring="roc_auc",  # Metric to optimize
        n_jobs=-1,  # Use all CPUs
        verbose=2,
    )

    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)


    # convert the best dict to a csv file
    df_best = pd.DataFrame(list(grid_search.best_params_.items()), columns=["parameter", "value"])
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    best_model = grid_search.best_estimator_
    # show the auc score
    print("Calculating AUC score...")
    auc_score = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc').mean()
    print("AUC Score:", auc_score)

    # convert the best dict to a csv file
    df_best = pd.DataFrame(list(grid_search.best_params_.items()), columns=["parameter", "value"])
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("Random Forest completed successfully!")

def random_forest_hyperopt(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["space_params_hyperopt"]

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

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    def objective(space):
        space["max_depth"] = int(space["max_depth"])
        space["max_features"] = None if space["max_features"] == 'None' else space["max_features"]
        print(f"Parameters: {space}")

        model = RandomForestClassifier(
            criterion = space['criterion'],
            max_depth = space['max_depth'],
            max_features = space['max_features'],
            min_samples_leaf = space['min_samples_leaf'],
            min_samples_split = space['min_samples_split'],
            n_estimators = space['n_estimators'],
            random_state=42
        )

        # Feature selection using model importance
        selector = SelectFromModel(model, threshold="mean")
        selector.fit(X, y)

        X_selected = selector.transform(X)  # Apply the feature selection on the whole dataset

        # Perform cross-validation with AUC score
        auc_score = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc').mean()

        return {"loss": -auc_score, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials
    )
    best = space_eval(space, best)

    print("Best Parameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    # make sure max depth is an integer
    best["max_depth"] = int(best["max_depth"])

    # convert the best dict to a csv file
    df_best = pd.DataFrame(list(best.items()), columns=["parameter", "value"])
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    # Evaluate on validation data
    best_model = RandomForestClassifier(random_state=42, **best)
    best_model.fit(X, y)

    # show the auc score
    print("Calculating AUC score...")
    auc_score = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc').mean()
    print("AUC Score:", auc_score)

    # Feature importances
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("Random Forest completed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Random Forest on the count data.")
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
        "--method",
        type=str,
        help="The method to use for hyperparameter optimization.",
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
    args = parser.parse_args()

    if args.method == "gridsearch":
        random_forest_gridsearch(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    elif args.method == "randomsearch":
        random_forest_randomsearch(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    elif args.method == "hyperopt":
        random_forest_hyperopt(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")
