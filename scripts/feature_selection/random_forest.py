import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score

sys.stdout.reconfigure(line_buffering=True)

def random_forest_gridsearch(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["param_grid"]
    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="accuracy",  # Metric to optimize
        n_jobs=-1,  # Use all CPUs
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Evaluate on validation data
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_val, y_val)
    print("Test Accuracy:", test_accuracy)

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=";", index=False)

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

    print("Random Forest completed successfully!")

def random_forest_randomsearch(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["param_grid"]

    # if 'none' is in the list, replace it with None
    for key, value in param_grid.items():
        if 'none' in value:
            value = [None if x == 'none' else x for x in value]
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
    print("Best Score:", grid_search.best_score_)

    # Evaluate on validation data
    best_model = grid_search.best_estimator_

    # show the auc score
    auc_score = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc').mean()
    print("AUC Score:", auc_score)

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=";", index=False)

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

    print("Random Forest completed successfully!")

def random_forest_hyperopt(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
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

    # space = {
    #     "criterion": hp.choice("criterion", ["entropy", "gini"]),
    #     "max_depth": hp.quniform("max_depth", 10, 1200, 10),
    #     "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    #     "min_samples_leaf": hp.uniform("min_samples_leaf", 0, 0.5),
    #     "min_samples_split": hp.uniform("min_samples_split", 0, 1),
    #     "n_estimators": hp.choice("n_estimators", [10, 50, 300, 750, 1200, 1300, 1500]),
    # }

    # print(f"Space: {space}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    def objective(space):
        space["max_depth"] = int(space["max_depth"])
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
        fn=objective, space=space, algo=tpe.suggest, max_evals=1, trials=trials
    )
    best = space_eval(space, best)

    print("Best Parameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    # make sure max depth is an integer
    best["max_depth"] = int(best["max_depth"])

    # convert the best dict to a csv file
    df_best = pd.DataFrame(list(best.items()), columns=["parameter", "value"])
    output_hp = "results/RF_hyperparameters.csv"
    os.makedirs(os.path.dirname(output_hp), exist_ok=True)
    df_best.to_csv(output_hp, sep=";", index=False, header=True)

    # Evaluate on validation data
    best_model = RandomForestClassifier(random_state=42, **best)
    best_model.fit(X, y)

    # show the auc score
    auc_score = cross_val_score(best_model, X, y, cv=5, scoring='roc_auc').mean()
    print("AUC Score:", auc_score)

    # Feature importances
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=";", index=False)

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

    args = parser.parse_args()

    random_forest_hyperopt(
        count_file=args.count_file,
        metadata_file=args.metadata_file,
        config_file=args.config_file,
        output_path_file=args.output_path_file,
        output_path_plot=args.output_path_plot,
    )
