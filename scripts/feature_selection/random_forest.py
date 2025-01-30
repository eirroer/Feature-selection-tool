import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

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
    # space = config_data["feature_selection"]["random_forest"]["param_grid"]
    # # param_grid:
    # #   n_estimators: [50, 100, 200]
    # #   max_depth: [1, 10, 20, 100]
    # #   min_samples_split: [2, 5, 10]
    # #   min_samples_leaf: [1, 2, 4]
    # #   random_state: [42]

    # # Convert the parameter grid to hyperopt space
    # for key, value in space.items():
    #     if isinstance(value, list):
    #         space[key] = hp.choice
    #     else:
    #         raise ValueError("Only lists are supported for hyperopt space.")

    # Define the hyperparameter space
    space = {
        "n_estimators": hp.choice(
            "n_estimators", [50, 100, 200, 300]
        ),  # Ensures valid integer values
        "max_depth": hp.choice(
            "max_depth", [None, 10, 20, 30]
        ),  # None or specific integers
        "max_features": hp.uniform(
            "max_features", 0.1, 1.0
        ),  # Uniform float between 0.1 and 1.0
        "min_samples_split": hp.quniform(
            "min_samples_split", 2, 20, 1
        ),  # Integer values between 2 and 20
    }
    print(f"Space: {space}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the objective function

    def objective(params):
        # print(f"Evaluating: {params}")

        # Ensure correct types for hyperparameters
        params["n_estimators"] = int(params["n_estimators"])  # Convert to integer
        params["max_depth"] = int(params["max_depth"]) if params["max_depth"] is not None else None  # Handle None case
        params["min_samples_split"] = int(params["min_samples_split"])  # Ensure integer

        # print(f"Hyperparameters corrected : {params}")

        # RandomForestClassifier model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"], 
            min_samples_split=params["min_samples_split"],
            random_state=42,
            n_jobs=-1,
        )

        # Feature selection using model importance
        selector = SelectFromModel(model, threshold="mean")
        selector.fit(X_train, y_train)

        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_val)

        model.fit(X_train_selected, y_train)
        preds = model.predict(X_test_selected)

        accuracy = accuracy_score(y_val, preds)

        return {"loss": -accuracy, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials
    )

    print("Best Parameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")

    # make sure min samples split is an integer
    best["min_samples_split"] = int(best["min_samples_split"])

    # if max_depth is 0 then set it to None
    if best["max_depth"] == 0:
        best["max_depth"] = None

    # Evaluate on validation data
    best_model = RandomForestClassifier(random_state=42, **best)
    best_model.fit(X_train, y_train)

    # test_accuracy
    test_accuracy = best_model.score(X_val, y_val)
    print("Test Accuracy:", test_accuracy)

    # Classification report
    y_pred = best_model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

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
