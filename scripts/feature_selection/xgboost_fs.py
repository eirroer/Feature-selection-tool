import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval


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

def xgboost_gridsearch(   
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Perform grid search with XGBoost."""

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["xgboost"]["param_grid_gridsearch"]
    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    xgb_model = XGBClassifier(eval_metric="logloss")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    df_best = pd.DataFrame(
            list(grid_search.best_params_.items()), columns=["parameter", "value"]
        )
    # Save the best hyperparameters to a file
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    best_model = grid_search.best_estimator_

    # show auc score
    print(f"Best AUC score: {grid_search.best_score_}")

    # Extract the feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )
    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("XGBoost grid search completed.")

def xgboost_randomsearch(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Perform random search with XGBoost."""

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["xgboost"]["param_grid_randomsearch"]
    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    xgb_model = XGBClassifier(eval_metric="logloss")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    random_search.fit(X, y)

    print(f"Best parameters: {random_search.best_params_}")
    df_best = pd.DataFrame(
            list(random_search.best_params_.items()), columns=["parameter", "value"]
        )
    # Save the best hyperparameters to a file
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    best_model = random_search.best_estimator_

    # show auc score
    print(f"Best AUC score: {random_search.best_score_}")

    # Extract the feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )
    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("XGBoost random search completed.")

def xgboost_hyperopt(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
    output_path_hyperparams,
):
    """Perform hyperopt search with XGBoost."""
    # raise NotImplementedError("Hyperopt search with XGBoost is not implemented yet.")
    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_space = config_data["feature_selection"]["xgboost"]["space_params_hyperopt"]

    space = {}
    for param_name, param_details in param_space.items():
        param_type = param_details["parameter_type"]
        values = param_details["values"]
        print(param_name, values)
        if param_type == "choice":
            space[param_name] = hp.choice(param_name, values)
        elif param_type == "quniform":
            space[param_name] = hp.quniform(param_name, *values)
        elif param_type == "uniform":
            # print(param_name, values)
            space[param_name] = hp.uniform(param_name, *values)
        elif param_type == "fixed":
            space[param_name] = list(values)
        else:
            raise ValueError(f"Invalid parameter type: {param_type}")
        
    # print(f"Parameters: {space}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    dtrain_clf = xgb.DMatrix(X, label=y)
    

    def objective(space):
        # print(f"Training with parameters: {params}")
        # print(f'type of params: {type(params)}')
        # xgb_model = XGBClassifier(*params, eval_metric="logloss")
        # cv_results = xgb_model.fit(X, y)
        # auc_score = cv_results.score(X, y)
        # return {"loss": -auc_score, "status": STATUS_OK}
    
        results = xgb.cv(space, 
                   dtrain=dtrain_clf, #DMatrix (xgboost specific)
                   num_boost_round=100, 
                   nfold=5, 
                   stratified=True,  
                   early_stopping_rounds=20,
                   metrics = ['logloss','auc','aucpr','error'])
  
        best_score = results['test-auc-mean'].max()
        return {'loss':-best_score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=2, trials=trials)
    best_params = space_eval(param_space, best)
    print(f"Best parameters: {best_params}")

    df_best = pd.DataFrame(
            list(best_params.items()), columns=["parameter", "value"]
        )
    # Save the best hyperparameters to a file
    os.makedirs(os.path.dirname(output_path_hyperparams), exist_ok=True)
    df_best.to_csv(output_path_hyperparams, sep=";", index=False, header=True)

    best_model = XGBClassifier(**best_params, eval_metric="logloss")
    best_model.fit(X, y)

    # show auc score
    auc_score = best_model.score(X, y)
    print(f"Best AUC score: {auc_score}")

    # Extract the feature importance
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )
    write_feature_importance_to_file(top_features, output_path_file)
    plot_feature_importance(top_features, output_path_plot)

    print("XGBoost hyperopt search completed.") 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XGBoost feature selection.")
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
        xgboost_gridsearch(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    elif args.method == "randomsearch":
        xgboost_randomsearch(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    elif args.method == "hyperopt":
        xgboost_hyperopt(
            args.count_file,
            args.metadata_file,
            args.config_file,
            args.output_path_file,
            args.output_path_plot,
            args.output_path_hyperparams,
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")
