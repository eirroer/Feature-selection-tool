rule xgboost_gridsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/XGB_gridsearch_feature_importance.csv",
        xgboost_feature_importance_plot="plots/xgboost/XGB_gridsearch_feature_importance.png",
        hyperparameters_gridsearch = "results/xgboost/XGB_gridsearch_hyperparameters.csv",
        xgboost_model="results/xgboost/XGB_gridsearch_model.pkl"
    log:
        "logs/xgboost_gridsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        hyperparameter_optimization_method="gridsearch",
        feature_selection_method="xgboost"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--hyperparameter_optimization_method", "{params.hyperparameter_optimization_method}",
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_model", output.xgboost_model
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule xgboost_randomsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/XGB_randomsearch_feature_importance.csv",
        xgboost_feature_importance_plot="plots/xgboost/XGB_randomsearch_feature_importance.png",
        hyperparameters_randomsearch = "results/xgboost/XGB_randomsearch_hyperparameters.csv",
        xgboost_model="results/xgboost/XGB_randomsearch_model.pkl"
    log:
        "logs/xgboost_randomsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        hyperparameter_optimization_method="randomsearch",
        feature_selection_method="xgboost"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--hyperparameter_optimization_method", "{params.hyperparameter_optimization_method}",
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_randomsearch,
            "--output_path_model", output.xgboost_model
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)


rule random_forest_gridsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/RF_gridsearch_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest/RF_gridsearch_feature_importance.png",
        hyperparameters_gridsearch = "results/random_forest/RF_gridsearch_hyperparameters.csv",
        pickle_file="results/random_forest/RF_gridsearch_model.pkl"
    log:
        "logs/random_forest_gridsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        hyperparameter_optimization_method="gridsearch",
        feature_selection_method="random_forest"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--hyperparameter_optimization_method", "{params.hyperparameter_optimization_method}",
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_model", output.pickle_file
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule random_forest_randomsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/RF_randomsearch_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest/RF_randomsearch_feature_importance.png",
        hyperparameters_randomsearch = "results/random_forest/RF_randomsearch_hyperparameters.csv",
        pickle_file="results/random_forest/RF_randomsearch_model.pkl"
    log:
        "logs/random_forest_randomsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        hyperparameter_optimization_method="randomsearch",
        feature_selection_method="random_forest"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--hyperparameter_optimization_method", "{params.hyperparameter_optimization_method}",
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_randomsearch,
            "--output_path_model", output.pickle_file
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)