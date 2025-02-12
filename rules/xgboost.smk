rule xgboost_gridsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/XGB_gridsearch_feature_importance.csv",
        xgboost_feature_importance_plot="plots/xgboost/XGB_gridsearch_feature_importance.png",
        hyperparameters_gridsearch = "results/xgboost/XGB_gridsearch_hyperparameters.csv",
    log:
        "logs/xgboost_gridsearch.log"
    params:
        script="scripts/feature_selection/xgboost_fs.py",
        config_file="config/config.yml",
        method="gridsearch"
    
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--method", "{params.method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
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
    log:
        "logs/xgboost_randomsearch.log"
    params:
        script="scripts/feature_selection/xgboost_fs.py",
        config_file="config/config.yml",
        method="randomsearch"
        
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--method", "{params.method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_randomsearch,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule xgboost_hyperopt:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/XGB_hyperopt_feature_importance.csv",
        xgboost_feature_importance_plot="plots/xgboost/XGB_hyperopt_feature_importance.png",
        hyperparameters_hyperopt = "results/xgboost/XGB_hyperopt_hyperparameters.csv",
    log:
        "logs/xgboost_hyperopt.log"
    params:
        script="scripts/feature_selection/xgboost_fs.py",
        config_file="config/config.yml",
        method="hyperopt"
        
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--method", "{params.method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_hyperopt,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)