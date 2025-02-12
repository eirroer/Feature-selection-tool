rule random_forest_all:
    input:
        "results/random_forest/RF_hyperopt_hyperparameters.csv",
        all_RF_feature_importance=expand("results/random_forest/RF_{method}_feature_importance.csv", method=["gridsearch", "randomsearch", "hyperopt"]),
        all_RF_feature_importance_plots=expand("plots/random_forest/RF_{method}_feature_importance.png", method=["gridsearch", "randomsearch", "hyperopt"]),

rule random_forest_gridsearch:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/RF_gridsearch_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest/RF_gridsearch_feature_importance.png",
        hyperparameters_gridsearch = "results/random_forest/RF_gridsearch_hyperparameters.csv",
    log:
        "logs/random_forest_gridsearch.log"
    params:
        script="scripts/feature_selection/random_forest.py",
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
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
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
    log:
        "logs/random_forest_randomsearch.log"
    params:
        script="scripts/feature_selection/random_forest.py",
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
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_randomsearch,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule random_forest_hyperopt:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/RF_hyperopt_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest/RF_hyperopt_feature_importance.png",
        hyperparameters_hyperopt = "results/random_forest/RF_hyperopt_hyperparameters.csv",
    log:
        "logs/random_forest_hyperopt.log"
    params:
        script="scripts/feature_selection/random_forest.py",
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
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_hyperopt,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)