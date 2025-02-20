rule random_forest:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/random_forest_feature_importance.csv",
        random_forest_feature_importance_plot="results/random_forest/random_forest_feature_importance.png",
        hyperparameters_gridsearch = "results/random_forest/random_forest_hyperparameters.csv",
        pickle_file="results/random_forest/random_forest_model.pkl",
        random_forest_score_file=temp("results/random_forest_best_model_scores.csv"),
        random_forest_roc_curve=temp("results/roc_curve_random_forest.png"),
    log:
        "logs/random_forest_gridsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        feature_selection_method="random_forest",
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.threshold_filter_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_model", output.pickle_file,
            "--output_path_score", output.random_forest_score_file,
            "--output_path_roc_curve", output.random_forest_roc_curve,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule xgboost:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/xgboost_feature_importance.csv",
        xgboost_feature_importance_plot="results/xgboost/xgboost_feature_importance.png",
        hyperparameters_gridsearch = "results/xgboost/xgboost_hyperparameters.csv",
        xgboost_model="results/xgboost/xgboost_model.pkl",
        xgboost_score_file=temp("results/xgboost_best_model_scores.csv"),
        xgboost_roc_curve=temp("results/roc_curve_xgboost.png"),
    log:
        "logs/xgboost_gridsearch.log"
    params:
        script="scripts/feature_selection/run_feature_selection.py",
        config_file="config/config.yml",
        feature_selection_method="xgboost"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.threshold_filter_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_model", output.xgboost_model,
            "--output_path_score", output.xgboost_score_file,
            "--output_path_roc_curve", output.xgboost_roc_curve,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule merge_model_scores:
    input:
        random_forest_score_file="results/random_forest_best_model_scores.csv",
        xgboost_score_file="results/xgboost_best_model_scores.csv",
        random_forest_roc_curve="results/roc_curve_random_forest.png",
        xgboost_roc_curve="results/roc_curve_xgboost.png",
    output:
        all_model_scores="results/all_best_model_scores_training.csv",
        all_roc_curves="results/all_roc_curves_training.png",
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "scripts/merge_model_scores.py",
            "--random_forest_score_file", input.random_forest_score_file,
            "--xgboost_score_file", input.xgboost_score_file,
            "--all_models_score_file", output.all_model_scores,
            "--random_forest_roc_curve", input.random_forest_roc_curve,
            "--xgboost_roc_curve", input.xgboost_roc_curve,
            "--output_path_roc_curve", output.all_roc_curves,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > logs/merge_model_scores.log 2>&1")
        shell(shell_cmd)
