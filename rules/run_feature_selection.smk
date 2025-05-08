
rule random_forest:
    input:
        pre_filtered_data="data/pre_filtered_count_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest/random_forest_feature_importance.csv",
        random_forest_feature_importance_plot="results/random_forest/random_forest_feature_importance.png",
        hyperparameters_gridsearch = "results/random_forest/random_forest_hyperparameters.csv",
        gridsearch_object="results/random_forest/random_forest_gridsearch.pkl",
        best_model="results/random_forest/random_forest_model.pkl",
        random_forest_score_file="results/model_CV_scores/random_forest_best_model_scores.csv",
        random_forest_roc_curve="results/model_CV_scores/random_forest_roc_curve.png",
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
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_gridsearch", output.gridsearch_object,
            "--output_path_model", output.best_model,
            "--output_path_score", output.random_forest_score_file,
            "--output_path_roc_curve", output.random_forest_roc_curve,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")
        # shell(shell_cmd)

rule xgboost:
    input:
        pre_filtered_data="data/pre_filtered_count_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        xgboost_feature_importance="results/xgboost/xgboost_feature_importance.csv",
        xgboost_feature_importance_plot="results/xgboost/xgboost_feature_importance.png",
        hyperparameters_gridsearch = "results/xgboost/xgboost_hyperparameters.csv",
        gridsearch_object="results/xgboost/xgboost_gridsearch.pkl",
        xgboost_best_model="results/xgboost/xgboost_model.pkl",
        xgboost_score_file="results/model_CV_scores/xgboost_best_model_scores.csv",
        xgboost_roc_curve="results/model_CV_scores/xgboost_roc_curve.png",
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
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--feature_selection_method", "{params.feature_selection_method}",
            "--output_path_file", output.xgboost_feature_importance,
            "--output_path_plot", output.xgboost_feature_importance_plot,
            "--output_path_hyperparams", output.hyperparameters_gridsearch,
            "--output_path_gridsearch", output.gridsearch_object,
            "--output_path_model", output.xgboost_best_model,
            "--output_path_score", output.xgboost_score_file,
            "--output_path_roc_curve", output.xgboost_roc_curve,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

# rule plot_roc_curves:
#     input:
#         config_file="config/config.yml",
#         best_models=expand("results/{model}/{model}_model.pkl", model=["random_forest", "xgboost"]),
#     output:
#         all_roc_curves="results/all_roc_curves_CV.png",

#     params:
#         script="scripts/plot_roc_curves.py",
#         model_names = ["random_forest", "xgboost"]
#     run:
#         best_models = [str(item) for item in input.best_models]
#         # Prepare the command to run the external Python script
#         cmd = [
#             "python", "{params.script}",
#             "--config_file", input.config_file,
#             "--best_models", *best_models,
#             "--model_names", *params.model_names,
#             "--output_path_roc_curve", output.all_roc_curves,
#         ]
#         # Log the command
#         shell_cmd = " ".join(cmd)
#         print(f"Running command: {shell_cmd}")
#         # Run the command and redirect stdout and stderr to the log file
#         # shell(shell_cmd + " > logs/plot_roc_curves.log 2>&1")
#         shell(shell_cmd)


rule get_CV_scores:
    input:
        scores=expand("results/model_CV_scores/{model}_best_model_scores.csv", model=["random_forest", "xgboost"]),
        roc_curves=expand("results/model_CV_scores/{model}_roc_curve.png", model=["random_forest", "xgboost"]),
    output:
        all_model_scores="results/all_best_model_scores_CV.csv",
        all_roc_curves="results/all_roc_curves_CV.png",
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "scripts/merge_model_scores.py",
            "--random_forest_score_file", input.scores[0],
            "--xgboost_score_file", input.scores[1],
            "--all_models_score_file", output.all_model_scores,
            "--random_forest_roc_curve", input.roc_curves[0],
            "--xgboost_roc_curve", input.roc_curves[1],
            "--output_path_roc_curve", output.all_roc_curves,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > logs/merge_model_scores.log 2>&1")
        shell(shell_cmd)
