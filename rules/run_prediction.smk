rule random_forest_prediction:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv",
        config_file="config/config.yml",
        random_forest_model="results/random_forest/random_forest_model.pkl"
    output:
        random_forest_classification_report="results_final_prediction/random_forest/random_forest_classification_report.csv",
        random_forest_confusion_matrix="results_final_prediction/random_forest/random_forest_confusion_matrix.png",
        # random_forest_feature_importance="results/random_forest/random_forest_feature_importance.csv",
        # random_forest_feature_importance_plot="plots/random_forest/random_forest_feature_importance.png",
        random_forest_roc_curve="results_final_prediction/random_forest/random_forest_roc_curve.png"
    log:
        "logs/random_forest_prediction.log"
    params:
        script="scripts/run_prediction.py",
        config_file="config/config.yml",
        model="random_forest"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_train_file", input.threshold_filter_data,
            "--count_test_file", input.count_test_data,
            "--metadata_test_file", input.metadata_test_data,
            "--config_file", input.config_file,
            "--model_file", input.random_forest_model,
            "--output_clasification_report", output.random_forest_classification_report,
            "--output_confusion_matrix", output.random_forest_confusion_matrix,
            "--output_roc_curve", output.random_forest_roc_curve
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule xgboost_prediction:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv",
        config_file="config/config.yml",
        xgboost_model="results/xgboost/xgboost_model.pkl"
    output:
        xgboost_classification_report="results_final_prediction/xgboost_classification_report.csv",
        xgboost_confusion_matrix="results_final_prediction/xgboost_confusion_matrix.png",
        # xgboost_feature_importance="results/xgboost/xgboost_feature_importance.csv",
        # xgboost_feature_importance_plot="plots/xgboost/xgboost_feature_importance.png",
        xgboost_roc_curve="results_final_prediction/xgboost_roc_curve.png"
    log:
        "logs/xgboost_prediction.log"
    params:
        script="scripts/run_prediction.py",
        config_file="config/config.yml",
        model="xgboost"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_train_file", input.threshold_filter_data,
            "--count_test_file", input.count_test_data,
            "--metadata_test_file", input.metadata_test_data,
            "--config_file", input.config_file,
            "--model_file", input.xgboost_model,
            "--output_clasification_report", output.xgboost_classification_report,
            "--output_confusion_matrix", output.xgboost_confusion_matrix,
            "--output_roc_curve", output.xgboost_roc_curve
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)

rule run_prediction:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv",
        config_file="config/config.yml",
        random_forest_model="results/random_forest/random_forest_model.pkl",
        xgboost_model="results/xgboost/xgboost_model.pkl",
        traning_scores="results/all_best_model_scores_CV.csv"
    output:
        classification_report="results_final_prediction/classification_report.txt",
        confusion_matrix="results_final_prediction/confusion_matrix.png",
        roc_curve="results_final_prediction/roc_curve_all.png"
    log:
        "logs/run_prediction.log"
    params:
        script="scripts/run_prediction.py",
        config_file="config/config.yml"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_train_file", input.threshold_filter_data,
            "--count_test_file", input.count_test_data,
            "--metadata_test_file", input.metadata_test_data,
            "--config_file", input.config_file,
            "--model_file_random_forest", input.random_forest_model,
            "--model_file_xgboost", input.xgboost_model,
            "--training_scores_file", input.traning_scores,
            "--output_clasification_report", output.classification_report,
            "--output_confusion_matrix", output.confusion_matrix,
            "--output_roc_curve", output.roc_curve
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)
