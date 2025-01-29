rule random_forest:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv"
    output:
        random_forest_feature_importance="results/random_forest_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest_feature_importance.png"
    log:
        "logs/random_forest.log"
    params:
        script="scripts/feature_selection/random_forest.py",
        n_estimators=config["feature_selection"]["random_forest"]["n_estimators"],
        max_depth=config["feature_selection"]["random_forest"]["max_depth"],
        random_state=config["feature_selection"]["random_forest"]["random_state"],

    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--n_estimators", str(params.n_estimators),
            "--max_depth", str(params.max_depth),
            "--random_state", str(params.random_state),
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")


        