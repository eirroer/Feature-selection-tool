rule random_forest:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        config_file="config/config.yml"
    output:
        random_forest_feature_importance="results/random_forest_feature_importance.csv",
        random_forest_feature_importance_plot="plots/random_forest_feature_importance.png"
    log:
        "logs/random_forest.log"
    params:
        script="scripts/feature_selection/random_forest.py",
        config_file="config/config.yml",
        
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python ", "{params.script}",
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--config_file", input.config_file,
            "--output_path_file", output.random_forest_feature_importance,
            "--output_path_plot", output.random_forest_feature_importance_plot,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        # shell(shell_cmd + " > {log} 2>&1")
        shell(shell_cmd)


        