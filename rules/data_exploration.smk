rule data_exploration:
    input:
        count_train_data="data/count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv",
        # config_file="config/config.yml"
    output:
        data_exploration_report_train="results/data_exploration/data_exploration_report.csv",
    params:
        script="scripts/data_exploration.py",
    log:
        "logs/data_exploration.log"
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.count_train_data,
            "--metadata_file", input.metadata_train_data,
            # "--config_file", input.config_file,
            "--output_path_report", output.data_exploration_report_train,
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")
        # shell(shell_cmd)