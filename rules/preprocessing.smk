
rule read_and_split_train_test_data:
    input:
        count_file=config["count_file"],
        metadata_file=config["metadata_file"],
        count_holdout_test_set=lambda wildcards: config.get("count_holdout_test_set", []),
        metadata_holdout_test_set=lambda wildcards: config.get("metadata_holdout_test_set", []),
    output:
        count_train_data="data/count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv"
    log:
        "logs/read_and_split_train_test_data.log"
    params:
        script="scripts/read_and_split_train_test_data.py",
        test_size=config["preprocessing"]["train_test_split_params"]["test_size"],
        random_state=config["preprocessing"]["train_test_split_params"]["random_state"]
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.count_file,
            "--metadata_file", input.metadata_file,
            "--count_train_output_path", output.count_train_data,
            "--metadata_train_output_path", output.metadata_train_data,
            "--count_test_output_path", output.count_test_data,
            "--metadata_test_output_path", output.metadata_test_data,
            "--test_size", str(params.test_size),
            "--random_state", str(params.random_state),
        ]
   
        # Add optional inputs if provided
        if input.count_holdout_test_set and input.metadata_holdout_test_set:
            cmd += ["--count_holdout_test_set", input.count_holdout_test_set]
            cmd += ["--metadata_holdout_test_set", input.metadata_holdout_test_set]


        # Run the command and redirect stdout and stderr to the log file
        shell(" ".join(cmd) + " > {log} 2>&1")

rule threshold_filter_data:
    input:
        count_train_data="data/count_train_data.csv"
    output:
        threshold_filtered_data="data/threshold_filtered_count_train_data.csv"
    log:
        "logs/threshold_filter_data.log"
    params:
        script="scripts/threshold_filter.py",
        min_count=config["preprocessing"]["threshold_filter_params"]["min_count"],
        min_samples=config["preprocessing"]["threshold_filter_params"]["min_samples"],
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.count_train_data,
            "--min_count", str(params.min_count),
            "--min_samples", str(params.min_samples),
            "--output_path", output.threshold_filtered_data
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")

rule normalize_train_count_data:
    input:
        threshold_filter_data="data/threshold_filtered_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv"
    output:
        normalized_count_train_data="data/normalized_count_train_data.csv"
    log:
        "logs/normalize_train_count_data.log"
    params:
        script="scripts/normalize.py",
        # Get the normalization methods from the config file and filter out the methods that are not used
        normalization_methods=[method for method, is_use in config["preprocessing"]["normalization_methods"].items() if is_use["use_method"]]
    run:
        print(f"normalization_methods = {params.normalization_methods}")

        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.threshold_filter_data,
            "--metadata_file", input.metadata_train_data,
            "--normalization_methods", " ".join(params.normalization_methods),
            "--output_path", output.normalized_count_train_data
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")

        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")

rule pre_filter_data:
    input:
        normalized_count_train_data="data/normalized_count_train_data.csv"
    output:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv"
    log:
        "logs/pre_filter_data.log"
    params:
        script="scripts/pre_filter.py",
        min_samples=config["preprocessing"]["pre_filter_params"]["min_samples"],
        min_features=config["preprocessing"]["pre_filter_params"]["min_features"]
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.normalized_count_train_data,
            "--min_samples", str(params.min_samples),
            "--min_features", str(params.min_features),
            "--output_path", output.pre_filtered_data
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")
