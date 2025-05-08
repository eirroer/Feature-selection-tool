
rule read_and_split_train_test_data:
    input:
        count_file=config["count_file"],
        metadata_file=config["metadata_file"],
        count_holdout_test_set=lambda wildcards: config.get("count_holdout_test_set", []),
        metadata_holdout_test_set=lambda wildcards: config.get("metadata_holdout_test_set", []),
    output:
        count_train_data=temp("data/count_train_data.csv"),
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

# rule filter_counts_by_metadata:
#     input:
#         count_train_data="data/count_train_data.csv",
#         metadata_train_data="data/metadata_train_data.csv"
#     output:
#         filtered_count_train_data="data/filtered_count_train_data.csv"
#     log:
#         "logs/filter_on_metadata.log"
#     params:
#         script="scripts/filter_on_metadata.py",
#         filter_column=config["preprocessing"]["filter_on_metadata"]["filter_column"],
#         filter_value=config["preprocessing"]["filter_on_metadata"]["filter_value"]
#     run:
#         # Prepare the command to run the external Python script
#         cmd = [
#             "python", "{params.script}",
#             "--count_file", input.count_train_data,
#             "--metadata_file", input.metadata_train_data,
#             "--filter_column", params.filter_column,
#             "--filter_value", params.filter_value,
#             "--output_path", output.filtered_count_train_data
#         ]
#         # Log the command
#         shell_cmd = " ".join(cmd)
#         print(f"Running command: {shell_cmd}")
#         # Run the command and redirect stdout and stderr to the log file
#         shell(shell_cmd + " > {log} 2>&1")

rule pre_filter_data:
    input:
        count_train_data="data/count_train_data.csv",
        config_file="config/config.yml"
    output:
        pre_filtered_data="data/pre_filtered_count_data.csv"
    log:
        "logs/pre_filter_data.log"
    params:
        script="scripts/pre_filter.py",
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.count_train_data,
            "--config_file", input.config_file,
            "--output_path", output.pre_filtered_data
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")


rule normalize_train_count_data:
    input:
        pre_filtered_data="data/pre_filtered_count_data.csv",
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
            "--count_file", input.pre_filtered_data,
            "--metadata_file", input.metadata_train_data,
            "--normalization_methods", " ".join(params.normalization_methods),
            "--output_path", output.normalized_count_train_data
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")

        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")

rule plot_pca:
    input:
        normalized_count_train_data="data/normalized_count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
    output:
        pca_plot="results/pca_plot.png"
    log:
        "logs/plot_pca.log"
    params:
        script="scripts/plot_pca.py",
        n_components=config["preprocessing"]["pca"]["n_components"],
        color_by=config["preprocessing"]["pca"]["color_by"]
    run:
        # Prepare the command to run the external Python script
        cmd = [
            "python", "{params.script}",
            "--count_file", input.normalized_count_train_data,
            "--metadata_file", input.metadata_train_data,
            "--n_components", str(params.n_components),
            "--color_by", str(params.color_by),
            "--output_path", output.pca_plot
        ]
        # Log the command
        shell_cmd = " ".join(cmd)
        print(f"Running command: {shell_cmd}")
        # Run the command and redirect stdout and stderr to the log file
        shell(shell_cmd + " > {log} 2>&1")

