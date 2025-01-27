rule all:
    input:
        "data/count_train_data.csv",
        "data/metadata_train_data.csv",
        "data/count_test_data.csv",
        "data/metadata_test_data.csv"

# Rule to preprocess and generate the train and test data
rule preprocess:
    input:
        count_file=config["count_file"],
        metadata_file=config["metadata_file"],
        counts_holdout_test_set=config.get("count_holdout_test_set", None),
        metadata_holdout_test_set=config.get("metadata_holdout_test_set", None)
    output:
        count_train_data="data/count_train_data.csv",
        metadata_train_data="data/metadata_train_data.csv",
        count_test_data="data/count_test_data.csv",
        metadata_test_data="data/metadata_test_data.csv"
    run:
        # Prepare the command to run the external Python script
        cmd = ["python", "preprocess.py", "--count_file", input.count_file, "--metadata_file", input.metadata_file, "--count_train_data", output.count_train_data, "--metadata_train_data", output.metadata_train_data]
        
        if input.counts_holdout_test_set and input.metadata_holdout_test_set:
            cmd.append(input.counts_holdout_test_set)  # Add the test RNA file if provided
            cmd.append(output.metadata_test_data)  # Add the output file for the test RNA

        # Run the Python script
        shell(" ".join(cmd))