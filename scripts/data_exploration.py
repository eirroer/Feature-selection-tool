
def get_data_composition(
    count_file: str,
    metadata_file: str,
    output_path_report: str,

) -> None:
    """Get the data composition of the count and metadata files."""
    import pandas as pd
    import os

    # Read the count data
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    # Print the data composition
    print(f"Number of samples: {count_data.shape[1]}")
    print(f"Number of features: {count_data.shape[0]}")
    print(f"Number of metadata columns: {metadata.shape[1]}")
    print(f'Metadata columns: {metadata.columns.tolist()}')


    metadata_counts = []

    # for col in metadata.columns:
    for col in ["sex", "condition"]:
        counts = metadata[col].value_counts(dropna=False)
        total = len(metadata[col])  # Total rows in the column (including NaNs)
        
        for val, count in counts.items():
            percentage = (count / total) * 100
            metadata_counts.append({
                "Metadata_col_name": col,
                "Value": val,
                "Count": count,
                "Percentage": round(percentage, 2)  # Rounded to 2 decimals
            })


    metadata_counts_df = pd.DataFrame(metadata_counts)
    print(metadata_counts_df)
    # os.makedirs(os.path.dirname(output_path_report), exist_ok=True)
    # metadata_counts_df.to_csv(output_path_report, index=False)


    metadata_condition_by_sex = []

    grouped = metadata.groupby("sex")["condition"].value_counts(dropna=False)
    for (sex_val, condition_val), count in grouped.items():
        total = metadata[metadata["sex"] == sex_val].shape[0]
        percentage = (count / total) * 100
        metadata_condition_by_sex.append({
            "Sex": sex_val,
            "Condition": condition_val,
            "Count": count,
            "Percentage": round(percentage, 2)
        })

    metadata_condition_by_sex_df = pd.DataFrame(metadata_condition_by_sex)
    print(metadata_condition_by_sex_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get the data composition of the count and metadata files.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--output_path_report", required=True, help="Path to save the report.")
    # parser.add_argument("--output_path_plot", required=True, help="Path to save the plot.")
    args = parser.parse_args()

    get_data_composition(
        count_file=args.count_file,
        metadata_file=args.metadata_file,
        output_path_report=args.output_path_report,
        # output_path_plot=args.output_path_plot,
    )
