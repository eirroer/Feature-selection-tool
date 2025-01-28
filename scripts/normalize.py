import pandas as pd
from rnanorm import TMM
from pydeseq2.dds import DeseqDataSet
from pydeseq2 import preprocessing as deseq2_preprocess


def tmm_normalize(count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the TMM normalized data."""  # TODO write tmm normalization method without rnanorm
        tmm = TMM()
        tmm_normalized_data = tmm.set_output(transform="pandas").fit_transform(
            count_data
        )
        return tmm_normalized_data

def cpm_normalize(count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the Counts Per Million normalized data."""
        cpm_normalized_data = count_data.div(count_data.sum(axis=0), axis=1) * 1e6
        return cpm_normalized_data

def vst(count_data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """Returns the VST normalized data."""
        print("Running VST normalization...")
        # Make sure your metadata contains the condition information
        # Assumes the metadata has a 'condition' column to specify sample conditions
        if "condition" not in metadata.columns:
            raise ValueError("Metadata must contain a 'condition' column")

        dds = DeseqDataSet(counts=count_data, metadata=metadata, design="~condition", quiet=True)
        dds.vst_fit(use_design=False)
        vst_counts = dds.vst_transform()  # Apply the VST
        vst_normalized_data = pd.DataFrame(
            vst_counts, index=count_data.index, columns=count_data.columns
        )

        return vst_normalized_data

def deseq2_normalize(count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the DESeq2 normalized data."""
        print("Running DESeq2 normalization...")
        deseq2_normalized_data, size_factors = deseq2_preprocess.deseq2_norm(count_data)
        # print("DESeq2 normalized count data.")
        # print(deseq2_normalized_data)

        return deseq2_normalized_data

def write_to_file(count_data, output_path):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count_data.to_csv(output_path, sep=";", index=True, header=True)


def run_normalization(count_data_file, metadata_file, normalization_methods, output_path):
    """Returns the normalized data based on the method given."""

    print("Running normalization...")
    print(f"Count data file: {count_data_file}")
    print(f"Metadata file: {metadata_file}")
    print(f"Normalization methods: {normalization_methods}")
    print(f"Output path: {output_path}")

    count_data = pd.read_csv(count_data_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    if normalization_methods:
        if 'tmm' in normalization_methods:
            count_data = tmm_normalize(count_data)
        if 'cpm' in normalization_methods:
            count_data = cpm_normalize(count_data)
        if 'vst' in normalization_methods:
            count_data = vst(count_data, metadata)
        if 'deseq2' in normalization_methods:
            count_data = deseq2_normalize(count_data)

    write_to_file(count_data, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize RNA count data.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--normalization_methods", nargs='+', type=str, required=True, help="Normalization methods to use.")
    parser.add_argument("--output_path", required=True, help="Path to save the normalized data.")
    args = parser.parse_args()

    run_normalization(
        count_data_file=args.count_file,
        metadata_file=args.metadata_file,
        normalization_methods=args.normalization_methods,
        output_path=args.output_path
    )
