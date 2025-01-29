import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def pca(
    count_data: pd.DataFrame,
    metadata: pd.DataFrame,
    n_components: int,
    color_by: str,
    count_file: str,
    metadata_file: str,
    output_path: str,
) -> None:
    """Perform PCA on the count data and write the results to a file in the output folder."""
    print(f"Performing PCA with {n_components} components")
    print(count_data.head())
    print(metadata.head())

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(count_data)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=count_data.index)

    print(pca_df.head())

    # check if color_by is in metadata
    if color_by in metadata.columns:
        pca_df = pca_df.merge(
            metadata[[color_by]], left_index=True, right_index=True
        )
    else:
        raise Warning(
            f"Column {color_by} not found in metadata, and could not be used to color PCA plot."
        )
    print(pca_df.head())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="PC1", y="PC2", hue=color_by, data=pca_df, palette="viridis", alpha=0.8
    )
    # add file name to title
    plt.suptitle("PCA Plot count data", fontsize=16)
    plt.title(f"Count data sourced from file: {count_file}\n Metadata sourced from file {metadata_file}" , fontsize=10)
    plt.xlabel(
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)"
    )
    plt.ylabel(
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)"
    )
    plt.legend(title=color_by)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", )


def run_pca(
    count_file: str,
    metadata_file: str,
    n_components: int,
    color_by: str,
    output_path: str,
) -> None:
    """Run PCA on the count data."""
    count_data = pd.read_csv(count_file, index_col=0, header=0, sep=";")
    metadata = pd.read_csv(metadata_file, index_col=0, header=0, sep=";")
    pca(
        count_data=count_data,
        metadata=metadata,
        n_components=n_components,
        color_by=color_by,
        count_file=count_file,
        metadata_file=metadata_file,
        output_path=output_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PCA on count data.")
    parser.add_argument("--count_file", type=str, help="The count data file.")
    parser.add_argument("--metadata_file", type=str, help="The metadata file.")
    parser.add_argument(
        "--n_components", type=int, help="The number of components for PCA."
    )
    parser.add_argument("--color_by", type=str, help="The metadata column to color by.")
    parser.add_argument("--output_path", type=str, help="The output path.")
    args = parser.parse_args()
    run_pca(
        args.count_file,
        args.metadata_file,
        args.n_components,
        args.color_by,
        args.output_path,
    )
