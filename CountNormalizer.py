import pandas as pd
import numpy as np
from rnanorm import TMM
from pydeseq2.dds import DeseqDataSet
from pydeseq2 import preprocessing as deseq2_preprocess
# import rpy2.robjects as ro
# import rpy2.robjects.pandas2ri as pd2ri

# Activate pandas conversion in rpy2
# pd2ri.activate()

class CountNormalizer:
    """A class to represent the count normalizer. Holds the methods for normalizing the count data."""

    def __init__(self, config_data: dict):
        self.config_data = config_data

    def tmm_normalize(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the TMM normalized data."""  # TODO write tmm normalization method with rnanorm
        tmm = TMM()
        tmm_normalized_data = tmm.set_output(transform="pandas").fit_transform(
            count_data
        )
        return tmm_normalized_data

        # # Step 1: Calculate library sizes (sum of counts for each row/sample)
        # lib_sizes = count_data.sum(axis=1)

        # # Step 2: Identify the reference sample (median library size)
        # ref_sample = lib_sizes.median()
        # ref_counts = count_data.loc[lib_sizes.idxmin()]

        # print(ref_counts + 1e-6)

        # # Step 3: Compute M-values and A-values
        # M_values = np.log2(
        #     # count_data.div(ref_counts + 1e-6, axis=1)
        #     count_data / ref_counts + 1e-6
        # )  # Adding a small constant to avoid log(0)
        # # A_values = 0.5 * (np.log2(count_data + 1e-6) + np.log2(ref_counts + 1e-6))

        # # print(M_values)

        # # Step 4: Trim extreme values (30% from each end)
        # trim_percent = 0.3
        # trim_n = int(count_data.shape[1] * trim_percent)

        # def trimmed_mean(x):
        #     sorted_x = np.sort(x)
        #     return sorted_x[trim_n:-trim_n].mean()

        # # Step 5: Compute TMM factors
        # tmm_factors = M_values.apply(trimmed_mean, axis=1)

        # # Step 6: Calculate normalization factors
        # normalization_factors = 2**tmm_factors

        # # Step 7: Apply scaling factors to normalize counts
        # normalized_counts = count_data.div(normalization_factors, axis=0) * ref_sample.sum()

        # print(normalized_counts)

        # return normalized_counts

    def tmm_normalize_rpy2(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the TMM normalized data using rpy2 and edgeR in R."""

        # Convert the pandas DataFrame to an R DataFrame
        r_count_data = pd2ri.py2rpy(count_data)

        # Define R code for TMM normalization
        r_code = """
        library(edgeR)
        
        # Convert the input data to a DGEList object
        dge <- DGEList(counts = count_data)
        
        # Perform TMM normalization
        dge <- calcNormFactors(dge, method = "TMM")
        
        # Return normalized counts
        normalized_counts <- cpm(dge, normalized.lib.sizes = TRUE)
        return(normalized_counts)
        """

        # Pass the count data to R and execute the code
        ro.globalenv["count_data"] = r_count_data
        ro.r(r_code)

        # Get the normalized data from R
        normalized_data_r = ro.r("normalized_counts")

        # Convert the R data back to a pandas DataFrame
        normalized_data = pd2ri.rpy2py(normalized_data_r)

        # Return the normalized data as a pandas DataFrame
        return normalized_data

    def cpm_normalize(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the CPM normalized data."""
        cpm_normalized_data = count_data.div(count_data.sum(axis=0), axis=1) * 1e6
        return cpm_normalized_data

    def vst(self, count_data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
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

    def deseq2_normalize(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the DESeq2 normalized data."""
        print("Running DESeq2 normalization...")
        deseq2_normalized_data, size_factors = deseq2_preprocess.deseq2_norm(count_data)
        # print("DESeq2 normalized count data.")
        # print(deseq2_normalized_data)

        return deseq2_normalized_data

    def normalize(self, count_data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """Returns the normalized data based on the method given."""
        normalization_methods = self.config_data["preprocessing"]["normalization_methods"]
        try:
            if normalization_methods["tmm"]["use_method"]:
                # return self.tmm_normalize_rpy2(count_data=count_data)
                return self.tmm_normalize(count_data=count_data)
            if normalization_methods["cpm"]["use_method"]:
                return self.cpm_normalize(count_data=count_data)
            if normalization_methods["vst"]["use_method"]:
                return self.vst(count_data=count_data, metadata=metadata)
            if normalization_methods["deseq2"]["use_method"]:
                return self.deseq2_normalize(count_data=count_data)
        except KeyError as e:
            raise ValueError(f"Normalization method {e} not implemented.")
