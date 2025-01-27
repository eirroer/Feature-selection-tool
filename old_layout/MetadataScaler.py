import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

class MetadataScaler:
    """A class to represent the metadata scaler. Holds the methods for scaling the metadata data."""

    def __init__(self, config_data: dict):
        self.config_data = config_data

    def standard_scale(self, metadata: pd.DataFrame, cols: pd.Series) -> pd.DataFrame:
        """Returns the metadata data scaled using the standard scaling method. On the columns specified."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(metadata[cols])
        standard_scaled = pd.DataFrame(
            X_scaled, columns=metadata.columns , index=metadata.index
        )
        return standard_scaled

    def min_max_scale(self, metadata: pd.DataFrame, cols: pd.Series) -> pd.DataFrame:
        """Returns the min-max scaled data."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(metadata[cols])
        min_max_scaled = pd.DataFrame(
            X_scaled, columns=metadata.columns, index=metadata.index
        )
        return min_max_scaled

    def max_abs_scale(self, metadata: pd.DataFrame, cols: pd.Series) -> pd.DataFrame:
        """Returns the max-abs scaled data."""
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(metadata[cols])
        max_abs_scaled = pd.DataFrame(
            X_scaled, columns=metadata.columns, index=metadata.index
        )
        return max_abs_scaled

    def scale(self, metadata: pd.DataFrame, cols: pd.Series) -> pd.DataFrame:
        """Returns the scaled data based on the method given."""

        scaling_methods = self.config_data["preprocessing"][
            "scaling_methods_metadata"
        ]

        try:
            if scaling_methods["standard_scale"]["use_method"]:
                return self.standard_scale(metadata=metadata, cols=cols)
            elif scaling_methods["min_max_scale"]["use_method"]:
                return self.min_max_scale(metadata=metadata, cols=cols)
            elif scaling_methods["max_abs_scale"]["use_method"]:
                return self.max_abs_scale(metadata=metadata, cols=cols)
        except KeyError as e:
            raise ValueError(f"Scaling method {e} not implemented.")

        # If no scaling method is selected, return the original count data
        return metadata
