
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

class CountScaler:
    """A class to represent the count scaler. Holds the methods for scaling the count data."""


    def __init__(self, config_data: dict):
        self.config_data = config_data

    def standard_scale(self, count_data: pd.DataFrame,) -> pd.DataFrame:
        """Returns the standard scaled data."""
        scaler = StandardScaler()
        scaler.fit(count_data)
        X_scaled = scaler.transform(count_data) 
        X_scaled = pd.DataFrame(
            X_scaled, columns=count_data.columns, index=count_data.index
        )
        return X_scaled
    
    def min_max_scale(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the min-max scaled data."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(count_data)
        X_scaled = pd.DataFrame(
            X_scaled, columns=count_data.columns, index=count_data.index
        )
        return X_scaled

    def max_abs_scale(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the max-abs scaled data."""
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(count_data)
        X_scaled = pd.DataFrame(
            X_scaled, columns=count_data.columns, index=count_data.index
        )
        return X_scaled

    def scale(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Returns the scaled data based on the method given."""
        
        scaling_methods = self.config_data["preprocessing"]["scaling_methods"]

        try:
            if scaling_methods["standard_scale"]["use_method"]:
                return self.standard_scale(count_data=count_data)
            elif scaling_methods["min_max_scale"]["use_method"]:
                return self.min_max_scale(count_data=count_data)
            elif scaling_methods["max_abs_scale"]["use_method"]:
                return self.max_abs_scale(count_data=count_data)
        except KeyError as e:
            raise ValueError(f"Scaling method {e} not implemented.")
