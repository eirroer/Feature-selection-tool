import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from Data import Data


class FeatureSelector:
    """A class to represent the feature selector. Holds the methods for selecting the features."""

    def __init__(self, data: Data):
        self.data = data

    def plot_feature_importance(
        self,
        top_n_features_names: list,
        top_n_features_importances: list,
        model_name: str = "Untitled",
    ) -> None:
        # Reverse the order of the features and their importances
        top_n_features_names = top_n_features_names[::-1]
        top_n_features_importances = top_n_features_importances[::-1]

        # Ensure the directory exists
        filename = f"outputs/{model_name}_feature_importances.png"
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure()
        plt.title(f"Feature Importances(model = {model_name})")
        plt.barh(
            range(len(top_n_features_importances)),
            top_n_features_importances,
            align="center",
        )
        plt.yticks(range(len(top_n_features_importances)), top_n_features_names)
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, format="png")

    def random_forest(self):
        X = self.data.count_train_data
        y = self.data.train_targets

        logging.info("Running Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        importances = rf_model.feature_importances_

        # Rank features by importance
        indices = np.argsort(importances)[::-1]

        # top features indices
        top_n_features = indices[:10]

        # Select the top N features
        X_selected = X.iloc[:, top_n_features]
        top_features_imporances = importances[top_n_features]

        self.plot_feature_importance(
            top_n_features_names=X_selected.columns[:10],
            top_n_features_importances=top_features_imporances,
            model_name="RandomForest",
        )

    def xgboost(self):
        X = self.data.count_train_data
        y = self.data.train_targets

        logging.info("Running XGBoost...")

        # Convert the DataFrame to numeric types
        X = X.apply(pd.to_numeric, errors="coerce")

        xgb_model = XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss"
        )
        xgb_model.fit(X, y)
        importances = xgb_model.feature_importances_

        # Rank features by importance
        indices = np.argsort(importances)[::-1]

        # Select top features
        top_n_features = indices[:10]
        top_features_importances = importances[top_n_features]
        all_features = np.asarray(self.data.count_train_data.columns)
        selected_feature_names = all_features[top_n_features]

        self.plot_feature_importance(
            top_n_features_names=selected_feature_names,
            top_n_features_importances=top_features_importances,
            model_name="XGBoost",
        )

    def lasso(self):
        raise NotImplementedError("Lasso feature selection not implemented yet.")
    
    def recursive_feature_elimination(self):
        raise NotImplementedError("Recursive Feature Elimination not implemented yet.")

