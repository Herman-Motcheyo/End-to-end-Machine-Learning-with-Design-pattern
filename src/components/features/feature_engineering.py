import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


# Concrete Strategy for Log Transformation
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Custom feature creation function
def FE(X):
    X['total_acid'] = X['fixed_acidity'] + X['volatile_acidity'] + X['citric_acid']
    X['acid / density'] = X['total_acid'] / X['density']
    X['alcohol_density'] = X['alcohol'] * X['density']
    X['sulphate/density'] = X['total_acid'] / X['density']
    X['sulphates/acid'] = X['sulphates'] / X['volatile_acidity']
    X['sulphates/chlorines'] = X['sulphates'] / X['chlorides']
    X['sulphates*alcohol'] = X['sulphates'] * X['alcohol']
    return X

# Example visualization usage (commented out for integration purposes)
# X = train_df.copy()
# X = FE(X)
# new_cols = X.columns.difference(train_df.columns)

# columns = new_cols
# n_cols = 4
# n_rows = math.ceil(len(columns)/n_cols)
# fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
# ax = ax.flatten()

# for i, column in enumerate(columns):
#     plot_axes = [ax[i]]
#     sns.boxplot(
#         y=X[column],
#         x=X.quality,
#         ax=ax[i],
#         palette=palette_rgb
#     )
#     ax[i].set_title(f'{column} Distribution')
#     ax[i].set_xlabel(None)

# for i in range(i+1, len(ax)):
#     ax[i].axis('off')

# plt.tight_layout()
