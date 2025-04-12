from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =====================================================
# Abstract Base Class for Univariate Analysis Strategy
# =====================================================
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None
        """
        pass


# =============================================
# Concrete Strategy for Numerical Feature
# =============================================
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using histogram and KDE.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The numerical column to be analyzed.

        Returns:
        None
        """
        print(f"\nAnalyzing numerical feature: {feature}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =============================================
# Concrete Strategy for Categorical Feature
# =============================================
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the frequency distribution of a categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The categorical column to be analyzed.

        Returns:
        None
        """
        print(f"\nAnalyzing categorical feature: {feature}")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=feature, palette="pastel")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


# ============================
# Context Class for Analyzer
# ============================
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy = None):
        """
        Initialize the UnivariateAnalyzer with an optional strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy, optional): The analysis strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Set a new analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be applied.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Execute the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The feature to be analyzed.
        """
        if self._strategy:
            self._strategy.analyze(df, feature)
        else:
            print("No strategy defined. Please set a strategy first.")

    def analyze_all(self, df: pd.DataFrame):
        """
        Perform univariate analysis on all features in the dataframe,
        automatically choosing the strategy based on the data type.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        """
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self.set_strategy(NumericalUnivariateAnalysis())
            else:
                self.set_strategy(CategoricalUnivariateAnalysis())

            print(f"\n--- Analyzing column: {column} ---")
            self.execute_analysis(df, column)
            print(f"--- Finished analyzing column: {column} ---")