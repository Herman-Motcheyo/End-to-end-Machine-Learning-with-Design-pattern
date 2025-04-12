from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis
# ------------------------------------------------
# This class defines a template for missing values analysis.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing them.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.
        """
        pass


# Concrete Class for Missing Values Identification and Visualization
# -------------------------------------------------------------------
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count and percentage of missing values per column.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if missing.empty:
            print("\n No missing values found.")
        else:
            print("\n🔍 Missing Values Summary:")
            missing_percent = (missing / len(df)) * 100
            result = pd.DataFrame({
                "Missing Values": missing,
                "Percentage (%)": missing_percent.round(2)
            })
            print(result)

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Displays a heatmap and bar plot of missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.
        """
        if df.isnull().sum().sum() == 0:
            print("\n No missing values to visualize.")
            return

        print("\n Visualizing Missing Values...")

        # Heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("🔬 Heatmap of Missing Values", fontsize=14)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.tight_layout()
        plt.show()

        # Bar Plot
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.index, y=missing.values, palette="flare")
        plt.xticks(rotation=45)
        plt.ylabel("Count of Missing Values")
        plt.title(" Missing Values by Column", fontsize=14)
        plt.tight_layout()
        plt.show()
