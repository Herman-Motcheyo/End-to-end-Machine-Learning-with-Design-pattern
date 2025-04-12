from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataInspectorStrategy(ABC):
    """
    Abstract base class for data inspection.
    """

    @abstractmethod
    def inspection(self, data: pd.DataFrame) -> None:
        """
        Inspect the data and print the results.
        """
        pass


class DataTypesInspectorStrategy(DataInspectorStrategy):
    """
    Concrete implementation of DataInspector for data type.
    """

    def inspection(self, data: pd.DataFrame) -> None:
        print("\nData Types and Non-null Counts:")
        print(data.info())


class DataSummaryInspectorStrategy(DataInspectorStrategy):
    """
    Concrete implementation of DataInspector for data summary.
    """

    def inspection(self, data: pd.DataFrame) -> None:
        print("\nData Summary:")
        print(data.describe())


class DataTypeVisualizationStrategy(DataInspectorStrategy):
    """
    Concrete implementation for visualizing data types in a bar plot.
    """

    def inspection(self, data: pd.DataFrame) -> None:
        print("\nVisualizing Data Types...")
        type_counts = data.dtypes.value_counts()
        type_counts = type_counts.reset_index()
        type_counts.columns = ['Data Type', 'Count']

        plt.figure(figsize=(8, 5))
        sns.barplot(x="Data Type", y="Count", data=type_counts, palette="viridis")
        plt.title("Distribution of Data Types in the Dataset")
        plt.ylabel("Number of Columns")
        plt.xlabel("Data Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class DataInspector:
    """
    Context class for data inspection.
    """

    def __init__(self, strategy: DataInspectorStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectorStrategy) -> None:
        self._strategy = strategy

    def inspect(self, data: pd.DataFrame) -> None:
        self._strategy.inspection(data)


if __name__ == "__main__":
    df = pd.read_csv("/home/herman/Documents/Career/ML_Project/End-to-end-ML-project-Design-pattern/data/raw/extracted/winequality-white.csv", sep=";")

    inspector = DataInspector(DataTypesInspectorStrategy())
    inspector.inspect(df)

    inspector.set_strategy(DataSummaryInspectorStrategy())
    inspector.inspect(df)

    inspector.set_strategy(DataTypeVisualizationStrategy())
    inspector.inspect(df)
