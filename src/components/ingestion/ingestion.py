import os
import pandas as pd
import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion."""

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from a given file path."""
        pass


class ZipDataIngestor(DataIngestor):
    """Concrete class to handle ingestion from ZIP files."""

    def ingest(self, file_path: str, prefer_filename: str = None) -> pd.DataFrame:
        file_path = Path(file_path)
        extract_dir = Path("../../../data/raw/extracted")
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Start ingesting data from: {file_path}")

        if file_path.suffix != ".zip":
            logger.error("Invalid file type. Expected a .zip file.")
            raise ValueError(f"Expected a zip file, got: {file_path.suffix}")

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logger.info(f"Files extracted to: {extract_dir.resolve()}")
        except zipfile.BadZipFile as e:
            logger.exception("The zip file is corrupted or invalid.")
            raise ValueError("Invalid zip file.") from e

        csv_files = list(extract_dir.glob("*.csv"))

        if not csv_files:
            logger.error("No CSV file found after extraction.")
            raise FileNotFoundError("No CSV file found in the extracted ZIP.")

        selected_csv = None
        if prefer_filename:
            for file in csv_files:
                if file.name == prefer_filename:
                    selected_csv = file
                    break
            if not selected_csv:
                logger.warning(f"Preferred file '{prefer_filename}' not found. Falling back to first CSV.")
                selected_csv = csv_files[0]
        else:
            if len(csv_files) > 1:
                logger.warning(f"Multiple CSVs found: {[f.name for f in csv_files]}")
                logger.info("No preferred file specified. Selecting the first CSV file.")
            selected_csv = csv_files[0]

        try:
            df = pd.read_csv(selected_csv)
            logger.info(f"Successfully loaded data from: {selected_csv.name}")
        except Exception as e:
            logger.exception("Failed to read the CSV file.")
            raise e

        return df


# Custom method to ingest CSV files directly
class CsvDataIngestor(DataIngestor):

    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".csv"):
            raise ValueError("Your file is not a csv file")
        
        df = pd.read_csv(file_path)
        return df
    

class IngestionFactory:
    @staticmethod
    def get_data_ingestion(extension :str) -> DataIngestor :
        if extension =="zip":
            return ZipDataIngestor()
        
        elif extension == "csv":
            return CsvDataIngestor()
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        


if __name__ == "__main__":
    # Example usage
    file_path = "/home/herman/Documents/Career/ML_Project/End-to-end-ML-project-Design-pattern/data/raw/wine+quality.zip"
    ingestor = IngestionFactory.get_data_ingestion("zip")
    df = ingestor.ingest(file_path, prefer_filename="winequality-white.csv")
    print(df.head())
