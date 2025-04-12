# **End-to-End ML Project with Design Patterns**

## Description
This project aims to demonstrate the implementation of **MLOps** best practices using **Design Patterns** to build a modular and robust machine learning pipeline. It includes features for data ingestion from ZIP and CSV files, as well as univariate and bivariate data analysis. The project applies patterns such as the **Factory Pattern** and **Strategy Pattern** to structure and easily extend the pipeline.

## Objectives
- Build a modular data processing pipeline.
- Apply **Design Patterns** (Factory, Strategy) for flexible development.
- Integrate data ingestion mechanisms for ZIP and CSV files.
- Analyze and visualize data for better understanding (univariate and bivariate analysis).

## Technologies Used
- **Python 3.x**: Primary programming language.
- **Pandas**: Data manipulation.
- **Matplotlib / Seaborn**: Data visualization.
- **Design Patterns**:
  - **Factory Pattern**: For flexible ingestion of different file types.
  - **Strategy Pattern**: For data analysis and inspection.
- **Logging**: Tracking and error management in the pipeline.


## Contributing
Contributions are welcome! If you wish to improve this project, please follow these steps:

Fork the project.

- Create a branch for your feature **(git checkout -b feature/xyz)**.

- Commit your changes **(git commit -m 'Add new feature')**.

- Push the branch **(git push origin feature/xyz)**.

##### Open a Pull Request.
## Project Features

### 1. Data Analysis
The project allows for data analysis via two types of strategies:

- **Univariate Analysis**: Exploration of individual data characteristics.

- **Bivariate Analysis**: Studying relationships between two data characteristics.

The Strategy Pattern is used to apply different analysis strategies.

#### Example of Univariate Analysis:

- Go to notebook/EDA.ipynb
### 2. Data Ingestion
The project supports ingesting data from two types of files:

- **ZIP Files**: The project automatically extracts CSV files from the ZIP file into a dedicated directory.
- **CSV Files**: CSV files are ingested directly without extraction.

The **Factory Pattern** is used to manage flexible ingestion of these different file types.

#### Example usage to ingest data from a ZIP file:



```python
if __name__ == "__main__":
    # Example usage
    file_path = "End-to-end-ML-project-Design-pattern/data/raw/wine+quality.zip"
    ingestor = IngestionFactory.get_data_ingestion("zip")
    df = ingestor.ingest(file_path, prefer_filename="winequality-white.csv")
    print(df.head())
