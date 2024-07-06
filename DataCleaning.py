# scripts/DataCleaning.py
import pandas as pd


def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Display missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # Option 1: Drop rows with missing values
    df = df.dropna()

    # Option 2: Fill missing values with a specific value (e.g., mean, median, mode)
    # df["column_name"] = df["column_name"].fillna(df["column_name"].mean())

    # Display data types
    print("Data types before conversion:")
    print(df.dtypes)

    # Convert data types if necessary
    # df["column_name"] = df["column_name"].astype("int")
    # df["date_column"] = pd.to_datetime(df["date_column"])

    # Check for duplicates
    print("Number of duplicates before removal:")
    print(df.duplicated().sum())

    # Remove duplicates
    df = df.drop_duplicates()

    print("Data cleaning completed.")
    return df
