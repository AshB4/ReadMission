# scripts/ExploreData.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder


def explore_data(df):
    # Display the first few rows of the DataFrame
    print("First few rows of the DataFrame:")
    print(df.head())

    # Display basic information about the DataFrame
    print("DataFrame information:")
    print(df.info())

    # Display basic statistics for numerical columns
    print("Basic statistics:")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values:")
    print(missing_values)

    # Convert categorical variables
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "yes" else 0)
    df["change"] = df["change"].apply(lambda x: 1 if x == "yes" else 0)
    df["diabetes_med"] = df["diabetes_med"].apply(lambda x: 1 if x == "yes" else 0)

    # Label encoding for ordinal categorical variables
    label_encoder = LabelEncoder()
    df["A1Ctest"] = label_encoder.fit_transform(df["A1Ctest"])
    df["glucose_test"] = label_encoder.fit_transform(df["glucose_test"])

    # One-hot encoding for nominal categorical variables
    df = pd.get_dummies(df, columns=["medical_specialty", "diag_1", "diag_2", "diag_3"])

    # Directory to save plots
    plot_dir = "./reports/figures/"

    # Create directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Visualize the distribution of readmissions
    plt.figure(figsize=(10, 6))
    sns.countplot(x="readmitted", data=df)
    plt.title("Distribution of Readmissions")
    plt.savefig(f"{plot_dir}distribution_of_readmissions.png")
    plt.show()

    # Explore the relationship between diabetes medication and readmissions
    plt.figure(figsize=(10, 6))
    sns.countplot(x="diabetes_med", hue="readmitted", data=df)
    plt.title("Readmissions by Diabetes Medication")
    plt.savefig(f"{plot_dir}readmissions_by_diabetes_medication.png")
    plt.show()

    # Explore the relationship between A1Ctest results and readmissions
    plt.figure(figsize=(10, 6))
    sns.countplot(x="A1Ctest", hue="readmitted", data=df)
    plt.title("Readmissions by A1C Test Results")
    plt.savefig(f"{plot_dir}readmissions_by_A1C_test_results.png")
    plt.show()

    # Explore the relationship between glucose test results and readmissions
    plt.figure(figsize=(10, 6))
    sns.countplot(x="glucose_test", hue="readmitted", data=df)
    plt.title("Readmissions by Glucose Test Results")
    plt.savefig(f"{plot_dir}readmissions_by_glucose_test_results.png")
    plt.show()

    # Pairplot for selected features
    selected_features = [
        "time_in_hospital",
        "n_lab_procedures",
        "n_medications",
        "diabetes_med",
        "readmitted",
    ]
    sns.pairplot(df[selected_features], hue="readmitted")
    plt.savefig(f"{plot_dir}pairplot_selected_features.png")
    plt.show()

    return df


if __name__ == "__main__":
    # Example usage
    csv_path = "./data/hospital_readmissions.csv"  # Update this path if necessary
    df = pd.read_csv(csv_path)
    df_cleaned = explore_data(df)
