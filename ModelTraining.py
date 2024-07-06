# scripts/ModelTraining.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(model, f"./reports/model/{model_name}_model.pkl")


def train_models(df):
    # Define feature columns and target variable
    feature_cols = [
        "time_in_hospital",
        "n_lab_procedures",
        "n_medications",
        "n_outpatient",
        "n_inpatient",
        "n_emergency",
        "A1Ctest",
        "glucose_test",
        "change",
        "diabetes_med",
    ]
    # Add one-hot encoded columns
    feature_cols.extend(df.columns[df.columns.str.startswith("medical_specialty_")])
    feature_cols.extend(df.columns[df.columns.str.startswith("diag_1_")])
    feature_cols.extend(df.columns[df.columns.str.startswith("diag_2_")])
    feature_cols.extend(df.columns[df.columns.str.startswith("diag_3_")])

    X = df[feature_cols]
    y = df["readmitted"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    for model_name, model in models.items():
        train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

    # Save the scaler
    joblib.dump(scaler, "./reports/model/scaler.pkl")


if __name__ == "__main__":
    # Load the cleaned dataset
    csv_path = "./data/CleanedData/hospital_readmissions_cleaned.csv"
    df = pd.read_csv(csv_path)
    train_models(df)
