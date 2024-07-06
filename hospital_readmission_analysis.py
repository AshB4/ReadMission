import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load the dataset
file_path = "../data/CleanedData/hospital_readmissions_cleaned.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Display the summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Convert age brackets to numeric values (e.g., '[70-80)' to 75)
age_mapping = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}
df["age"] = df["age"].map(age_mapping)

# Plot distributions of key variables
plt.figure(figsize=(10, 6))
sns.histplot(df["readmitted"], kde=False, bins=30)
plt.title("Distribution of Readmissions")
plt.xlabel("Readmitted")
plt.ylabel("Count")
plt.savefig("../reports/figures/distribution_of_readmissions.png")
plt.show()

# Correlation matrix to identify relationships between variables
plt.figure(figsize=(20, 16))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', annot_kws={"size": 8})
plt.title("Correlation Matrix")
plt.savefig("../reports/figures/correlation_matrix.png")
plt.show()

# Select relevant features and target variable
features = ["age", "time_in_hospital", "n_lab_procedures", "n_medications",
            "n_inpatient", "glucose_test", "diabetes_med"]
X = df[features]
y = df["readmitted"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Visualize the coefficients
coefficients = pd.DataFrame(model.coef_[0], X.columns, columns=["Coefficient"])
coefficients.plot(kind="barh")
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.savefig("../reports/figures/logistic_regression_coefficients.png")
plt.show()
