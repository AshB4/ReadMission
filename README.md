
# Predicting Hospital Readmissions

![Hospital Readmissions](https://www.docwirenews.com/wp-content/uploads/2019/05/readmissions.jpg)

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Information](#dataset-information)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Modeling](#modeling)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Hospital readmission rates are a critical metric in evaluating the quality of care provided by hospitals. This project aims to predict hospital readmissions, particularly focusing on diabetic patients. By identifying key factors that contribute to readmissions, hospitals can implement strategies to reduce readmission rates, improve patient care, and decrease healthcare costs.

## Dataset Information

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/dubradave/hospital-readmissions). It contains ten years of patient information with various features including age, time in hospital, number of procedures, and whether the patient was readmitted.

### Dataset Features

- `age`: Age bracket of the patient
- `time_in_hospital`: Days spent in hospital (from 1 to 14)
- `n_lab_procedures`: Number of laboratory procedures performed during the hospital stay
- `n_procedures`: Number of procedures performed during the hospital stay
- `n_medications`: Number of medications administered during the hospital stay
- `n_outpatient`: Number of outpatient visits in the year before the hospital stay
- `n_inpatient`: Number of inpatient visits in the year before the hospital stay
- `n_emergency`: Number of visits to the emergency room in the year before the hospital stay
- `medical_specialty`: Specialty of the admitting physician
- `diag_1`: Primary diagnosis
- `diag_2`: Secondary diagnosis
- `diag_3`: Additional secondary diagnosis
- `glucose_test`: Glucose serum test result (high, normal, not performed)
- `A1Ctest`: A1C test result (high, normal, not performed)
- `change`: Whether there was a change in diabetes medication (yes, no)
- `diabetes_med`: Whether a diabetes medication was prescribed (yes, no)
- `readmitted`: Whether the patient was readmitted (yes, no)

## Installation

To set up this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/AshB4/ReadMission.git
    cd ReadMission
    ```
2. **Create and activate a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```
3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running Scripts

1. **Data Cleaning**:
    ```sh
    python scripts/DataCleaning.py
    ```
2. **Exploratory Data Analysis**:
    ```sh
    python scripts/ExploreData.py
    ```
3. **Modeling**:
    ```sh
    python scripts/ModelTraining.py
    ```
4. **Evaluation**:
    ```sh
    python scripts/Evaluation.py
    ```

### Jupyter Notebooks

You can also explore the data and run the analysis step-by-step using the Jupyter notebooks located in the `notebooks/` directory. Start Jupyter Notebook with:
```sh
jupyter notebook
```

## Exploratory Data Analysis (EDA)

In the EDA notebook (`notebooks/prediction-on-hospital-readmission.ipynb`), we explore the dataset to understand the distribution of features, identify any missing values, and visualize relationships between features and the target variable (`readmitted`).

## Modeling

In the modeling notebook (`notebooks/prediction-on-hospital-readmission.ipynb`), we train various machine learning models to predict hospital readmissions. The models include:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Results

The evaluation notebook (`notebooks/prediction-on-hospital-readmission.ipynb`) presents the results of our models, including accuracy, precision, recall, and F1-score. We also discuss the importance of different features in predicting readmissions and suggest potential interventions to reduce readmission rates.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
