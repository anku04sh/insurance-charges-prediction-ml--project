# Insurance Charges Prediction using Machine Learning

A machine learning project that analyzes health insurance customer data and builds a regression model to predict insurance charges.

## Project Overview

The objective of this project is to understand the factors associated with medical insurance costs and use those features to build a predictive machine learning model. The project covers exploratory data analysis, data cleaning, feature engineering, categorical encoding, feature scaling, model training, and model evaluation.

## Dataset

The dataset contains 1,338 records and 7 columns:

- `age` - age of the insurance beneficiary
- `sex` - gender
- `bmi` - body mass index
- `children` - number of children/dependents covered by insurance
- `smoker` - smoking status
- `region` - residential region
- `charges` - medical insurance charges, used as the target variable

The raw dataset is included in this repository for reproducibility.

## Machine Learning Workflow

1. Import and inspect the dataset
2. Check data types and missing values
3. Check and remove duplicate records
4. Perform exploratory data analysis
5. Analyze numerical relationships and correlations
6. Engineer BMI categories
7. Encode categorical variables using one-hot encoding
8. Separate features and target variable
9. Split the data into training and testing sets
10. Standardize numerical features using `StandardScaler`
11. Train a Linear Regression model
12. Evaluate the model using R² and Adjusted R²

## Model

**Linear Regression** is used as the baseline regression algorithm because the target variable, insurance `charges`, is continuous.

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn

## Project Structure

```text
insurance-charges-prediction-ml--project/
├── project1(2).py    # EDA, preprocessing, feature engineering, training and evaluation
├── i(2).csv          # Raw insurance dataset
└── README.md         # Project documentation
```

## Key Concepts Demonstrated

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Duplicate handling
- Feature engineering
- One-hot encoding
- Feature scaling
- Train-test split
- Regression modelling
- Model evaluation
- R² and Adjusted R²

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/anku04sh/insurance-charges-prediction-ml--project.git
cd insurance-charges-prediction-ml--project
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the Python project

```bash
python "project1(2).py"
```

Make sure the dataset file `i(2).csv` is in the same directory as the Python script.

## Learning Outcome

This project demonstrates an end-to-end beginner-to-intermediate machine learning workflow, from raw tabular data exploration and preprocessing to regression model training and evaluation. It also provides practical experience with pandas, visualization libraries, categorical feature handling, scaling, and scikit-learn.


