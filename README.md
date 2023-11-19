# XGBoost Auto Tuner

This project implements an XGBoost Auto Tuner using Python and the Tkinter GUI toolkit. The application allows users to load a dataset, perform data preprocessing tasks, and optimize hyperparameters for an XGBoost model. The XGBoost model is trained, evaluated, and its results are displayed using SHAP (SHapley Additive exPlanations) plots.

Feature analyses
![image](https://github.com/JeroenKreuk/gui_xgboost_shap/assets/85551796/d8ead68b-38a1-4f51-9ad6-f000eff1d082)

Tkinter GUI
![image](https://github.com/JeroenKreuk/gui_xgboost_shap/assets/85551796/9ab27f4a-df06-4571-9f9d-d3eb85d2f779)


## Features

- **Dataset Loading**: Load a CSV dataset and display basic information about the data, including the total number of observations.

- **Data Preprocessing**: Select a target/label column, restrict observations randomly, and create dummy variables for categorical columns.

- **Feature Importance**: Utilize a RandomForestRegressor to calculate and display feature importances based on the selected target column.

- **Feature Selection**: Perform feature selection based on a specified importance threshold. Display the selected features and provide options for scaling and SMOTE.

- **SMOTE & Scaling**: Apply Synthetic Minority Over-sampling Technique (SMOTE) and scaling to the selected features. Options for scaling and SMOTE are available.

- **Hyperparameter Optimization**: Use Optuna to optimize hyperparameters for an XGBoost model.

- **Model Evaluation**: Train the XGBoost model with the optimized hyperparameters, perform cross-validation, and display results including cross-validation scores, mean cross-validation score, train score, test score, and feature importance.

- **SHAP Plots**: Visualize SHAP (SHapley Additive exPlanations) plots for the trained XGBoost model.

## Requirements

- Python 3.x
- Tkinter
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- optuna
- xgboost
- shap

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/JeroenKreuk/gui_xgboost_shap.git
