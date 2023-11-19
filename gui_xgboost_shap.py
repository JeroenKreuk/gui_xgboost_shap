import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog as fd, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import shap

# Create a GUI root
root = tk.Tk()

# Specify the title and dimensions to root
root.title('XGBoost auto tuner')
root.geometry('1800x1000')

#create labelframes
label_frame_input = ttk.Labelframe(root, text='Inputs', width=300, height=350)
label_frame_input.grid(row=0, column=0, sticky='n')
label_frame_dummies = ttk.Labelframe(root, text='Dummies', width=300, height=350)
label_frame_dummies.grid(row=0, column=1, sticky='n')
label_frame_feature_importance = ttk.Labelframe(root, text='Feature importance', width=300, height=350)
label_frame_feature_importance.grid(row=0, column=2, sticky='n')
label_frame_feature_selection = ttk.Labelframe(root, text='Feature selection', width=300, height=350)
label_frame_feature_selection.grid(row=0, column=3, sticky='n')
label_frame_feature_SMOTE_scaling = ttk.Labelframe(root, text='SMOTE & scaling', width=300, height=350)
label_frame_feature_SMOTE_scaling.grid(row=0, column=4, sticky='n')
label_frame_hyperparameters = ttk.Labelframe(root, text='Hyperparameters', width=300, height=350)
label_frame_hyperparameters.grid(row=0, column=5, sticky='n')
label_frame_results = ttk.Labelframe(root, text='Results', width=300, height=350)
label_frame_results.grid(row=0, column=6, sticky='n')

# Create an open file button
open_button = tk.Button(label_frame_input, text='Open database', command=lambda: OpenFile())
open_button.grid(row=0, column=0, sticky='nw')

#create column selection option
def create_label_frame_input():
    """
    Create and configure the input frame with a label and listbox for selecting the target/label column.

    Returns:
        tk.Frame: The configured input frame.
        tk.StringVar: Variable to store the selected column.
        tk.Listbox: Listbox widget for displaying available columns.
    """
    global column_selection
    global listbox_columns
    # Label for selecting target/label column
    tk.Label(label_frame_input, font="none 7 bold", text="Select target/label column:").grid(row=3, column=0, sticky='w') # place widget with empty text, will be filled later o
    # Variable to store the selected column
    column_selection = tk.StringVar()
    column_selection.set([])
    # Listbox for displaying available columns
    listbox_columns = tk.Listbox(label_frame_input, listvariable=column_selection)
    listbox_columns.grid(row=4, column=0, sticky='nw', rowspan = 10)

def OpenFile():
    """
    Open a file dialog to choose a CSV file and read the data.

    Returns:
    - str: Location of the selected database file.
    - list: Columns of the database.
    """
    global name
    global data
    global entry_observations
    name = fd.askopenfilename(initialdir="", filetypes=(("Text File", "*.csv"), ("All Files", "*.*")), title="Choose a file.")
    data = pd.read_csv(name, error_bad_lines=False)
    list(data.columns)
    column_selection.set(list(data.columns))
    tk.Button(label_frame_dummies, text='process', command=lambda: dummifying()).grid(row=0, column=0, sticky='w')
    total_rows = data.shape[0]
    tk.Label(label_frame_input, font="none 7 bold", text="Total observations: " + str(total_rows)).grid(row=14, column=0, sticky='w')  # place widget with empty text, will be filled late

    # Create an Entry widget
    tk.Label(label_frame_input, font="none 7 bold", text="Restict observations (random):").grid(row=15, column=0, sticky='w')
    entry_observations = tk.Entry(label_frame_input, text="")
    entry_observations.grid(row=16, column=0, padx=10, pady=10)

def clear_label_frame(name_label_frame):
    """
    Clear all label widgets in the specified label frame.

    Args:
    - name_label_frame (ttk.Labelframe): The label frame to be cleared.
    """
    for widget in name_label_frame.grid_slaves():
        if widget.winfo_class() == "Label":
            widget.destroy()

def dummifying():
    """
    Quick process of the database using a Chunk function and return some basic details of the selected column.

    Returns:
    - pd.DataFrame: Used database file.
    - str: Selected column.
    - float: Process time.
    - int: Observations in the database.
    - float: Average of the selected column.
    - float: Maximum value of the selected column.
    - float: Minimum value of the selected column.
    """
    clear_label_frame(label_frame_dummies)
    global data
    global X
    global y
    global X_encoded

    try:# checks if there was a column selected
        selection = listbox_columns.get(listbox_columns.curselection())
    except:
        tk.messagebox.showerror("warning", "Select column then Process database")
        return

    # check if the entry of the restrict observations is correct.
    value = int(entry_observations.get())
    if value == "":
        pass
    if isinstance(value, int) == True:
        data = data.sample(n=value, random_state=42).copy()
    else:
        tk.messagebox.showerror("Insert interger or leave empty")
        pass

    X = data.drop([selection], axis=1)
    y = data[selection]

    print("X")
    print(X.head())
    # Because we are trying to find the most significant correlations with another categorical variable ('Default'), it is very important to ensure we encode our categorical to ensure accurate feature selection.
    # One-hot encode all object (categorical) columns
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
    print("X_encoded")
    print(X_encoded.head())
    tk.Label(label_frame_dummies, font="none 7 bold", text="Target column: " + str(selection)).grid(row=2, column=0, sticky='w') # place widget with empty text, will be filled later o
    tk.Label(label_frame_dummies, font="none 7 bold", text="Dummyfied columns:").grid(row=4, column=0, sticky='w')  # place widget with empty text, will be filled later

    row_number = 8
    for dummy in X.select_dtypes(include=['object']):
        tk.Label(label_frame_dummies, font="none 7", text=str(dummy)).grid(row=row_number, column=0,sticky='w') # place widget with empty text, will be filled later
        row_number = row_number + 1

    tk.Button(label_frame_feature_importance, text='process', command=lambda: feature_importance()).grid(row=0, column=0, sticky='w')

def feature_importance():
    """
    Calculate and display feature importances using RandomForestRegressor.
    """
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_regressor.fit(X_encoded, y)
    feature_importances = rf_regressor.feature_importances_
    global importance_df
    importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index()

    for index, row in importance_df.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"],3)
        tk.Label(label_frame_feature_importance, font="none 7", text=str(index) + " " + str(importance) + " " + feature).grid(row=index +1, column=0,sticky='w') # place widget with empty text, will be filled later
    # Create a StringVar to hold the default value
    tk.Button(label_frame_feature_selection, text='process', command=lambda: feature_selection()).grid(row=0, column=0, sticky='w')
    global default_value_selection
    default_value_selection = tk.StringVar()
    default_value_selection.set(0.01)  # Set the default value to "0.01"

    # Create an Entry widget and set its textvariable to the default value
    entry = tk.Entry(label_frame_feature_selection, textvariable=default_value_selection)
    entry.grid(row=1, column=0, sticky='w')


def feature_selection():
    """
    Perform feature selection based on the specified importance threshold.

    This function retrieves the importance threshold from the default_value_selection widget,
    filters features based on the threshold, updates the global variable X_encoded_filtered,
    and displays the selected features along with their importance scores in the tkinter window.

    Additionally, it includes checkboxes for specifying whether to apply scaling and/or SMOTE (Synthetic Minority Over-sampling Technique).
    """
    # Retrieve importance threshold
    value_selection = float(default_value_selection.get())

    # Clear the label frame for feature selection
    clear_label_frame(label_frame_feature_selection)

    # Filter features based on the importance threshold
    df_selection = importance_df[importance_df['Importance'] > value_selection]
    columns_to_filter = df_selection['Feature'].tolist()

    # Update global variable X_encoded_filtered
    global X_encoded_filtered
    X_encoded_filtered = X_encoded.loc[:, columns_to_filter]

    # Display selected features in the tkinter window
    tk.Label(label_frame_feature_selection, font="none 7 bold", text="Selected features:").grid(row=2, column=0, sticky='w')
    for index, row in df_selection.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"], 3)
        tk.Label(label_frame_feature_selection, font="none 7", text=f"{index} {importance} {feature}").grid(row=index + 3, column=0, sticky='w')

    # Button to trigger further processing (e.g., SMOTE scaling)
    tk.Button(label_frame_feature_SMOTE_scaling, text='Process', command=lambda: SMOTE_scaling()).grid(row=0, column=0, sticky='w')

    # Checkboxes for scaling and SMOTE
    global checkbox_scaling_var
    checkbox_scaling_var = tk.BooleanVar(value=True)
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="Scaling", variable=checkbox_scaling_var).grid(row=2, column=0, sticky='w')

    global checkbox_SMOTE_var
    checkbox_SMOTE_var = tk.BooleanVar()
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="SMOTE", variable=checkbox_SMOTE_var).grid(row=3, column=0, sticky='w')

    # Display information about unique target values and their occurrences
    unique_features = y.unique()
    value_counts = y.value_counts()
    row = 4
    for value, count in value_counts.items():
        tk.Label(label_frame_feature_SMOTE_scaling, font="none 7", text=f"Target value {value}: {count} occurrences").grid(row=row, column=0, sticky='w')
        row += 1

    # Display the original X_encoded dataframe
    print("X_encoded")
    print(X_encoded.head())



class Dropdown:
    """
    Create a dropdown menu in the specified label frame.

    Args:
    - label_frame (ttk.Labelframe): The label frame where the dropdown will be created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - row (int): Row position in the label frame.
    - column (int): Column position in the label frame.
    - label (str, optional): Label for the dropdown.

    Attributes:
    - label_frame (ttk.Labelframe): The label frame where the dropdown is created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - combo_var (tk.StringVar): StringVar to store the selected item.
    - combo (ttk.Combobox): Combobox widget.
    """
    def __init__(self, label_frame, options, default_value, row, column, label=None):
        self.label_frame = label_frame
        self.options = options
        self.default_value = default_value

        # Create a Label if specified
        if label:
            self.label = ttk.Label(label_frame, text=label)
            self.label.grid(row=row, column=column, padx=5, pady=5)

        # Create a StringVar to store the selected item
        self.combo_var = tk.StringVar()

        # Create a Combobox widget
        self.combo = ttk.Combobox(label_frame, textvariable=self.combo_var)
        self.combo['values'] = options
        self.combo.set(default_value)
        self.combo.grid(row=row, column=column + 1, padx=5, pady=5)

    def get_selected_value(self):
        return self.combo_var.get()

def SMOTE_scaling():
    """
    Apply SMOTE and scaling to the selected features.

    This function applies SMOTE (Synthetic Minority Over-sampling Technique) and scaling to the selected features.
    It resamples the dataset using SMOTE if the checkbox_SMOTE_var is selected, and scales the data if checkbox_scaling_var is selected.

    Global Variables:
    - X_scaled: Scaled feature matrix
    - y_resampled: Resampled target variable

    Parameters:
    None

    Returns:
    None
    """
    global X_scaled
    global y_resampled
    if checkbox_SMOTE_var.get():
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_encoded_filtered, y)
        y_resampled.value_counts(normalize=True)
        print("vink staat aan")
    else:
        X_resampled, y_resampled = X_encoded_filtered, y
    if checkbox_scaling_var.get():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        print("vink scaling staat aan")
    else:
        X_scaled = X_resampled

    tk.Button(label_frame_hyperparameters, text='process & show results', command=lambda: run_models(X_scaled, y_resampled)).grid(row=0, column=0, sticky='w')
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Results XGBoost Hyperparameters:").grid(row=1, column=0, sticky='w')

def run_optimize(trial, X, y):
    """
    Optimize the XGBoost model hyperparameters using Optuna.

    Parameters:
    trial (optuna.Trial): An Optuna trial object for hyperparameter optimization.
    X_scaled: Your input features
    y_resampled: Your target variable

    Returns:
    float: The validation accuracy of the optimized model.
    """
    # Define the hyperparameter search space
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'gamma': trial.suggest_loguniform('gamma', 1e-9, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
    }

    # Create and train the XGBoost model with the current hyperparameters
    model = xgb.XGBClassifier(**xgb_params, objective='binary:logistic', random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    val_accuracy = model.score(X_val, y_val)

    return val_accuracy

def initialize_xgb_model(best_params):
    """
    Initialize an XGBoost classifier with hyperparameters provided in `best_params`.

    Parameters:
    - best_params (dict): Dictionary containing hyperparameters for the XGBoost model.

    Returns:
    xgb.XGBClassifier: Initialized XGBoost classifier.

    """
    return xgb.XGBClassifier(
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        gamma=best_params['gamma'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        objective='binary:logistic',
        random_state=42,
    )

def train_model(model, X_train, y_train):
    """
    Train the given model on the provided training data.

    Parameters:
    - model: The machine learning model to be trained.
    - X_train (array-like): Training input features.
    - y_train (array-like): Target values for training.

    Returns:
    The trained model.

    """
    model.fit(X_train, y_train)
    return model

def calculate_feature_importance(model):
    """
    Calculate the percentage importance of each feature in the trained model.

    Parameters:
    - model: Trained machine learning model.

    Returns:
    dict: Dictionary containing feature names and their percentage importance.

    """
    importance_type = 'weight'
    feature_importance = model.get_booster().get_score(importance_type=importance_type)
    total_importance = sum(feature_importance.values())
    percentage_importance = {feature: (importance / total_importance) for feature, importance in feature_importance.items()}
    return percentage_importance

def perform_cross_validation(model, X, y):
    """
    Perform cross-validation on the given model using the provided data.

    Parameters:
    - model: Machine learning model to be evaluated.
    - X (array-like): Input features for cross-validation.
    - y (array-like): Target values for cross-validation.

    Returns:
    array-like: Array of accuracy scores from cross-validation.

    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores

def explain_model_with_shap(model, X, instance_index=0, feature_index='auto'):
    """
    Explain the model predictions using SHAP (SHapley Additive exPlanations).

    Parameters:
    - model: Trained machine learning model.
    - X (array-like): Input features for SHAP values calculation.
    - instance_index (int): Index of the instance to explain.
    - feature_index: Feature index to focus on in force plot ('auto' selects the feature with the highest magnitude SHAP value).

    Returns:
    None

    """
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # Summary Plot
    shap.summary_plot(shap_values, X, feature_names=X.columns)

    # Force Plot for a specific instance
    shap.force_plot(explainer.expected_value, shap_values[instance_index, :], X.iloc[instance_index, :])

def run_models(X_scaled, y_resampled):
    """
    Train and evaluate machine learning models using the provided scaled input features and resampled target values.

    Parameters:
    - X_scaled (array-like): Scaled input features for training the models.
    - y_resampled (array-like): Resampled target values for training the models.

    Returns:
    None

    This function performs the following steps:
    1. Splits the data into training and testing sets.
    2. Conducts hyperparameter optimization using Optuna with the `run_optimize` function.
    3. Displays the best hyperparameters in a tkinter window.
    4. Initializes an XGBoost model with the best hyperparameters.
    5. Trains the model on the training set.
    6. Performs cross-validation and evaluates the model on the training and testing sets.
    7. Prints the cross-validation scores, mean cross-validation score, train score, test score, and feature importance.
    8. Displays SHAP (SHapley Additive exPlanations) plots using the `explain_model_with_shap` function.

    Note: This function assumes the existence of several helper functions such as `train_test_split`, `optuna.create_study`,
    `run_optimize`, `initialize_xgb_model`, `train_model`, `perform_cross_validation`, and `calculate_feature_importance`.

    """
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: run_optimize(trial, X_train, y_train), n_trials=10)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    #Visualize bet parameters in tkinter
    count = 10  # Initialize a count variable
    for key, value in best_params.items():
        tk.Label(label_frame_hyperparameters, font="none 7 bold", text=f"{key} = {round(value, 3)}",fg = "green").grid(row=count, column=0, sticky='w')
        count += 1

    best_model = initialize_xgb_model(best_params)
    best_model = train_model(best_model, X_train, y_train)

    validation_scores = perform_cross_validation(best_model, X_train, y_train)
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    percentage_importance = calculate_feature_importance(best_model)

    print("Cross-Validation Scores:", validation_scores)
    print("Mean Cross-Validation Score:", np.mean(validation_scores))
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    print(f"Feature importance: {percentage_importance}")

    tk.Label(label_frame_results, font="none 7 bold", text=f"Cross-Validation Score = {round(np.mean(validation_scores), 3)}", fg="blue").grid(row=1, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text=f"Train Score = {round(train_score,3)}", fg="blue").grid(row=2, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text=f"Test Score = {round(test_score, 3)}", fg="blue").grid(row=3, column=0, sticky='w')

    # Display SHAP plots
    explain_model_with_shap(best_model, X_test, instance_index=0, feature_index='auto')

create_label_frame_input()
root.mainloop()