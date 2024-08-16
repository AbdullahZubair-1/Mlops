import os
import pandas as pd
import yaml
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import importlib

# Function to read a YAML file
def read_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None

# Function to load the data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Data file not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path} is empty")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None

# Function to split the data into training and test sets
def split_data(data, test_size=0.2, random_state=42):
    X = data.drop("stroke", axis=1)
    y = data["stroke"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to create the preprocessing pipeline
def create_preprocessor():
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Function to save the preprocessing pipeline separately
def save_preprocessing_pipeline(preprocessor, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(preprocessor, file)
    print(f"Preprocessing pipeline saved at {file_path}")

# Function to train and evaluate the model
def train_and_evaluate_model(model, preprocessor, X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions, output_dict=True)
    confusion_mat = confusion_matrix(y_test, predictions)

    evaluation_report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": confusion_mat.tolist(),  # Convert to list for YAML serialization
    }
    return evaluation_report, pipeline

# Function to save the model
def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

# Function to save the evaluation report as YAML
def save_evaluation_report_as_yaml(report, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        yaml.dump(report, file, default_flow_style=False)
    print(f"Evaluation report saved at {file_path}")

# Function for hyperparameter tuning
def hyperparameter_tuning(model_class, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model_class, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Main function to evaluate the model
def model_evaluation(expected_score):
    # Load the dataset
    data_file_path = "./heart_disease_prediction/healthcare-dataset-stroke-data.csv"
    data = load_data(data_file_path)
    if data is None:
        return

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Create the preprocessor
    preprocessor = create_preprocessor()

    # Save the preprocessing pipeline separately
    preprocessing_pipeline_path = "./artifact/preprocessing_pipeline.pkl"
    save_preprocessing_pipeline(preprocessor, preprocessing_pipeline_path)

    # Load model configurations
    model_config_path = "./config/model.yaml"
    model_config = read_yaml(model_config_path)

    if model_config is None:
        return

    best_model = None
    best_model_name = ""
    best_model_report = None

    # Train and evaluate models
    for module_key, module_value in model_config["model_selection"].items():
        module_name = module_value["module"]
        class_name = module_value["class"]
        params = module_value["params"]

        # Dynamically import the module and class
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Hyperparameter tuning
        param_grid = params.get("param_grid", {})
        if param_grid:
            model = hyperparameter_tuning(model_class, param_grid, X_train, y_train)
        else:
            model = model_class()

        report, pipeline = train_and_evaluate_model(model, preprocessor, X_train, y_train, X_test, y_test)

        if best_model is None or report["accuracy"] > best_model_report["accuracy"]:
            best_model = pipeline
            best_model_name = class_name
            best_model_report = report

    if best_model_report and best_model_report["accuracy"] >= expected_score:
        print(f"Best model ({best_model_name}) meets the expected score. Saving model...")

        # Save the best model
        model_path = "./artifact/best_model.pkl"
        save_model(best_model, model_path)

        # Save the evaluation report
        final_report = {
            "best_model": best_model_name,
            "accuracy": best_model_report["accuracy"],
            "classification_report": best_model_report["classification_report"],
            "confusion_matrix": best_model_report["confusion_matrix"],
        }
        yaml_report_path = "./artifact/model_evaluation_report.yaml"
        save_evaluation_report_as_yaml(final_report, yaml_report_path)
    else:
        print(f"Best model ({best_model_name}) does not meet the expected score. Model discarded.")

if __name__ == "__main__":
    expected_score = 0.5  # Set your expected score
    model_evaluation(expected_score)
