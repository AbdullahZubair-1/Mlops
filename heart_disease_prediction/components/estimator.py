# # estimator.py

# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix, accuracy_score


# def load_and_evaluate(model_path, preprocessing_path, test_data_path):
#     # Load the model and preprocessing pipeline
#     model = joblib.load(model_path)
#     preprocessing_pipeline = joblib.load(preprocessing_path)

#     # Load the test data
#     test_data = pd.read_csv(test_data_path)

#     # Extract true labels before transformation
#     y_test = test_data["stroke"].values  # Adjust based on your dataset

#     # Apply preprocessing to the test data
#     transformed_data = preprocessing_pipeline.transform(test_data)

#     # Make predictions
#     predictions = model.predict(transformed_data)

#     # Calculate and print the accuracy
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Accuracy: {accuracy:.4f}")


# def predict(model_path, preprocessing_path, user_data):
#     # Load the model and preprocessing pipeline
#     model = joblib.load(model_path)
#     preprocessing_pipeline = joblib.load(preprocessing_path)

#     # Ensure the user data has all required columns with correct data types
#     required_columns = [
#         "age",
#         "hypertension",
#         "heart_disease",
#         "avg_glucose_level",
#         "bmi",
#         "gender",
#         "ever_married",
#         "work_type",
#         "Residence_type",
#         "smoking_status",
#     ]

#     # Reorder and fill missing columns if necessary
#     for col in required_columns:
#         if col not in user_data.columns:
#             user_data[col] = 0

#     # Apply preprocessing to the user data
#     transformed_data = preprocessing_pipeline.transform(user_data)

#     # Make predictions
#     predictions = model.predict(transformed_data)
#     return predictions


# estimator.py

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def load_and_evaluate(model_path, preprocessing_path, test_data_path):
    """
    Load the trained model and preprocessing pipeline, apply preprocessing to the test data,
    make predictions, and print the accuracy.
    
    Parameters:
    - model_path: Path to the trained model file.
    - preprocessing_path: Path to the preprocessing pipeline file.
    - test_data_path: Path to the test data CSV file.
    """
    # Load the model and preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(preprocessing_path)

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Extract true labels before transformation
    y_test = test_data["stroke"].values  # Adjust based on your dataset

    # Apply preprocessing to the test data
    transformed_data = preprocessing_pipeline.transform(test_data)

    # Make predictions
    predictions = model.predict(transformed_data)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

def predict(model_path, preprocessing_path, user_data):
    """
    Load the trained model and preprocessing pipeline, apply preprocessing to the user data,
    make predictions, and return the predictions.
    
    Parameters:
    - model_path: Path to the trained model file.
    - preprocessing_path: Path to the preprocessing pipeline file.
    - user_data: DataFrame containing user data to be predicted.
    
    Returns:
    - predictions: Model predictions for the provided user data.
    """
    # Load the model and preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = joblib.load(preprocessing_path)

    # Ensure the user data has all required columns with correct data types
    required_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    # Reorder and fill missing columns if necessary
    for col in required_columns:
        if col not in user_data.columns:
            user_data[col] = 0

    # Apply preprocessing to the user data
    transformed_data = preprocessing_pipeline.transform(user_data)

    # Make predictions
    predictions = model.predict(transformed_data)
    return predictions

if __name__ == "__main__":
    # Example usage for load_and_evaluate
    model_path = "./artifact/best_model.pkl"
    preprocessing_path = "./artifact/preprocessing_pipeline.pkl"
    test_data_path = "./path/to/your/test_data.csv"

    # Uncomment to evaluate the model
    # load_and_evaluate(model_path, preprocessing_path, test_data_path)

    # Example usage for predict
    user_data = pd.DataFrame({
        "age": [45],
        "hypertension": [0],
        "heart_disease": [0],
        "avg_glucose_level": [75],
        "bmi": [25],
        "gender": ["Male"],
        "ever_married": ["Yes"],
        "work_type": ["Private"],
        "Residence_type": ["Urban"],
        "smoking_status": ["Never smoked"]
    })

    # Uncomment to make predictions
    # predictions = predict(model_path, preprocessing_path, user_data)
    # print(f"Predictions: {predictions}")
