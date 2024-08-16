# from dotenv import load_dotenv
# import subprocess
# import pandas as pd
# from heart_disease_prediction.components import estimator
# from heart_disease_prediction.entity.config_entity import DataIngestionConfig
# from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
# from heart_disease_prediction.components.data_ingestion import DataIngestion
# from heart_disease_prediction.components.data_transformation import DataTransformation
# from heart_disease_prediction.components.model_trainer import (
#     read_transformed_data,
#     model_evaluation,
# )

# # Load environment variables from the .env file
# load_dotenv()

# # Initialize Data Ingestion
# data_ingestion_config = DataIngestionConfig()  # Use default configuration
# data_ingestion = DataIngestion(data_ingestion_config)
# diArtifacts = data_ingestion.initiate_data_ingestion()

# # Extract the paths
# test_csv_path = diArtifacts.test_file_path  # Use correct attribute
# schema_path = "./config/schema.yaml"

# # Validate the test data
# subprocess.run(
#     [
#         "python",
#         "heart_disease_prediction/components/data_validation.py",
#         test_csv_path,
#         schema_path,
#     ]
# )

# # Data Transformation
# data_transformation = DataTransformation(diArtifacts, schema_path)
# transformation_artifacts = data_transformation.initiate_data_transformation()

# print("Data transformation completed. Artifacts:", transformation_artifacts)

# # Read transformed data and evaluate the model
# read_transformed_data()
# expected_score = 0.85  # Define the expected accuracy score
# model_evaluation(expected_score)

# print("Model evaluation completed.")

# # Paths to your files
# model_path = "./artifact/best_model.pkl"
# preprocessing_path = "./artifact/preprocessing.pkl"
# test_data_path = diArtifacts.test_file_path

# # Function to get user input and predict
# def get_user_input_and_predict():
#     print("Please enter the following details:")
#     user_data = {
#         "age": input("Enter age : "),
#         "hypertension": input("Enter hypertension (0 for No, 1 for Yes): "),
#         "heart_disease": input("Enter heart disease (0 for No, 1 for Yes): "),
#         "avg_glucose_level": input("Enter average glucose level : "),
#         "bmi": input("Enter BMI : "),
#         "gender": input("Enter gender (Male/Female): "),
#         "ever_married": input("Ever married (Yes/No): "),
#         "work_type": input(
#             "Work type (Private/Self-employed/Govt_job/Children/Never_worked): "
#         ),
#         "Residence_type": input("Residence type (Urban/Rural): "),
#         "smoking_status": input(
#             "Smoking status (formerly smoked/never smoked/smokes/Unknown): "
#         ),
#     }

#     # Convert to DataFrame
#     user_df = pd.DataFrame([user_data])

#     # Convert appropriate columns to numeric
#     numeric_columns = [
#         "age",
#         "hypertension",
#         "heart_disease",
#         "avg_glucose_level",
#         "bmi",
#     ]
#     for column in numeric_columns:
#         user_df[column] = pd.to_numeric(user_df[column])

#     # Predict using the estimator
#     prediction = estimator.predict(model_path, preprocessing_path, user_df)
#     print(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")

# # Execute prediction
# get_user_input_and_predict()





from fastapi import FastAPI, HTTPException
from heart_disease_prediction.components.model_trainer import (
    load_data,
    create_preprocessor,
    save_preprocessing_pipeline,
    train_and_evaluate_model,
    save_model,
    save_evaluation_report_as_yaml,
    hyperparameter_tuning,
    read_yaml,
)
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction App!"}

@app.post("/train")
def train_model(expected_score: float):
    data_file_path = "./heart_disease_prediction/healthcare-dataset-stroke-data.csv"
    data = load_data(data_file_path)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Data file not found")

    X_train, X_test, y_train, y_test = split_data(data)

    preprocessor = create_preprocessor()

    preprocessing_pipeline_path = "./artifact/preprocessing_pipeline.pkl"
    save_preprocessing_pipeline(preprocessor, preprocessing_pipeline_path)

    model_config_path = "./config/model.yaml"
    model_config = read_yaml(model_config_path)

    if model_config is None:
        raise HTTPException(status_code=404, detail="Model configuration file not found")

    best_model = None
    best_model_name = ""
    best_model_report = None

    for module_key, module_value in model_config["model_selection"].items():
        module_name = module_value["module"]
        class_name = module_value["class"]
        params = module_value["params"]

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

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
        model_path = "./artifact/best_model.pkl"
        save_model(best_model, model_path)

        final_report = {
            "best_model": best_model_name,
            "accuracy": best_model_report["accuracy"],
            "classification_report": best_model_report["classification_report"],
            "confusion_matrix": best_model_report["confusion_matrix"],
        }
        yaml_report_path = "./artifact/model_evaluation_report.yaml"
        save_evaluation_report_as_yaml(final_report, yaml_report_path)
        
        return {"message": "Training completed successfully", "report": final_report}
    else:
        return {"message": f"Best model ({best_model_name}) does not meet the expected score. Model discarded."}

@app.get("/predict")
def make_prediction():
    # Add your prediction logic here
    return {"message": "Prediction endpoint"}
