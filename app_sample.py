# from fastapi import FastAPI
# from heart_disease_prediction.pipline.training_pipeline import ModelTrainingPipeline

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Heart Disease Prediction App!"}

# @app.get("/train")
# def train_pipeline():
#     # Initialize the training pipeline with the schema file path
#     pipeline = ModelTrainingPipeline(schema_filepath="config/schema.yaml")
    
#     # Start the training process
#     training_artifacts = pipeline.start_training()
    
#     return {
#         "message": "Training pipeline completed successfully!",
#         "artifacts": training_artifacts
#     }

# @app.post("/predict")
# def predict(data: dict):
#     """
#     Make a prediction based on the input data.
#     """
#     # Prepare the input data for prediction
#     # Convert the incoming JSON data to a DataFrame or the format required by your prediction pipeline
#     # Example:
#     # df = pd.DataFrame([data])
#     # prediction = PredictionPipeline(df).predict()
    
#     # For demonstration, we'll return a placeholder message
#     # Replace the following line with your actual prediction logic
#     prediction = "Prediction logic needs to be implemented"

#     return {
#         "message": "Prediction completed!",
#         "prediction": prediction
#     }


from fastapi import FastAPI, HTTPException
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
from heart_disease_prediction.components.model_trainer import ModelTraining
from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
import os

app = FastAPI()

@app.post("/train")
def train_model(expected_score: float):
    try:
        # Define configuration paths
        ingestion_config = DataIngestionConfig(
            raw_data_path="./heart_disease_prediction/healthcare-dataset-stroke-data.csv",
            train_data_path="./artifact/train_data.csv",
            test_data_path="./artifact/test_data.csv",
            feature_store_file_path="./artifact/feature_store.csv",
            train_test_split_ratio=0.2
        )
        
        transformation_config = DataTransformationConfig(
            transformed_train_dir="./artifact/transformed_train",
            transformed_test_dir="./artifact/transformed_test",
            preprocessing_obj_file_path="./artifact/preprocessing.pkl"
        )
        
        # Ensure necessary directories exist
        os.makedirs('./artifact', exist_ok=True)
        os.makedirs('./artifact/transformed_train', exist_ok=True)
        os.makedirs('./artifact/transformed_test', exist_ok=True)

        # Execute Data Ingestion
        ingestion_process = DataIngestion(ingestion_config)
        ingestion_output = ingestion_process.initiate_data_ingestion()

        # Execute Data Transformation
        data_transformation = DataTransformation(
            data_ingestion_artifact=ingestion_output,
            schema_path="./config/schema.yaml"
        )
        transformation_output = data_transformation.initiate_data_transformation()

        # Execute Model Training
        model_training = ModelTraining(
            data_transformation_artifact=transformation_output,
            config_path="./config/model.yaml",
            expected_score=expected_score
        )
        training_output = model_training.initiate_model_training()

        return {
            "message": "Training pipeline completed!",
            "ingestion_output": ingestion_output,
            "transformation_output": transformation_output,
            "training_output": training_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


