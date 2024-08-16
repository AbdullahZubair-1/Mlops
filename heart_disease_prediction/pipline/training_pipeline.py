# heart_disease_prediction/pipeline/training_pipeline.py

from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
import subprocess

class ModelTrainingPipeline:
    def __init__(self, schema_filepath: str):
        self.schema_filepath = schema_filepath

    def execute_data_ingestion(self):
        # Perform Data Ingestion
        ingestion_process = DataIngestion(DataIngestionConfig())
        ingestion_output = ingestion_process.initiate_data_ingestion()
        print("Data ingestion process completed.")
        return ingestion_output

    def execute_data_validation(self, validation_csv_path: str):
        # Perform Data Validation
        subprocess.run(
            [
                "python",
                "heart_disease_prediction/components/data_validation.py",
                validation_csv_path,
                self.schema_filepath,
            ]
        )
        print("Data validation process completed.")

    def execute_data_transformation(self, ingestion_output):
        # Perform Data Transformation
        transformation_process = DataTransformation(ingestion_output, self.schema_filepath)
        transformation_output = transformation_process.initiate_data_transformation()
        print("Data transformation process completed. Artifacts generated:", transformation_output)
        return transformation_output

    def start_training(self):
        # Execute data ingestion
        ingestion_output = self.execute_data_ingestion()

        # Execute data validation
        self.execute_data_validation(ingestion_output.test_file_path)

        # Execute data transformation
        transformation_output = self.execute_data_transformation(ingestion_output)

        # Return model and preprocessing file paths for future use
        return {
            "model_filepath": "./artifact/best_model.pkl",
            "preprocessing_filepath": "./artifact/preprocessing.pkl",
            "test_data_filepath": ingestion_output.test_file_path,
        }
