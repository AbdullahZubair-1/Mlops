from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
from dotenv import load_dotenv
import os
import subprocess

# Load environment variables from the .env file
load_dotenv()

# Data Ingestion
data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()

# Extract the test.csv path and schema.yaml path
test_csv_path = diArtifacts.test_file_path
schema_path = "./config/schema.yaml"

# Validate the test data
subprocess.run(
    [
        "python",
        "heart_disease_prediction/components/data_validation.py",
        test_csv_path,
        schema_path,
    ]
)

# Data Transformation
data_transformation = DataTransformation(diArtifacts, schema_path)
transformation_artifacts = data_transformation.initiate_data_transformation()

print("Data transformation completed. Artifacts:", transformation_artifacts)
