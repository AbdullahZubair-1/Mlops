from heart_disease_prediction.entity.config_entity import DataIngestionConfig
from heart_disease_prediction.entity.artifact_entity import DataIngestionArtifact
from heart_disease_prediction.components.data_ingestion import DataIngestion
from heart_disease_prediction.components.data_transformation import DataTransformation
from dotenv import load_dotenv
import os
import subprocess

# Load environment variables from the .env file
load_dotenv(dotenv_path=r'E:\Python\1\ATS-Internship\02. MLOps-Project\.env')

# Verify that the environment variable is set
mongodb_url = os.getenv('MONGODB_URL')
print(f"MONGODB_URL: {mongodb_url}")

if not mongodb_url:
    raise ValueError("MONGODB_URL environment variable not set.")

# Data Ingestion
data_ingestion = DataIngestion(DataIngestionConfig)
diArtifacts = data_ingestion.initiate_data_ingestion()


# Data Validation
# use this in command python "E:/Python/1/ATS-Internship/02. MLOps-Project/heart_disease_prediction/components/data_validation.py" "E:/Python/1/ATS-Internship/02. MLOps-Project/heart_disease_prediction/healthcare-dataset-stroke-data.csv" "E:/Python/1/ATS-Internship/02. MLOps-Project/config/schema.yaml"

# Initialize DataIngestion
data_ingestion_config = DataIngestionConfig()  # Ensure you initialize this correctly
data_ingestion = DataIngestion(data_ingestion_config)
diArtifacts = data_ingestion.initiate_data_ingestion()  # Get artifacts

# Paths for validation
test_csv_path = diArtifacts.test_file_path
schema_path = r"E:\Python\1\ATS-Internship\02. MLOps-Project\config\schema.yaml"

# Validate the data
try:
    validate_data(test_csv_path, schema_path)  # Call the validate_data function
    print("Data validation passed. Proceeding to transformation.")

    # Proceed with data transformation
    data_transformation = DataTransformation()  # Initialize DataTransformation
    ingested_data = pd.read_csv(test_csv_path)  # Load the data for transformation
    transformed_data = data_transformation.transform(ingested_data)
except Exception as e:
    print(f"Data validation failed: {e}")

# Run validation script as a subprocess (optional)
subprocess.run(
    [
        "python",
        "E:/Python/1/ATS-Internship/02. MLOps-Project/heart_disease_prediction/components/data_validation.py",
        test_csv_path,
        schema_path,
    ]
)

# # Data Transformation
# data_transformation = DataTransformation(diArtifacts, schema_path)
# transformation_artifacts = data_transformation.initiate_data_transformation()

# print("Data transformation completed. Artifacts:", transformation_artifacts)










