from dotenv import load_dotenv
from heart_disease_prediction.pipline.training_pipeline import TrainingPipeline
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline

# Load environment variables from the .env file
load_dotenv()

# Define paths
schema_path = "./config/schema.yaml"

# Initialize the TrainingPipeline with the schema path
training_pipeline = TrainingPipeline(schema_path)

# Execute the training steps
diArtifacts = training_pipeline.run_data_ingestion()
training_pipeline.run_data_validation(diArtifacts.test_file_path)
transformation_artifacts = training_pipeline.run_data_transformation(diArtifacts)

# Extract paths for model and preprocessing
model_path = "./artifact/best_model.pkl"
preprocessing_path = "./artifact/preprocessing.pkl"

# Prediction Pipeline
prediction_pipeline = PredictionPipeline(model_path, preprocessing_path)


# Function to get user input and predict
def get_user_input_and_predict():
    print("Please enter the following details:")
    user_data = {
        "age": input("Enter age (e.g., 45): "),
        "hypertension": input("Enter hypertension (0 for No, 1 for Yes): "),
        "heart_disease": input("Enter heart disease (0 for No, 1 for Yes): "),
        "avg_glucose_level": input("Enter average glucose level (e.g., 85.96): "),
        "bmi": input("Enter BMI (e.g., 22.3): "),
        "gender": input("Enter gender (Male/Female): "),
        "ever_married": input("Ever married (Yes/No): "),
        "work_type": input(
            "Work type (Private/Self-employed/Govt_job/Children/Never_worked): "
        ),
        "Residence_type": input("Residence type (Urban/Rural): "),
        "smoking_status": input(
            "Smoking status (formerly smoked/never smoked/smokes/Unknown): "
        ),
    }

    prediction = prediction_pipeline.predict(user_data)
    print(f"Prediction: {prediction}")


# Execute prediction
get_user_input_and_predict()
