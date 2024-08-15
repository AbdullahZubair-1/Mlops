from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from heart_disease_prediction.pipline.training_pipeline import TrainingPipeline
from heart_disease_prediction.pipline.prediction_pipeline import PredictionPipeline
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Define paths
schema_path = "./config/schema.yaml"
model_path = "./artifact/best_model.pkl"
preprocessing_path = "./artifact/preprocessing.pkl"

# Initialize the TrainingPipeline and PredictionPipeline
training_pipeline = TrainingPipeline(schema_path)
prediction_pipeline = PredictionPipeline(model_path, preprocessing_path)

# Create FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def welcome_page(request: Request):
    return templates.TemplateResponse(
        "welcome.html", {"request": request, "title": "Welcome"}
    )


# @app.get("/train", response_class=HTMLResponse)
# async def train_pipeline(request: Request):
#     # Run the training pipeline
#     di_artifacts = training_pipeline.run_data_ingestion()
#     training_pipeline.run_data_validation(di_artifacts.test_file_path)
#     transformation_artifacts = training_pipeline.run_data_transformation(di_artifacts)

#     # Access file paths from the transformation_artifacts dictionary
#     # transformed_train_file_path = transformation_artifacts.get(
#     #     "transformed_train_file_path"
#     # )

#     transformed_train_file_path = "./artifact/transformed_train/train_transformed.csv"

#     # Check if the path is valid
#     if not transformed_train_file_path:
#         return templates.TemplateResponse(
#             "train.html",
#             {
#                 "request": request,
#                 "title": "Training",
#                 "training_result": "Training completed, but no transformed data file found.",
#                 "sample_data": "",
#             },
#         )


@app.get("/train", response_class=HTMLResponse)
async def train_pipeline(request: Request):
    # Run the training pipeline
    di_artifacts = training_pipeline.run_data_ingestion()
    training_pipeline.run_data_validation(di_artifacts.test_file_path)
    transformation_artifacts = training_pipeline.run_data_transformation(di_artifacts)

    # Manually specify the path to the transformed data file
    transformed_train_file_path = "./artifact/transformed_train/train_transformed.csv"

    # Check if the path is valid and the file exists
    try:
        transformed_data = pd.read_csv(transformed_train_file_path)
        # Convert the entire DataFrame to HTML
        sample_data = transformed_data.to_html(
            classes="table table-striped", index=False
        )
        training_result = (
            "Training completed successfully! Here is all of the transformed data:"
        )
    except FileNotFoundError:
        sample_data = ""
        training_result = "Training completed, but no transformed data file found."

    # Return training completion message and sample data
    return templates.TemplateResponse(
        "train.html",
        {
            "request": request,
            "title": "Training",
            "training_result": training_result,
            "sample_data": sample_data,
        },
    )

    # Load a sample of the transformed data to display
    transformed_data = pd.read_csv(transformed_train_file_path)
    sample_data = transformed_data.head(10).to_html(
        classes="table table-striped", index=False
    )

    # Return training completion message and sample data
    training_result = (
        "Training completed successfully! Here is a sample of the transformed data:"
    )
    return templates.TemplateResponse(
        "train.html",
        {
            "request": request,
            "title": "Training",
            "training_result": training_result,
            "sample_data": sample_data,
        },
    )


@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse(
        "test.html",
        {
            "request": request,
            "title": "Testing",
            "prediction_result": "Please submit the form to get a prediction.",
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_pipeline(
    request: Request,
    age: int = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    gender: str = Form(...),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    residence_type: str = Form(...),
    smoking_status: str = Form(...),
):
    # Prepare user input as a dictionary
    user_data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status,
    }

    # Predict using the PredictionPipeline
    prediction = prediction_pipeline.predict(user_data)
    prediction_result = f"Prediction: {prediction}"
    return templates.TemplateResponse(
        "test.html",
        {
            "request": request,
            "title": "Prediction",
            "prediction_result": prediction_result,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
