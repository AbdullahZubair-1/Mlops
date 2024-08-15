import pandas as pd
import yaml
import os
import importlib
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pickle


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


def load_transformed_data(file_path):
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


def process_data():
    train_file_path = "./artifact/transformed_train/train_transformed.csv"
    test_file_path = "./artifact/transformed_test/test_transformed.csv"

    transformed_train = load_transformed_data(train_file_path)
    transformed_test = load_transformed_data(test_file_path)

    return transformed_train, transformed_test


def read_transformed_data():
    transformed_train, transformed_test = process_data()
    if transformed_train is not None:
        print("Training data:")
        print(transformed_train.shape)
    if transformed_test is not None:
        print("Test data:")
        print(transformed_test.shape)


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
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
    return evaluation_report, model


def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def save_evaluation_report_as_yaml(report, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Reformat the report to match the required structure
    formatted_report = {
        "accuracy": report["accuracy"],
        "best_model": report["best_model"],
        "classification_report": {
            "f1-score": report["classification_report"]["macro avg"]["f1-score"],
            "precision": report["classification_report"]["macro avg"]["precision"],
            "recall": report["classification_report"]["macro avg"]["recall"],
            "macro avg": {
                "f1-score": report["classification_report"]["macro avg"]["f1-score"],
                "precision": report["classification_report"]["macro avg"]["precision"],
                "recall": report["classification_report"]["macro avg"]["recall"],
                "support": report["classification_report"]["macro avg"]["support"],
            },
            "weighted avg": {
                "f1-score": report["classification_report"]["weighted avg"]["f1-score"],
                "precision": report["classification_report"]["weighted avg"][
                    "precision"
                ],
                "recall": report["classification_report"]["weighted avg"]["recall"],
                "support": report["classification_report"]["weighted avg"]["support"],
            },
        },
        "confusion_matrix": report["confusion_matrix"],
    }

    with open(file_path, "w") as file:
        yaml.dump(formatted_report, file, default_flow_style=False)
    print(f"Evaluation report saved at {file_path}")


def model_evaluation(expected_score):
    transformed_train, transformed_test = process_data()
    model_config_path = os.path.join(
        os.path.dirname(__file__), "../../config/model.yaml"
    )
    model_config = read_yaml(model_config_path)

    if (
        transformed_train is not None
        and transformed_test is not None
        and model_config is not None
    ):
        X_train = transformed_train.drop("stroke", axis=1)
        y_train = transformed_train["stroke"]
        X_test = transformed_test.drop("stroke", axis=1)
        y_test = transformed_test["stroke"]

        best_model = None
        best_model_name = ""
        best_model_report = None

        for module_key, module_value in model_config["model_selection"].items():
            module_name = module_value["module"]
            class_name = module_value["class"]
            params = module_value["params"]

            # Dynamically import the module and class
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            model = model_class(**params)

            report, trained_model = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test
            )

            if best_model is None or report["accuracy"] > best_model_report["accuracy"]:
                best_model = trained_model
                best_model_name = class_name
                best_model_report = report

        if best_model_report and best_model_report["accuracy"] >= expected_score:
            print(
                f"Best model ({best_model_name}) meets the expected score. Saving model..."
            )
            best_model_report["best_model"] = (
                best_model_name  # Add best model name to the report
            )

            # Create a new report structure that only contains the best model's report
            final_report = {
                "best_model": best_model_name,
                "accuracy": best_model_report["accuracy"],
                "classification_report": best_model_report["classification_report"],
                "confusion_matrix": best_model_report["confusion_matrix"],
            }

            model_path = "./artifact/best_model.pkl"
            save_model(best_model, model_path)

            # Save evaluation report as YAML
            yaml_report_path = "./artifact/model_evaluation_report.yaml"
            save_evaluation_report_as_yaml(final_report, yaml_report_path)
            print(f"Evaluation report saved at {yaml_report_path}")
        else:
            print(
                f"Best model ({best_model_name}) does not meet the expected score. Model discarded."
            )


if __name__ == "__main__":
    expected_score = 0.5  # Set your expected score
    model_evaluation(expected_score)
