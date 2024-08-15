import os
import pandas as pd
import sys
import yaml


def load_csv_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")
    data = pd.read_csv(file_path)
    return data


def load_schema(schema_path):
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"The schema file at path {schema_path} does not exist."
        )
    with open(schema_path, "r") as file:
        schema = yaml.safe_load(file)
    return schema


def extract_column_names(schema):
    # Extract column names from the schema
    columns = schema.get("columns", [])
    column_names = [list(column.keys())[0] for column in columns]
    return column_names


class DataValidator:
    def __init__(self, data, schema):
        self.data = data
        self.schema = schema
        self.results = {
            "num_columns": None,
            "column_names": None,
            "numerical_columns": None,
            "categorical_columns": None,
        }

    def validate_num_columns(self):
        expected_columns = extract_column_names(self.schema)
        if len(expected_columns) != len(self.data.columns):
            self.results["num_columns"] = (
                False,
                len(expected_columns),
                len(self.data.columns),
            )
        else:
            self.results["num_columns"] = (
                True,
                len(expected_columns),
                len(self.data.columns),
            )

    def validate_column_names(self):
        expected_columns = extract_column_names(self.schema)
        if set(expected_columns) != set(self.data.columns):
            self.results["column_names"] = (
                False,
                expected_columns,
                self.data.columns.tolist(),
            )
        else:
            self.results["column_names"] = (
                True,
                expected_columns,
                self.data.columns.tolist(),
            )

    def validate_numerical_columns(self):
        expected_numerical_columns = self.schema.get("numerical_columns", [])
        actual_numerical_columns = self.data.select_dtypes(
            include=["number"]
        ).columns.tolist()
        if set(expected_numerical_columns) != set(actual_numerical_columns):
            self.results["numerical_columns"] = (
                False,
                expected_numerical_columns,
                actual_numerical_columns,
            )
        else:
            self.results["numerical_columns"] = (
                True,
                expected_numerical_columns,
                actual_numerical_columns,
            )

    def validate_categorical_columns(self):
        expected_categorical_columns = self.schema.get("categorical_columns", [])
        actual_categorical_columns = self.data.select_dtypes(
            include=["object"]
        ).columns.tolist()
        if set(expected_categorical_columns) != set(actual_categorical_columns):
            self.results["categorical_columns"] = (
                False,
                expected_categorical_columns,
                actual_categorical_columns,
            )
        else:
            self.results["categorical_columns"] = (
                True,
                expected_categorical_columns,
                actual_categorical_columns,
            )

    def validate(self):
        self.validate_num_columns()
        self.validate_column_names()
        self.validate_numerical_columns()
        self.validate_categorical_columns()

    def summary(self):
        for check, result in self.results.items():
            status, expected, found = result
            if status:
                print(f"{check.replace('_', ' ').title()} check passed.")
            else:
                print(
                    f"{check.replace('_', ' ').title()} check failed. Expected: {expected}, Found: {found}"
                )


def validate_data(file_path, schema_path):
    data = load_csv_data(file_path)
    schema = load_schema(schema_path)

    print("Loaded data from:", file_path)
    print(data.head())  # Print the first few rows of the data

    validator = DataValidator(data, schema)
    validator.validate()
    validator.summary()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_validation.py <path_to_csv> <path_to_schema>")
    else:
        validate_data(sys.argv[1], sys.argv[2])
