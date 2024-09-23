import os
import warnings
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Set up local MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        sys.exit(1)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:.4f}, l1_ratio={:.4f}):".format(alpha, l1_ratio))
        print("  RMSE: {:.4f}".format(rmse))
        print("  MAE: {:.4f}".format(mae))
        print("  R2: {:.4f}".format(r2))

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log dataset info
        mlflow.set_tag("dataset_name", "Wine Quality")
        mlflow.set_tag("dataset_url", csv_url)
        mlflow.set_tag("dataset_description", "The dataset contains various chemical properties of wines, such as acidity, pH, alcohol, etc., along with their quality ratings.")
        mlflow.set_tag("input_columns", ", ".join(train_x.columns))
        mlflow.set_tag("target_column", "quality")
        mlflow.set_tag("dataset_shape", f"{data.shape[0]} rows, {data.shape[1]} columns")

        # Log a description of the project
        mlflow.set_tag("project_description", "This project aims to predict the quality of wine based on its chemical properties using an ElasticNet model.")

        mlflow.sklearn.log_model(lr, "model")
        
        # Store the run ID
        run_id = run.info.run_id

    # Print the run ID after the MLflow run has completed
    print("Model saved in run %s" % run_id)