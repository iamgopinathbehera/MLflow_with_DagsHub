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
import dagshub

# Initialize logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Set up local MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")

# Initialize DagsHub for tracking
dagshub.init(repo_owner='iamgopinathbehera', repo_name='MLflow_with_DagsHub', mlflow=True)

# Define evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Main training logic
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV. Error: %s", e)
        sys.exit(1)

    # Split the dataset into training and testing sets
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Model hyperparameters (ElasticNet)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start the MLflow run
    with mlflow.start_run() as run:
        # Initialize and fit the ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions
        predicted_qualities = lr.predict(test_x)

        # Evaluate model performance
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Log results and parameters
        print("ElasticNet model (alpha={:.4f}, l1_ratio={:.4f}):".format(alpha, l1_ratio))
        print("  RMSE: {:.4f}".format(rmse))
        print("  MAE: {:.4f}".format(mae))
        print("  R2: {:.4f}".format(r2))

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log additional metadata using tags
        mlflow.set_tag("dataset_name", "Wine Quality")
        mlflow.set_tag("dataset_url", csv_url)
        mlflow.set_tag("dataset_description", "Chemical properties of wines and their quality ratings.")
        mlflow.set_tag("input_columns", ", ".join(train_x.columns))
        mlflow.set_tag("target_column", "quality")
        mlflow.set_tag("dataset_shape", f"{data.shape[0]} rows, {data.shape[1]} columns")
        mlflow.set_tag("project_description", "Predict wine quality using ElasticNet model.")

        # Log the model
        mlflow.sklearn.log_model(lr, "model")

        # Store and print the run ID
        run_id = run.info.run_id
        print("Model saved in run %s" % run_id)
