## ML FLow experiements

import dagshub
dagshub.init(repo_owner='iamgopinathbehera', repo_name='MLflow_with_DagsHub', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



MLFLOW_TRACKING_URI=https://dagshub.com/iamgopinathbehera/MLflow_with_DagsHub.mlflow

MLFLOW_TRACKING_USERNAME= iamgopinathbehera

MLFLOW_TRACKING_PASSWORD=494b3fc0abe6850bb18527e5ce78e2b9ce45e8f9

python script.py
