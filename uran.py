import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read CSV file
    try:
        data = pd.read_csv("C://Users//rokha//Downloads//urban_//futuristic_city_traffic.csv")
        # Perform dummy encoding
        data = pd.get_dummies(data, columns=["Vehicle Type", "Weather", "Economic Condition"], drop_first=True)
        # Perform one-hot encoding
        data = pd.get_dummies(data, columns=["City", "Day Of Week"], drop_first=False)

        train, test = train_test_split(data)

        train_x = train.drop(["Is Peak Hour"], axis=1)
        test_x = test.drop(['Is Peak Hour'], axis=1)
        train_y = train[["Is Peak Hour"]]
        test_y = test[['Is Peak Hour']]
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)
        sys.exit(1)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    remote_server_uri = 'https://dagshub.com/ummefahad/Urban_data_Analysis.mlflow'
    mlflow.set_tracking_uri(remote_server_uri)

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print(" RMSE: %s" % rmse)
        print(" MAE: %s" % mae)
        print(" R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("rmse", rmse)
        mlflow.log_param("mae", mae)
        mlflow.log_param("r2", r2)

        mlflow.sklearn.log_model(lr, "model")
