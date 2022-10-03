import warnings
import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
# from sklearn.dummy import DummyRegressor

# from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from typing import Tuple

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
# mlflow.set_tracking_uri(
#     'http://127.0.0.1:5000')
# mlflow.set_experiment("tutorial")

# print(mlflow.get_registry_uri())


def eval_metrics(
    actual: pd.DataFrame, pred: pd.DataFrame
) -> Tuple[float, ...]:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def example_custom_metric_fn(eval_df, builtin_metrics, artifacts_dir):
    """
    This example custom metric function creates a metric based on the
    ``prediction`` and ``target`` columns in ``eval_df`` and a metric derived
    from existing metrics in ``builtin_metrics``. It also generates and saves
    a scatter plot to ``artifacts_dir`` that visualizes the relationship
    between the predictions and targets for the given model to a file
    as an image artifact.
    """
    metrics = {
        key: val
        for key, val in zip(
            ["rmse", "mae", "r2"],
            eval_metrics(eval_df["prediction"], eval_df["target"]),
        )
    }
    plt.scatter(eval_df["prediction"], eval_df["target"])
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    artifacts = {"example_scatter_plot_artifact": plot_path}
    return metrics, artifacts


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-"
        "learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, "
            "check your internet connection. Error: %s",
            e,
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    # test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    # test_y = test[["quality"]]

    alpha = float(sys.argv[1])
    l1_ratio = float(sys.argv[2])

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "run_name")

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # baseline = DummyRegressor(strategy='median').fit(train_x, train_y)

        # predicted_qualities = lr.predict(test_x)

        # (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        # print(
        #     "Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        # print("  RMSE: %s" % rmse)
        # print("  MAE: %s" % mae)
        # print("  R2: %s" % r2)

        model_uri = mlflow.sklearn.log_model(lr, "model").model_uri
        # baseline_uri = mlflow.sklearn.log_model(
        #     baseline, 'baseline').model_uri
        _ = mlflow.evaluate(
            model_uri,
            test,
            targets="quality",
            model_type="regressor",
            dataset_name="winequality-red",
            custom_metrics=[example_custom_metric_fn],
            # baseline_model=baseline_uri,
            evaluator_config={"default": {"log_model_explainability": False}},
        )
        # mlflow.log_param("alpha", alpha)
        # mlflow.log_param("l1_ratio", l1_ratio)
        # mlflow.log_metric("rmse", rmse)
        # mlflow.log_metric("r2", r2)
        # mlflow.log_metric("mae", mae)

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     # Register the model
        #     # There are other ways to use the Model Registry,
        #     # which depends on the use case,
        #     # please refer to the doc for more information:
        #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #     mlflow.sklearn.log_model(
        #         lr, "model", registered_model_name="ElasticnetWineModel"
        #     )
        # else:
        #     mlflow.sklearn.log_model(lr, "model")

    return "done"


if __name__ == "__main__":
    main()
