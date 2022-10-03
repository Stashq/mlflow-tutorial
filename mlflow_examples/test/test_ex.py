from mlflow_examples._2 import eval_metrics, main
import numpy as np


def test_eval_metrics():
    y_true = np.random.randn(20)
    y_pred = np.random.randn(20)
    rmse1, mae1, _ = eval_metrics(y_true, y_pred)
    rmse2, mae2, _ = eval_metrics(y_pred, y_true)
    assert rmse1 == rmse2
    assert mae1 == mae2


def test_main():
    assert main() == "done"
