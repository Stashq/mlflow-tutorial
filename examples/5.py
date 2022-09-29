import mlflow.pyfunc
from sklearn import datasets
import pandas as pd

mlflow.set_tracking_uri('http://127.0.0.1:5000')

model_name = "rfc"
model_version = 1

model = mlflow.pyfunc.load_model(
    f'models:/{model_name}/{model_version}'
)

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
y_pred = model.predict(iris_train)
print(y_pred)
