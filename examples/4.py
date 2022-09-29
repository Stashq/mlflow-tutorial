import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.client import MlflowClient

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


# mlflow setup
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("tutorial")

# training
iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

# Signature creating vol 1
input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Signature creating vol 2
signature = infer_signature(iris_train, clf.predict(iris_train))

# Model logging
model_info = mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)

# Model registration
registration_info = mlflow.register_model(
    model_info.model_uri, 'rfc', tags={'dataset': 'iris'})
print(model_info.model_uri)

# changing
client = MlflowClient()
client.transition_model_version_stage(
    name="rfc",
    version=registration_info.version,
    stage="Production"
)


client.update_model_version(
    name="rfc",
    version=registration_info.version,
    description="This model version is a scikit-learn random forest."
)
