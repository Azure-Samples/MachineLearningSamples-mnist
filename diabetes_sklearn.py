from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Import Run from azureml.core, 
# and get handle of current run for logging and history purposes
from azureml.core.run import Run
run = Run.get_submitted_run()

X, y = load_diabetes(return_X_y=True)

columns = ['age', 'gender', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = {"train":{"x":x_train, "y":y_train},
        "test":{"x":x_test, "y":y_test}}

alpha = 0.55

reg = Ridge(alpha=alpha)
reg.fit(data["train"]["x"], data["train"]["y"])
print('Fitted a model')

preds = reg.predict(data["test"]["x"])
mse = mean_squared_error(preds, data["test"]["y"])

# Log metrics
run.log("alpha", alpha)
run.log("mse", mse)
run.log("columns", columns)

os.makedirs('./outputs', exist_ok=True)
model_path = "outputs/model.pkl"

# Save model as part of the run history
with open(model_path, "wb") as file:
    joblib.dump(reg, file)

print('Validated a model, mean squared error: ',mse)
