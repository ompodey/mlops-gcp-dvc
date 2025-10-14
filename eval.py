# model_check.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/iris.csv")

# Split into train/test for evaluation
train, test = train_test_split(
    data, test_size=0.2, stratify=data['species'], random_state=42
)

X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test['species']

# Load the model
model = joblib.load("model.joblib")

# Predict and evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Sanity check / test set accuracy: {acc:.4f}")

# Optional: check a single sample for sanity
sample = X_test.iloc[0:1]
sample_pred = model.predict(sample)
print(f"Sanity check prediction for first sample: {sample_pred[0]}")
