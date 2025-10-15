import pandas as pd
import pytest
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@pytest.fixture
def data():
    df= pd.read_csv("data/iris.csv")  
    return df

@pytest.fixture
def model():
    mdl= joblib.load("model.joblib") 
    return mdl

# Sanity check
def test_data_sanity(data):
    assert data.isnull().sum().sum()== 0,"Data contains null values!"
    expected_cols =['sepal_length','sepal_width', 'petal_length','petal_width', 'species']
    assert all(col in data.columns for col in expected_cols), "Missing expected columns!"
    print(f"Data has successfully passed the sanity check : {len(data)} rows, columns: {list(data.columns)}")


def test_model_evaluation(data, model):
    _, eval_data = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=42)
    X_eval =eval_data[['sepal_length','sepal_width','petal_length','petal_width']]
    y_eval = eval_data['species']
    y_pred= model.predict(X_eval)
    test_acc = accuracy_score(y_eval, y_pred)

    train_metrics = pd.read_csv("metrics.csv") 
    training_acc = train_metrics['accuracy'].values[0]
    
    sample_df = eval_data.copy()
    sample_df['predicted'] = y_pred
    if len(sample_df) > 20:
        sample_df = sample_df.head(20)
    
    sample_df['training_accuracy'] = training_acc
    sample_df['test_accuracy'] = test_acc

    sample_df.to_csv("metrics_test.csv", index=False)  
    
    print("\nEvaluation Metrics Sample:")
    print(sample_df.to_string(index=False))    
    assert test_acc > 0.7, "Test accuracy too low!"
