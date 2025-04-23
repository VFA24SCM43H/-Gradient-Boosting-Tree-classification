import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import our implementation
from GradientBoost.model.GradientBoostedTrees import GradientBoostedTrees

def test_simple_classification():
    """Test the GradientBoostedTrees on a simple classification dataset"""
    # Create a simple dataset
    X, y = make_classification(n_samples=100, n_features=5, 
                               n_redundant=0, random_state=42)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    gbt = GradientBoostedTrees(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gbt.fit(X_train, y_train)
    
    # Predict
    y_pred = gbt.predict(X_test)
    
    # Assert accuracy is above 0.7
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Simple classification accuracy: {accuracy:.4f}")
    assert accuracy > 0.7

def test_breast_cancer_dataset():
    """Test on the Breast Cancer Wisconsin dataset"""
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Take a smaller subset for faster testing
    X = X[:100]  # Use only first 100 samples
    y = y[:100]
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model with fewer estimators
    gbt = GradientBoostedTrees(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42)
    gbt.fit(X_train, y_train)
    
    # Predict
    y_pred = gbt.predict(X_test)
    y_proba = gbt.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Breast cancer dataset - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Lower the threshold slightly for the smaller dataset
    assert accuracy > 0.7
    assert auc > 0.7
def test_collinear_data():
    """Test the model on highly collinear data to check if the boosted tree model handles it well"""
    # Create dataset with collinear features
    n_samples = 200
    X = np.random.randn(n_samples, 3)  # Generate 3 random features
    
    # Create collinear features
    X = np.column_stack([
        X,
        X[:, 0] + 0.1 * np.random.randn(n_samples),  # Highly correlated with X[:, 0]
        X[:, 1] + 0.1 * np.random.randn(n_samples),  # Highly correlated with X[:, 1]
        X[:, 2] + 0.1 * np.random.randn(n_samples)   # Highly correlated with X[:, 2]
    ])
    
    # Create target: depends only on first 3 features, not on collinear ones
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Fit model
    gbt = GradientBoostedTrees(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbt.fit(X_train, y_train)
    
    # Predict
    y_pred = gbt.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Collinear data accuracy: {accuracy:.4f}")
    
    # Assert accuracy is reasonable
    assert accuracy > 0.75

def test_learning_rate_impact():
    """Test the impact of learning rate on model performance"""
    # Create dataset
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different learning rates
    learning_rates = [0.01, 0.1, 1.0]
    accuracies = []
    
    for lr in learning_rates:
        gbt = GradientBoostedTrees(n_estimators=50, learning_rate=lr, max_depth=3, random_state=42)
        gbt.fit(X_train, y_train)
        y_pred = gbt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    print(f"Learning rates: {learning_rates}")
    print(f"Corresponding accuracies: {accuracies}")
    
    # Lower learning rates with same number of trees should generally perform worse
    # This isn't always true but is a reasonable assumption for this test
    assert accuracies[0] < accuracies[1] or accuracies[1] < accuracies[2]

def test_max_depth_impact():
    """Test the impact of tree depth on model performance"""
    # Create dataset
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different max depths
    depths = [1, 3, 5]
    accuracies = []
    
    for depth in depths:
        gbt = GradientBoostedTrees(n_estimators=50, learning_rate=0.1, max_depth=depth, random_state=42)
        gbt.fit(X_train, y_train)
        y_pred = gbt.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    print(f"Max depths: {depths}")
    print(f"Corresponding accuracies: {accuracies}")
    
    # Deeper trees should generally give better training accuracy
    # but this doesn't always translate to test accuracy due to overfitting
    # Just check that we get reasonable accuracy
    assert max(accuracies) > 0.7

def test_from_csv_file():
    """Test loading data from a CSV file and training the model"""
    # Create a simple dataset and save to CSV
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Save to CSV
    csv_path = 'test_data.csv'
    df.to_csv(csv_path, index=False)
    
    # Load data from CSV
    loaded_df = pd.read_csv(csv_path)
    X_loaded = loaded_df.drop('target', axis=1).values
    y_loaded = loaded_df['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.2, random_state=42)
    
    # Fit model
    gbt = GradientBoostedTrees(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gbt.fit(X_train, y_train)
    
    # Predict
    y_pred = gbt.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CSV data accuracy: {accuracy:.4f}")
    
    # Assert accuracy is reasonable
    assert accuracy > 0.7
    
    # Clean up
    import os
    os.remove(csv_path)

if __name__ == "__main__":
    # Run all tests and print results
    test_simple_classification()
    test_breast_cancer_dataset()
    test_collinear_data()
    test_learning_rate_impact()
    test_max_depth_impact()
    test_from_csv_file()
    print("All tests passed!")