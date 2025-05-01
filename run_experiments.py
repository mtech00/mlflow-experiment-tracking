
import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mlflow.tracking import MlflowClient

# Data loader function. Load the Iris dataset for demonstration.
def load_data():
    print("Loading data...")
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names

#   Feature engineering function . Create different feature sets 
def create_feature_sets(X):
    """Create different feature sets to experiment with."""
    print("Creating feature sets...")
    
    # Basic features (use original features)
    basic_features = X
    
    # Subset of features (first two columns)
    subset_features = X[:, :2]
    
    # Squared features (add x²)
    squared_features = np.hstack([X, X**2])
    
    return {
        "basic": basic_features,
        "subset": subset_features, 
        "squared": squared_features
    }

# Train function with logging to MLFlow
def train_and_log_model(model, model_name, X_train, X_val, y_train, y_val, 
                        feature_set_name, feature_names, params=None):
    
    # new MLFlow run
    with mlflow.start_run(run_name=f"{model_name}_{feature_set_name}"):
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log which feature set we're using
        mlflow.log_param("feature_set", feature_set_name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("val_f1", val_f1)
        
        
        # Create confusion matrix visualization
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save plot to file
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        print(f"Model: {model_name}, Feature Set: {feature_set_name}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        
        return val_f1  # Return validation F1 score for comparison

# Function for hyperparameter tuning for best combination via GridSearch
def optimize_hyperparameters(model_type, X_train, X_val, y_train, y_val, feature_set_name, 
                            feature_names):

    # Define parameter grids based on model type
    if model_type == "logistic_regression":
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    elif model_type == "random_forest":
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 15]
        }
        model = RandomForestClassifier(random_state=42)
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='f1_weighted', 
        n_jobs=-1
    )
    
    # Train model with grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Log the best model
    with mlflow.start_run(run_name=f"{model_type}_{feature_set_name}_best"):
        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param("feature_set", feature_set_name)
        mlflow.log_param("optimized", True)
        
        # Evaluate model
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("val_f1", val_f1)
        
        
        print(f"Best {model_type} model with {feature_set_name} features:")
        print(f"Best parameters: {best_params}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        
        return best_model, best_params, val_f1


def main():
    # Set MLFlow tracking URI server from docker
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    # Silence Git warnings from MLFlow
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    
    # Create an experiment
    experiment_name = "iris_classification"
    mlflow.set_experiment(experiment_name)
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Create different feature sets
    feature_sets = create_feature_sets(X_train)
    val_feature_sets = create_feature_sets(X_val)
    test_feature_sets = create_feature_sets(X_test)
    
    # Scale features
    for feature_set_name in feature_sets:
        scaler = StandardScaler()
        feature_sets[feature_set_name] = scaler.fit_transform(feature_sets[feature_set_name])
        val_feature_sets[feature_set_name] = scaler.transform(val_feature_sets[feature_set_name])
        test_feature_sets[feature_set_name] = scaler.transform(test_feature_sets[feature_set_name])
    
    # Define models to experiment
    models = {
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42)
    }
    
    # temporary results
    results = {}
    
    # Basic experiments with different models and feature sets
    print("Running basic experiments...")
    for model_name, model in models.items():
        for feature_set_name in feature_sets:
            val_f1 = train_and_log_model(
                model,
                model_name, 
                feature_sets[feature_set_name], 
                val_feature_sets[feature_set_name],
                y_train, 
                y_val,
                feature_set_name,
                feature_names
            )
            
            # Store result based metrics key format
            key = f"{model_name}_{feature_set_name}"
            results[key] = val_f1
    
    # Find the best model/feature set combination
    best_combination = max(results.items(), key=lambda x: x[1])
    
    # The key format should be "{model_name}_{feature_set}"
    # For example: "logistic_regression_basic" -> model="logistic_regression", feature_set="basic"
    best_key = best_combination[0]
    
    # Extract the feature set (the last part after the last underscore)
    best_feature_set = best_key.split("_")[-1]
    
    # Extract the model name (everything before the last underscore)
    best_model_name = "_".join(best_key.split("_")[:-1])
    
    print(f"\nBest model/feature combination: {best_combination[0]} with F1 score: {best_combination[1]:.4f}")
    print(f"Model name: {best_model_name}")
    print(f"Feature set: {best_feature_set}")
    
    # Hyperparameter tuning for the best combination with GridSearchCV
    print(f"\nOptimizing hyperparameters for best model: {best_model_name} with {best_feature_set} features")
    best_model, best_params, best_val_f1 = optimize_hyperparameters(
        best_model_name,
        feature_sets[best_feature_set],
        val_feature_sets[best_feature_set],
        y_train,
        y_val,
        best_feature_set,
        feature_names
    )
    
    # Final evaluation on test set
    print("\nFinal evaluation of best model on test set...")
    with mlflow.start_run(run_name=f"{best_model_name}_{best_feature_set}_final"):

        mlflow.log_params(best_params)
        mlflow.log_param("feature_set", best_feature_set)
        mlflow.log_param("final_evaluation", True)
        

        y_test_pred = best_model.predict(test_feature_sets[best_feature_set])

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)
        
        # Register final model to mlflow model registry
        mlflow.sklearn.log_model(
            best_model, 
            "model",
            registered_model_name="iris_classification_best_model"
        )

        # Update as a latest version
        latest_version = mlflow.MlflowClient().get_latest_versions(
             "iris_classification_best_model"
        )[0].version
        client = MlflowClient()

        # Alias for active tagging rather than hard version
        client.set_registered_model_alias(
            name="iris_classification_best_model",
            alias="production",
            version=latest_version
        )        
        print(f"Final model: {best_model_name} with {best_feature_set} features")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        # Create final confusion matrix on test data
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Final Test Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save plot to file
        plt.savefig("final_confusion_matrix.png")
        mlflow.log_artifact("final_confusion_matrix.png")
        plt.close()


if __name__ == "__main__":
    main()
