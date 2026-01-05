"""
Simple training script:
- loads iris dataset from sklearn
- trains a LogisticRegression
- saves model to model.pkl
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import json

import mlflow
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a logistic regression model on the iris dataset with MLflow tracking.")
    parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000', help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='flower_Set_experiment_2', help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default='sample_run_1', help='MLflow run name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size fraction')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train/test split')
    parser.add_argument('--max_iter', type=int, default=200, help='Max iterations for LogisticRegression')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', help='Directory to save artifacts')
    parser.add_argument('--model_filename', type=str, default='model.pkl', help='Model filename')
    parser.add_argument('--metrics_filename', type=str, default='metrics.json', help='Metrics filename')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    # Set the MLflow experiment name (creates if not exists)
    mlflow.set_experiment(args.experiment_name)
    # Start a new MLflow run for tracking this experiment
    with mlflow.start_run(run_name=args.run_name):
        # Load the iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state)

        # Train a logistic regression model
        model = LogisticRegression(max_iter=args.max_iter)
        model.fit(X_train, y_train)

        # Log all relevant parameters to MLflow
        mlflow.log_params({
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "train_size": 1.0 - args.test_size,
            "split_ratio": f"{int((1.0-args.test_size)*100)}:{int(args.test_size*100)}",
            "random_state": args.random_state,
            "artifacts_dir": args.artifacts_dir,
            "model_filename": args.model_filename,
            "metrics_filename": args.metrics_filename,
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
            "experiment_name": args.experiment_name,
            "run_name": args.run_name
        })

        # Save the trained model to disk
        os.makedirs(args.artifacts_dir, exist_ok=True)
        model_path = os.path.join(args.artifacts_dir, args.model_filename)
        joblib.dump(model, model_path)
        # Log the model file as an artifact in MLflow
        mlflow.log_artifact(model_path)

        # Evaluate the model and save metrics
        acc = model.score(X_test, y_test)
        metrics = {"accuracy": float(acc)}
        metrics_path = os.path.join(args.artifacts_dir, args.metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        # Log accuracy metric and metrics file to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(metrics_path)

        # Log prediction parameters (input features and predictions) to MLflow
        predictions = model.predict(X_test)
        prediction_info = {
            "X_test": X_test.tolist(),
            "predictions": predictions.tolist(),
            "y_test": y_test.tolist()
        }
        prediction_path = os.path.join(args.artifacts_dir, "predictions.json")
        with open(prediction_path, "w") as f:
            json.dump(prediction_info, f)
        mlflow.log_artifact(prediction_path)

        # Print results
        print(f"Saved model to {model_path}")
        print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
