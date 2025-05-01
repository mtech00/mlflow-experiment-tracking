
# Report generator for MLFlow runs 


import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
import os

# Connect to MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()

# Create reports directory
if not os.path.exists("reports"):
    os.makedirs("reports")

# Get experiment
experiment = client.get_experiment_by_name("iris_classification")
if not experiment:
    print("Experiment 'iris_classification' not found.")
    exit()

# Get all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=""
)

print(f"Found {len(runs)} runs in MLflow")

# Extract data from runs
data = []
for run in runs:
    # Skip final evaluation runs because this one is on the test data.
    if run.data.params.get("final_evaluation") == "true":
        continue
    
    # Get run info
    run_name = run.info.run_name
    if not run_name:
        continue
    
    # Parse run name to get model and feature set
    parts = run_name.split("_")
    if parts[-1] in ["basic", "subset", "squared"]:
        feature_set = parts[-1]
        model_name = "_".join(parts[:-1])
    else:
        if "best" in parts or "final" in parts:
            if len(parts) >= 3 and parts[-2] in ["basic", "subset", "squared"]:
                feature_set = parts[-2]
                model_name = "_".join(parts[:-2])
            else:
                continue
        else:
            continue
    
    # Get metrics and parameters
    val_accuracy = run.data.metrics.get("val_accuracy", None)
    val_f1 = run.data.metrics.get("val_f1", None)
    train_accuracy = run.data.metrics.get("train_accuracy", None)
    train_f1 = run.data.metrics.get("train_f1", None)
    
    # Check if hyperparameter tuning was performed
    optimized = "optimized" in run.data.params and run.data.params["optimized"] == "true"
    
    # Create run label for plotting
    run_label = f"{model_name}_{feature_set}"
    if optimized:
        run_label += "_optimized"
    
    # Add to temporary data
    data.append({
        "run_id": run.info.run_id,
        "model_name": model_name,
        "feature_set": feature_set,
        "optimized": optimized,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "run_label": run_label
    })

# Convert to DataFrame and drop duplicates
df = pd.DataFrame(data)
if df.empty:
    print("No valid runs found.")
    exit()

df = df.drop_duplicates(subset=["model_name", "feature_set", "optimized"], keep="last")

print(f"Processed {len(df)} valid runs")

#  Create a model comparison chart based on feature sets
try:
    base_df = df[~df["optimized"]]
    if not base_df.empty:

        pivot = base_df.pivot(index="model_name", columns="feature_set", values="val_f1")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind="bar", ax=ax)

        plt.title("Model Comparison by Feature Set (F1 Score)")
        plt.xlabel("Model")
        plt.ylabel("Validation F1 Score")
        plt.ylim(0, 1.0)
        plt.legend(title="Feature Set")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        
 
        plt.savefig("reports/model_comparison.png")
        plt.close()
        print("Created model comparison chart")
except Exception as e:
    print(f"Could not create model comparison chart: {e}")

#  Create a comparison of all runs  
try:
    
    # Sort by validation F1 score
    sorted_df = df.sort_values(by="val_f1", ascending=False)
    

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))  

    y_pos = range(len(sorted_df))
    ax.barh(y_pos, sorted_df["val_f1"], height=0.4, label="Validation F1", color="blue", alpha=0.7)
    ax.barh([y + 0.4 for y in y_pos], sorted_df["train_f1"], height=0.4, label="Training F1", color="green", alpha=0.7)
    

    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels(sorted_df["run_label"])
    

    ax.set_title("All Runs Comparison (F1 Score)")
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("reports/all_runs_comparison.png")
    plt.close()
    print("Created all runs comparison chart")
except Exception as e:
    print(f"Could not create all runs comparison chart: {e}")

#  Create a performance summary table as columnar data 
try:
    summary = df.groupby(["model_name", "feature_set", "optimized"]).agg({
        "val_accuracy": "mean",
        "val_f1": "mean",
        "train_accuracy": "mean",
        "train_f1": "mean"
    }).reset_index()
    
    # Save as CSV
    summary.to_csv("reports/model_summary.csv", index=False)
    print("Created performance summary table")
except Exception as e:
    print(f"Could not create performance summary: {e}")

#  Create a combined report on .md format 
try:
    # Get the best model
    best_row = df.loc[df["val_f1"].idxmax()]
    
    # Get final model info 
    final_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.final_evaluation = 'true'"
    )
    
    if final_runs:
        final_run = final_runs[0]
        test_accuracy = final_run.data.metrics.get("test_accuracy", None)
        test_f1 = final_run.data.metrics.get("test_f1", None)
        final_model_info = True
    else:
        test_accuracy = None
        test_f1 = None
        final_model_info = False
    

    report = f"""# MLflow Experiment Report

## Best Model
- Model: {best_row['model_name']}
- Feature Set: {best_row['feature_set']}
- Validation F1: {best_row['val_f1']:.4f}
- Validation Accuracy: {best_row['val_accuracy']:.4f}
"""

    if final_model_info:
        report += f"""
## Final Model Evaluation
- Test F1: {test_f1:.4f}
- Test Accuracy: {test_accuracy:.4f}
"""

    report += """
## Visualizations
- Model comparison chart: ![Model Comparison](model_comparison.png)
- All runs comparison: ![All Runs](all_runs_comparison.png)

## Model Performance Summary
"""
    
    # Added the model summary table to the report
    model_table = summary.to_markdown(index=False, floatfmt=".4f")
    report += model_table
    
    # Write combined report
    with open("reports/experiment_report.md", "w") as f:
        f.write(report)
    print("Created combined report")
except Exception as e:
    print(f"Could not create combined report: {e}")

print("Done! All outputs saved to the 'reports' directory")

