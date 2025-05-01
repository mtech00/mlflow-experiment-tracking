
# script to create a model card 

import os
import mlflow
import json
from datetime import datetime

# Connect to MLflow Server
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = mlflow.MlflowClient()

# Get registered model info
model_name = "iris_classification_best_model"
version = client.get_latest_versions(model_name)[0]
run = client.get_run(version.run_id)

# Create model card
model_card = {
    "model_name": model_name,
    "version": version.version,
    "created_at": datetime.now().strftime("%Y-%m-%d"),
    "metrics": {k.replace("metrics.", ""): v for k, v in run.data.metrics.items()},
    "parameters": run.data.params,
    "run_id": version.run_id
}

# Save files model card ; Model card for the best model; additionally, a release info file for the Git tag creator.
with open(f"model_card_v{version.version}.json", 'w') as f:
    json.dump(model_card, f, indent=2)

with open("model_release_info.json", 'w') as f:
    json.dump({"model_name": model_name, "version": version.version}, f)

print(f"Created model card for {model_name} v{version.version}")
