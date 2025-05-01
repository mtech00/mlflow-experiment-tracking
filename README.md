
![MLflow Logo](https://www.mlflow.org/docs/2.1.1/_static/MLflow-logo-final-black.png)

# MLFlow Experiment Tracking for Iris Classification

This repository demonstrates a comprehensive experiment tracking workflow using MLFlow for machine learning model development. The project showcases how to systematically track, compare, and manage multiple experiments and store models while working with the classic Iris classification dataset, followed by implementing a pseudo-release process for the best model.

## Project Overview

At its core, this project illustrates how MLFlow transforms the machine learning development process by providing:

1.  **Structured Experiment Tracking**: Every model training run is systematically recorded with associated metrics, parameters, and artifacts
2.  **Automated Metric Logging**: Performance metrics like accuracy and F1 score are captured for both training and validation sets
3.  **Parameter Tracking**: All model configurations and hyperparameters are stored for reproducibility
4.  **Artifact Management**: Visualizations and model files are preserved as artifacts
5.  **Model Registry**: The best-performing models are registered for versioning and deployment

The system uses a Docker Compose architecture with separate containers for the MLFlow tracking server and the experimentation environment, ensuring clean separation of concerns.

## 1. Project Structure Overview

The repository follows a well-organized structure that separates different aspects of the experimentation pipeline:

```
project_root/
│
├── artifacts/                          # MLFlow artifact storage
│   └── 584969361315763526/             # Experiment ID
│       └── a8c9dd7fe42f4e99ab1413c065b12b09/   # Run ID
│           └── artifacts/              # Run artifacts
│               └── model/              # Stored model files
│                   ├── MLmodel         # Model metadata
│                   ├── conda.yaml      # Conda environment specs
│                   ├── model.pkl       # Serialized model
│                   ├── python_env.yaml # Python environment
│                   └── requirements.txt # Dependencies
│
├── mlruns/                             # MLFlow experiment tracking data
│   ├── 584969361315763526/            # Experiment data
│   │   ├── [run_ids]/                 # Individual run data
│   │   │   ├── meta.yaml              # Run metadata
│   │   │   ├── metrics/               # Logged metrics
│   │   │   ├── params/                # Logged parameters
│   │   │   └── tags/                  # Run tags
│   │   └── meta.yaml                  # Experiment metadata
│   └── models/                         # MLFlow model registry
│       └── iris_classification_best_model/  # Registered model
│           ├── aliases/               # Model aliases
│           ├── meta.yaml              # Model metadata
│           └── version-1/             # Model version data
│
├── reports/                           # Generated reports
│   ├── experiment_report.md           # Main experiment report
│   ├── model_comparison.png           # Visualization of model comparison
│   ├── all_runs_comparison.png        # Comparison of all experiments groupby feature sets
│   └── model_summary.csv              # Tabular summary of model performances
│
├── .gitignore                          # Git ignore configuration
├── docker-compose.yml                  # Docker Compose configuration
├── model_card_v1.json                  # Model card for release model
├── model_release_info.json             # Model release metadata for git tag script
├── requirements.txt                    # Project dependencies
├── run_experiments.py                  # Main experiment script
├── create_model_card.py                # Model card generation script
├── create_git_tag.py                   # Git tag creation and push script
└── create_report.py                    # Report generation script


``` 
## MLFlow: What Should Be Done

## Tasks

1.  **Conduct several experiments**:
    
    -   Use different features
    -   Do hyperparameter search
    -   Try different models/network architectures
2.  **Track results** of each experiment in MLFlow
    
3.  **Make a pseudo-release** of the best model code so far:
    
    -   Clean up code
    -   Create a separate release branch/tag
    -   Or use another release strategy
4.  **Store artifacts** of the best model
    

## Criteria

-   Experiments should be **fully reproducible**
-   Both **client and server are implemented**:
    -   Two running containers:
        -   One with your model and pipeline
        -   One with just an MLFlow server
-   Include **visualizations / reports** on experiment results

## MLFlow: The Central Component

MLFlow serves as the backbone of this project, addressing common challenges in machine learning experimentation:

### Problem: Tracking Experiments Across Multiple Iterations

Without proper tracking, data scientists often lose track of which parameters led to the best results, especially when running many experiments. In our scenario, we will try different combinations of feature/model sets before applying hyperparameter tuning for the best option, which leads to many experiments that must be tracked.

### Solution: MLFlow's Experiment Tracking

```python
# Example from run_experiments.py
with mlflow.start_run(run_name=f"{model_name}_{feature_set_name}"):
    # Log parameters
    mlflow.log_param("feature_set", feature_set_name)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)


```

This approach ensures every experimental configuration is tracked with its corresponding performance metrics, creating a comprehensive history of all model iterations.

### Problem: Managing Model Versions

As models improve, keeping track of different versions becomes challenging.

### Solution: MLFlow's Model Registry

#### MLFlow Model Registry

The project registers the best model in the MLFlow model registry:

```python
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


```

This registry approach provides several advantages:

1.  It centralizes model storage
    
2.  It enables versioning of models with important aliasing functionality. While a version typically matches a specific version ID, an alias like 'production' always refers to the currently relevant model. This means even if you change the model architecture, the 'production' alias continues to point to the current best model, rather than requiring hard-coded version numbers. This approach provides an opportunity for easy rollbacks if problems arise, without needing to change any code that references the model. [MLflow Documentation: Using Registered Model Aliases](https://mlflow.org/docs/latest/model-registry/#using-registered-model-aliases)
      
3.  It connects models back to their originating experiments
    

The MLFlow Model Registry becomes a single source of truth for model deployment.

The model registry provides a centralized repository for model versions, making it easy to track which model is currently in production and its performance characteristics.

#### Artifact Logging

```python
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


```
![All Runs Comparison](https://raw.githubusercontent.com/mtech00/MLE-24-25/main/module-3-experiment-tracking/reports/all_runs_comparison.png)



The project logs visualizations as artifacts, providing valuable insights into model behavior. The confusion matrices show how well each model discriminates between different Iris species, making the results more interpretable.

## Experiment Design and Results


1.  **Multiple Model Types**: Testing both Logistic Regression and Random Forest classifiers
2.  **Feature Engineering Variations**: Experimenting with three different feature engineering approaches:
    -   Basic features (original Iris dataset features)
    -   Subset features (partial feature selection)
    -   Squared features (adding polynomial terms)

![Model Comparison](https://raw.githubusercontent.com/mtech00/MLE-24-25/main/module-3-experiment-tracking/reports/model_comparison.png)

3.  **Hyperparameter Optimization**: We used Grid search for simplicity, but in production this is a very inefficient approach. Using HyperOpt with Tree-structured Parzen Estimator (Bayesian optimization) would be more efficient for finding the best model configurations, as it intelligently samples the parameter space rather than exhaustively testing all combinations.
### MLFlow Experiment Structure

The MLFlow experiment `iris_classification` contains multiple runs organized by:

-   Model type (logistic_regression, random_forest)
-   Feature set (basic, subset, squared)

The MLFlow UI provides an interactive way to explore these results, filter experiments, and compare performance across different runs.

![MLflow Run Comparison](https://mlflow.org/docs/latest/assets/images/intro-run-comparison-343538373d1561da97daf77bf670fc54.png) [Source: MLflow Documentation - Run Comparison UI](https://mlflow.org/docs/latest/getting-started/intro-quickstart.html)

This comprehensive artifact storage ensures that the model can be reliably loaded in the future, even as dependencies evolve.

#### Model Card Generation

The `create_model_card.py` script generates comprehensive model documentation:

```python
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


```

The generated model card in `model_card_v1.json` captures essential information:

```json
{
  "model_name": "iris_classification_best_model",
  "version": "1",
  "created_at": "2025-04-10",
  "metrics": {
    "test_accuracy": 0.9666666666666667,
    "test_f1": 0.9664109121909632
  },
  "parameters": {
    "C": "10.0",
    "feature_set": "basic",
    "final_evaluation": "True",
    "solver": "lbfgs"
  },
  "run_id": "a8c9dd7fe42f4e99ab1413c065b12b09"
}


```

This documentation is crucial for model governance, helping future users understand the model's capabilities, limitations, and configuration.

#### Git-Based Versioning with Tags

Another good practice is Git Hash Tagging in MLflow.

To track code versions with your models, add Git commit hashes as MLflow tags. This one-liner captures the current commit hash and adds it to your run:

```python
# Inside your MLflow run
mlflow.set_tag("git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode())

```

This creates a clear link between models and code, making experiments fully reproducible by allowing you to check out the exact code version. Another option is simply saving code as an artifact to save training code:
```
mlflow.log_artifact("train.py")
```
The `create_git_tag.py` script implements Git-based versioning for model releases:

```python
# Load model info
with open("model_release_info.json", 'r') as f:
    info = json.load(f)
model_name = info["model_name"]
version = info["version"]
tag_name = f"model-v{version}"

print(f"Creating git tag for {model_name} version {version}")
try:
    # Create tag depends on model info 
    subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Model release v{version}"], check=True)
    print(f"Created tag: {tag_name}")
    
    # Push tag if requested
    if args.push:
        print(f"Pushing tag to origin...")
        subprocess.run(["git", "push", "origin", tag_name], check=True)


```

This approach connects model versions to specific code states in the Git repository, ensuring complete reproducibility. When a model is tagged, it becomes possible to check out exactly the code that produced it, which is essential for debugging and auditing. It also gives opportunity for releases.

## MLFlow's Artifact Tracking

Beyond tracking metrics and parameters, MLFlow stores important artifacts:

1.  **Confusion Matrices**: Visualizations showing model prediction errors
2.  **Serialized Models**: Pickled model files for later deployment (in this project only for selected production model)
3.  **Environment Information**: Dependencies and requirements for reproducing the environment

### Comprehensive Model Artifact Storage

The project saves complete model artifacts with all necessary context:

```python
mlflow.sklearn.log_model(
    best_model, 
    "model",
    registered_model_name="iris_classification_best_model"
)


```

The artifact storage includes:

-   `model.pkl`: The serialized model file
-   `MLmodel`: Metadata about the model format and loading mechanism
-   `conda.yaml`: Conda environment specification
-   `python_env.yaml`: Python environment specification
-   `requirements.txt`: Package dependencies

## MLflow Signature Enforcement

In our project, we did not apply this but it's absolutely worth mentioning. Signature enforcement is a critical feature in MLflow that validates incoming prediction requests against your model's expected input schema. If the data doesn't match the expected format, MLflow rejects the request and returns an error instead of producing potentially meaningless predictions.

## MLFlow Server Configuration

## Architecture



### Artifact Storage on the Host File System

MLflow provides functionality to easily create Docker images without customization, but this approach is often inefficient and not considered best practice except for rapid prototyping scenarios. In today's era of advanced code generation tools, creating a custom Docker base image can be just as straightforward as using MLflow's built-in Docker image creation.

[MLflow Documentation: Building Docker Images for Models](https://mlflow.org/docs/latest/getting-started/quickstart-2/#build-a-container-image-for-your-model)

For example, a model can be containerized using:

```bash
mlflow models build-docker --model-uri "models:/wine-quality/1" --name "qs_mlops"

```

```
┌─────────────────────┐     ┌─────────────────────┐
│  MLFlow Server      │     │  Model Service      │
│  - Tracking UI      │     │  - Experiments      │
│  - Model Registry   │◄────┤  - Model Training   │
│  - Artifact Storage │     │  - Reporting        │
└─────────────────────┘     └─────────────────────┘

```

MLflow can store metadata and artifacts through various configuration options. In this experiment, we use localhost for simplicity, though this approach has limited utility in real-world scenarios. A better practice is implementing a remote tracking server, which enhances collaborative operations across teams and ensures consistent data storage. Another option is leveraging managed infrastructure like Databricks, which provides an integrated MLflow platform with additional enterprise functionality.

![MLflow Tracking](https://mlflow.org/docs/latest/assets/images/tracking-setup-overview-3d8cfd511355d9379328d69573763331.png) [Source: MLflow Documentation - Tracking Server Setup](https://mlflow.org/docs/latest/tracking/tracking-server.html)

After configuring the core MLflow system, we can also customize the artifact storage approach. For scenarios where we frequently access artifacts without needing the full MLflow core functionality, we can implement direct access to the artifact repository. This configuration pattern improves efficiency by eliminating unnecessary intermediary steps in the artifact retrieval process.

![MLflow Tracking (No Artifact Server)](https://mlflow.org/docs/latest/assets/images/tracking-setup-no-serve-artifacts-9e21c03b857275a42dc667e4454fba37.png) [Source: MLflow Documentation - Tracking Server Configuration](https://mlflow.org/docs/latest/tracking/tracking-server.html)

MLflow provides an official, preconfigured base image that can be found in the [GitHub MLflow Container Registry](https://github.com/mlflow/mlflow/pkgs/container/mlflow). This image contains all necessary dependencies and configurations for running MLflow services without additional setup.

-   **mlflow-server**: Provides the tracking server, UI, and artifact storage

```yaml
mlflow-server:
  image: python:3.9-slim
  ports:
    - "5000:5000"
  volumes:
    - ./mlruns:/mlruns
    - ./artifacts:/artifacts
  environment:
    - MLFLOW_TRACKING_URI=http://localhost:5000
  command: >
    bash -c "apt-get update && apt-get install -y gcc python3-dev && 
             pip install mlflow==2.8.0 && 
             mlflow server --host 0.0.0.0 --port 5000 
             --backend-store-uri file:///mlruns 
             --default-artifact-root /artifacts"

```

This setup provides:

-   A persistent storage for experiments in the `mlruns` directory
-   Artifact storage in the `artifacts` directory
-   Web UI access on port 5000

### 3.1 Full Reproducibility

The project ensures reproducibility through several mechanisms: Containerized Environment The containers share volumes for MLFlow data and artifacts, ensuring seamless integration.

Controlled Randomness with fixed seeds, ensuring deterministic results.

### Systematic Experimentation Workflow

The `run_experiments.py` script implements a complete experimentation workflow: This implementation follows a systematic approach:

1.  **Setup**: Configuring MLFlow and creating an experiment
2.  **Data Preparation**: Loading, splitting, and preprocessing data
3.  **Feature Engineering**: Creating multiple feature sets
4.  **Model Definition**: Setting up different model architectures
5.  **Experimentation**: Training models on different feature sets
6.  **Optimization**: Tuning hyperparameters for the best approach
7.  **Final Evaluation**: Testing the optimized model on holdout data
8.  **Model Registration**: Registering the best model for deployment

## Automated Reporting from MLFlow Data

The project includes scripts to generate reports directly from MLFlow experiment data:

```python
# Extract data from MLFlow runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=""
)

# Process and visualize the results
for run in runs:
    run_name = run.info.run_name
    val_accuracy = run.data.metrics.get("val_accuracy", None)
    val_f1 = run.data.metrics.get("val_f1", None)
    # Generate visualizations and reports


```

## Running the MLFlow Environment

To start experimenting with MLFlow tracking:

1.  **Start the containers**:
    
    ```bash
    docker-compose up 
    
    
    ```
    
    then switch to another terminal to
    
2.  **Run experiments**:
    
    ```bash
    docker exec test-model-service-1 python run_experiments.py
    
    
    ```
    
3.  **Generate reports**:
    
    ```bash
    docker exec test-model-service-1 python create_report.py
    
    
    ```
    
4.  **Create model card**:
    
    ```bash
    docker exec test-model-service-1 python create_model_card.py
    
    
    ```
5.  **Create git tag and push**:
    
    ```bash
    python create_git_tag.py  --push 
    ```    


Due to Git integration complexity within containers, we'll handle the tagging and pushing operations on the host system using only base Python libraries. The `--push` flag is optional; i

This approach connects model versions to specific code states in the Git repository, ensuring complete reproducibility. When a model is tagged, it becomes possible to check out exactly the code that produced it, which is essential for debugging and auditing. It also provides an opportunity for creating formal releases.

After creating the tag, you can generate a formal release from it through your Git platform. In our example, you can find such a release at [https://github.com/mtech00/MLE-24-25/releases/tag/model-v1](https://github.com/mtech00/MLE-24-25/releases/tag/model-v1). This process automatically packages everything as a compressed archive, and Git tracks all changes through the tagging system, making version management seamless.




5.  **Access the MLFlow UI**:
    
    ```
    http://localhost:5000
    
    
    ```
    

In the MLFlow UI, you can:

-   Compare experiments side-by-side
-   Sort and filter by metrics
-   View parameter configurations
-   Download artifacts
-   Register and manage models

## Benefits of MLFlow for ML Development

This project demonstrates several key benefits of using MLFlow:

1.  **Reproducibility**: Every experiment is fully documented with code, data, and environment
2.  **Collaboration**: Team members can view and build upon each other's experiments
3.  **Efficiency**: Less time spent on manual tracking and more on model improvement
4.  **Governance**: Complete history of model development for auditing and compliance
5.  **Deployment Readiness**: Smooth transition from experimentation to production

## Conclusion and Future Work

## 7. Conclusion

This project demonstrates a comprehensive implementation of MLFlow for experiment tracking and model management. It satisfies all specified requirements:

1.  **Multiple Experiment Types**:
    -   Different feature engineering approaches
    -   Hyperparameter optimization
    -   Multiple model architectures
2.  **Comprehensive MLFlow Tracking**:
    -   Parameter logging
    -   Metric tracking
    -   Artifact management
3.  **Model Release Strategy**:
    -   MLFlow model registry
    -   Git tagging for versions
    -   Model card generation
4.  **Artifact Storage**:
    -   Model files
    -   Environment specifications
    -   Visualization artifacts

### Future Enhancements

1.  **MLFlow Serving**: Add model serving capabilities for real-time inference

# Serving Models with MLflow

MLflow makes it simple to deploy trained models as REST APIs. Here are the main ways to serve your models:

```bash
# Local serving with MLflow CLI
mlflow models serve -m "models:/iris_classification_best_model/production" -p 5000

# With signature enforcement enabled
mlflow models serve -m "models:/iris_classification_best_model/production" --enable-mlserver -p 5000

# Deploy as a Docker container
mlflow models build-docker -m "models:/iris_classification_best_model/production" -n "iris-classifier"
docker run -p 5000:8080 iris-classifier

```

Once deployed, you can make prediction requests:

```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type:application/json" \
  -d '{"columns":["sepal_length","sepal_width","petal_length","petal_width"], 
       "data":[[5.1, 3.5, 1.4, 0.2]]}'

```

MLflow serving supports various deployment patterns including batch inference, real-time predictions, and A/B testing through model aliases.

2.  **Advanced Tracking**: Implement dataset versioning and model lineage tracking
3.  **Automated Retraining**: Set up scheduled retraining using tracked experiments
4.  **Performance Monitoring**: Integrate drift detection and model performance monitoring



## References

-   [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
-   [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/model-registry.html)
-   [MLflow Tracking Server Setup](https://mlflow.org/docs/latest/tracking/tracking-server.html)
-   [MLflow Docker Container Registry](https://github.com/mlflow/mlflow/pkgs/container/mlflow)
-   [MLflow Quickstart Guide](https://mlflow.org/docs/latest/getting-started/quickstart-2/)
