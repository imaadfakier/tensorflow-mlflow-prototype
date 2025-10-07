# TensorFlow + MLflow Sprint

A machine learning project using **TensorFlow** for model training and **MLflow** for experiment tracking, logging, and model management.  
This project provides a complete ML workflow, from training models to logging metrics, checkpoints, and artifacts.

---

## Features

- **TensorFlow Model Training**: Train deep learning models with flexible configurations.
- **MLflow Experiment Tracking**: Log metrics, parameters, artifacts, checkpoints, and tensorboard events.
- **Model Management**: Save and load models via MLflow's versioned artifacts.
- **Reproducibility**: Maintain experiment history with detailed logging.

---

## Project Structure

```

.
├── train.py # Main script for training models
├── requirements.txt # Python dependencies
├── my_model.keras # Example trained model
├── mlruns/ # MLflow experiment logs
│ ├── <experiment_id>/ # Experiment directories
│ │ ├── <run_id>/ # Individual run directories
│ │ │ ├── artifacts/ # Model artifacts, checkpoints, tensorboard logs
│ │ │ ├── metrics/ # Logged metrics (accuracy, loss, val_accuracy, val_loss, etc.)
│ │ │ ├── params/ # Model hyperparameters
│ │ │ ├── tags/ # MLflow tags and metadata
│ │ │ └── inputs/ # Input metadata
├── models/ # Saved models and MLflow-managed model artifacts

```

> **Note:** MLflow automatically creates the `mlruns/` folder when experiments are run.

---

## Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/tf-mlflow-sprint.git
cd tf-mlflow-sprint
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Start MLflow UI (optional)**

```bash
mlflow ui
# Access at http://localhost:5000
```

---

## Usage

- **Train a model**

```bash
python train.py
```

- **View experiment metrics & checkpoints**
  Navigate to `mlruns/` or use the MLflow UI:

```bash
mlflow ui
```

- **Load a saved MLflow model**

```python
import mlflow
model = mlflow.keras.load_model("runs:/<RUN_ID>/model")
predictions = model.predict(X_test)
```

---

## MLflow Artifacts

- **Checkpoints**: Latest checkpoints stored in `artifacts/checkpoints/`
- **TensorBoard Logs**: `artifacts/tensorboard_logs/` for visualizing training
- **Model Summaries**: `artifacts/model_summary.txt`
- **Metrics & Params**: Detailed logs under `metrics/` and `params/`
- **Saved Models**: Versioned under `models/` and `mlruns/<experiment>/<run>/artifacts/`

---

## Dependencies

- Python 3.11+
- TensorFlow
- MLflow
- numpy, pandas, scikit-learn

---

## License

MIT License
