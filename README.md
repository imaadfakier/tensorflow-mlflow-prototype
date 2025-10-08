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

<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 55 52" src="https://github.com/user-attachments/assets/796f7539-b10f-44ef-80ed-d3c16b4fc6e6" />
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 56 19" src="https://github.com/user-attachments/assets/feef7dd2-3613-49ea-85a0-c51d9fbc7094" />
---
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 54 08" src="https://github.com/user-attachments/assets/8b06fda7-54cd-484c-aaa2-fd9095715f87" />
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 54 13" src="https://github.com/user-attachments/assets/6e7a89fa-8a92-47a8-8dac-b75945ead027" />
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 53 12" src="https://github.com/user-attachments/assets/6f6bb11e-a01c-4d1b-9897-ed5c56a85aa8" />
---
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 54 22" src="https://github.com/user-attachments/assets/8ab1752a-af6b-4cd9-9e05-a9146800a44f" />
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 54 27" src="https://github.com/user-attachments/assets/d7efba6f-c50a-4ca9-b39b-75efd45e21a5" />
<img width="1453" height="760" alt="Screenshot 2025-10-07 at 17 53 30" src="https://github.com/user-attachments/assets/eacce1bb-f23f-4b0d-8177-0032dc6ab096" />


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
