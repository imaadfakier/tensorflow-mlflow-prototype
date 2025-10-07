import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

print("--- TensorFlow and MLflow Sprint ---")

# --- 1. MLflow SETUP: Enable autologging ---
# This is the magic line. It will automatically track everything.
mlflow.tensorflow.autolog()

# --- 2. DATA LOADING & PREPROCESSING ---
# Load the classic MNIST dataset of handwritten digits
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension for the CNN
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(f"--- Data loaded. Training shape: {x_train.shape} ---")

# --- 3. MODEL BUILDING (CNN) ---
# Build a simple but effective Convolutional Neural Network
model = tf.keras.models.Sequential(
    [
        # Input layer
        tf.keras.layers.Input(shape=(28, 28, 1)),
        # First convolution layer: finds basic features like edges
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        # Pooling layer: reduces the size
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Second convolution layer: finds more complex features
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Flatten the results to feed into a dense layer
        tf.keras.layers.Flatten(),
        # A standard dense layer for classification
        tf.keras.layers.Dense(128, activation="relu"),
        # Output layer: 10 neurons for 10 digits (0-9)
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


# --- 4. MODEL COMPILATION ---
# Define how the model will learn
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print("--- Model compiled ---")
model.summary()

# --- 5. MODEL TRAINING ---
with mlflow.start_run() as run:
    print("--- Starting model training with MLflow ---")
    # Train the model for 5 epochs (quick but effective)
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the trained model
    model.save("my_model.keras")
    mlflow.log_artifact("my_model.keras")

    print("--- Training complete ---")

    # --- 6. MODEL EVALUATION ---
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"--- Final test accuracy: {acc:.4f} ---")

    # --- 7. BONUS: Log a confusion matrix for better insights ---
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    # Save the plot and log it as an artifact in MLflow
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    print("--- Confusion matrix logged to MLflow artifacts ---")
