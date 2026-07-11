"""
InternGrow Machine Learning Track - Task 3
Advanced OCR Engine (Handwritten Character Recognition)
------------------------------------------------
Identifies handwritten digits using a Convolutional Neural Network (CNN).

Dataset: scikit-learn's `load_digits` (1,797 8x8 grayscale images of
handwritten digits 0-9, derived from the same family of data as MNIST).
This is used instead of downloading the full MNIST/EMNIST dataset so the
script runs fully offline. To use full-size 28x28 MNIST or EMNIST
characters instead, replace `load_data()` with
`tf.keras.datasets.mnist.load_data()` (requires internet access) - the
rest of the pipeline (model, training, evaluation) works unchanged.

Approach: A CNN built with TensorFlow/Keras.

Author: (your name here)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# --------------------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# --------------------------------------------------------------------------
def load_data():
    digits = load_digits()
    X = digits.images  # shape (n_samples, 8, 8)
    y = digits.target
    print(f"Loaded {X.shape[0]} images of shape {X.shape[1]}x{X.shape[2]}")

    # Normalize pixel values to [0, 1] and add channel dimension for CNN
    X = X.astype("float32") / 16.0  # pixel range is 0-16 in this dataset
    X = np.expand_dims(X, axis=-1)  # (n, 8, 8, 1)
    return X, y


def show_sample_images(X, y, out_path="sample_digits.png"):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Sample digit images saved to {out_path}")


# --------------------------------------------------------------------------
# 2. BUILD CNN MODEL
# --------------------------------------------------------------------------
def build_cnn(input_shape=(8, 8, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# 3. TRAIN, EVALUATE, VISUALIZE
# --------------------------------------------------------------------------
def plot_training_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Training history plot saved to {out_path}")


def plot_confusion_matrix(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Digit Recognition")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")


def main():
    X, y = load_data()
    show_sample_images(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    model = build_cnn()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)

    model.save("handwritten_digit_cnn.keras")
    print("\nModel saved to handwritten_digit_cnn.keras")


if __name__ == "__main__":
    main()
