"""
train.py
--------
Trains a food image classifier on the Food-11 dataset using transfer
learning (MobileNetV2 pretrained on ImageNet).

Expected dataset layout (see prepare_data.py if you need to build this):

    dataset/
        train/
            Bread/
            Dairy product/
            Dessert/
            Egg/
            Fried food/
            Meat/
            Noodles-Pasta/
            Rice/
            Seafood/
            Soup/
            Vegetable-Fruit/
        val/
            ... same class folders ...
        test/                (optional, used for final evaluation)
            ... same class folders ...

Usage:
    python train.py --data_dir ./dataset --epochs 15 --fine_tune_epochs 10
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # so it works headless
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import CLASS_NAMES, IMG_SIZE, BATCH_SIZE, MODEL_DIR, MODEL_PATH, save_class_indices


def build_data_generators(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for the first training phase

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history, fine_tune_history, out_path="training_curves.png"):
    acc = history.history["accuracy"] + (fine_tune_history.history["accuracy"] if fine_tune_history else [])
    val_acc = history.history["val_accuracy"] + (fine_tune_history.history["val_accuracy"] if fine_tune_history else [])
    loss = history.history["loss"] + (fine_tune_history.history["loss"] if fine_tune_history else [])
    val_loss = history.history["val_loss"] + (fine_tune_history.history["val_loss"] if fine_tune_history else [])

    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Food-11 classifier (MobileNetV2 transfer learning).")
    parser.add_argument("--data_dir", default="./dataset", help="Path to reorganized dataset (train/val/test folders)")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for the frozen-base phase")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Epochs for the fine-tuning phase (0 to skip)")
    parser.add_argument("--fine_tune_at", type=int, default=100, help="Unfreeze base model layers from this index onward")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the frozen-base phase")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for the fine-tuning phase")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Building data generators...")
    train_gen, val_gen = build_data_generators(args.data_dir)
    save_class_indices(train_gen.class_indices)
    print("Class indices:", train_gen.class_indices)

    print("Building model...")
    model, base_model = build_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\n=== Phase 1: training classifier head (base frozen) ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    fine_tune_history = None
    if args.fine_tune_epochs > 0:
        print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
        base_model.trainable = True
        for layer in base_model.layers[: args.fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    # Final save (in case ModelCheckpoint didn't trigger on the very last epoch)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    plot_history(history, fine_tune_history)

    # Optional: evaluate on test set if present
    test_dir = os.path.join(args.data_dir, "test")
    if os.path.isdir(test_dir):
        print("\nEvaluating on test set...")
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=CLASS_NAMES,
            shuffle=False,
        )
        loss, acc = model.evaluate(test_gen)
        print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
