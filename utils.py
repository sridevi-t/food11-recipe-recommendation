"""
utils.py
--------
Shared constants and helper functions used across the project.
"""

import os
import json

# Food-11 classes in the same order as the folders in the dataset
CLASS_NAMES = [
    "apple_pie",
    "cheesecake",
    "chicken_curry",
    "french_fries",
    "fried_rice",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "omelette",
    "pizza",
    "sushi",
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "food11_mobilenetv2.h5")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")
RECIPES_DB_PATH = os.path.join(
    os.path.dirname(__file__), "recipes_db.json"
)


def save_class_indices(class_indices: dict):
    """Save class index mapping produced by Keras."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    idx_to_class = {
        v: k for k, v in class_indices.items()
    }

    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)


def load_class_indices() -> dict:
    """Load the saved index-to-class mapping."""
    with open(CLASS_INDEX_PATH, "r") as f:
        return json.load(f)


def load_recipes_db() -> dict:
    """Load the recipe database."""
    with open(RECIPES_DB_PATH, "r") as f:
        return json.load(f)