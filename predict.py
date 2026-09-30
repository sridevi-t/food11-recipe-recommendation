"""
predict.py
----------
Loads the trained model and predicts the food class + confidence for a
given image. Can be used as a library (import predict_image) or from the
command line.

Usage:
    python predict.py --image path/to/photo.jpg
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from utils import IMG_SIZE, MODEL_PATH, load_class_indices

_model = None
_idx_to_class = None


def _load_model_and_labels():
    """Lazily load the model + label map once, then cache them."""
    global _model, _idx_to_class
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _idx_to_class is None:
        _idx_to_class = load_class_indices()
    return _model, _idx_to_class


def predict_image(image_path: str, top_k: int = 3):
    """
    Predict the food class for a single image.

    Returns:
        dict with:
            "label": top predicted class name (str)
            "confidence": top prediction probability (float, 0-1)
            "top_k": list of (class_name, probability) tuples, sorted desc.
    """
    model, idx_to_class = _load_model_and_labels()

    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]  # shape (num_classes,)

    top_indices = preds.argsort()[::-1][:top_k]
    top_k_results = [(idx_to_class[str(i)], float(preds[i])) for i in top_indices]

    return {
        "label": top_k_results[0][0],
        "confidence": top_k_results[0][1],
        "top_k": top_k_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict food class for an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--top_k", type=int, default=3, help="How many top predictions to show")
    args = parser.parse_args()

    result = predict_image(args.image, top_k=args.top_k)

    print(f"\nPredicted: {result['label']}  (confidence: {result['confidence'] * 100:.2f}%)")
    print("\nTop predictions:")
    for name, prob in result["top_k"]:
        print(f"  {name:<20s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
