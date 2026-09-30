# Food-11 Classifier + Recipe Recommender

A complete pipeline:

```
Food Image -> Deep Learning Classification (MobileNetV2) -> Food Name + Confidence
           -> Recipe Recommendation (filterable by Vegetarian/Non-Veg & Cuisine)
```

Built for the **Food-11 dataset** (Kaggle / EPFL, 11 categories, ~9900+ images total).

## 1. Project structure

```
food11_recipe_system/
├── prepare_data.py     # reorganizes raw Food-11 files into class-named folders
├── train.py             # trains MobileNetV2 (transfer learning) on the dataset
├── predict.py            # loads the trained model, predicts class + confidence
├── recommend.py          # recipe lookup with vegetarian / cuisine filters
├── recipes_db.json       # recipe database (11 classes x 4 recipes, multiple cuisines)
├── app.py                # Streamlit web app tying everything together
├── utils.py              # shared constants / helpers (class names, paths)
├── requirements.txt
├── models/               # trained model + label map get saved here
└── sample_data/          # (empty) put a few test images here if you like
```

## 2. The 11 Food-11 classes

`Bread`, `Dairy product`, `Dessert`, `Egg`, `Fried food`, `Meat`,
`Noodles-Pasta`, `Rice`, `Seafood`, `Soup`, `Vegetable-Fruit`.

## 3. Setup

```bash
pip install -r requirements.txt
```

(GPU strongly recommended for training; CPU works but is slow.)

## 4. Prepare the dataset

Download **Food-11** from Kaggle and unzip it. Two possible layouts:

**A. Raw EPFL layout** (files like `0_0.jpg`, `1_23.jpg`, split into
`training/`, `validation/`, `evaluation/` folders) — reorganize it into
class-named subfolders that Keras expects:

```bash
python prepare_data.py --source /path/to/Food-11 --dest ./dataset
```

This produces:

```
dataset/
├── train/
│   ├── Bread/
│   ├── Dairy product/
│   └── ... (11 folders)
├── val/
│   └── ... (same 11 folders)
└── test/
    └── ... (same 11 folders)
```

**B. Already class-named folders** — some Kaggle mirrors ship the dataset
this way already. If so, just point `train.py` at that folder directly
(skip `prepare_data.py`), as long as the split folders are named
`train`, `val`, (optionally `test`) and each contains the 11 class
subfolders with the exact names listed above (edit `CLASS_NAMES` in
`utils.py` if your folder names differ slightly).

## 5. Train the model

```bash
python train.py --data_dir ./dataset --epochs 15 --fine_tune_epochs 10
```

What it does:
- Loads MobileNetV2 pretrained on ImageNet, freezes it, and trains a new
  classification head for `--epochs` epochs.
- Then unfreezes the top layers of MobileNetV2 (from `--fine_tune_at`
  onward) and fine-tunes at a lower learning rate for
  `--fine_tune_epochs` epochs.
- Uses data augmentation (rotation, shifts, zoom, flips) on the training
  set to reduce overfitting on ~9900 images.
- Saves the best model (by validation accuracy) to
  `models/food11_mobilenetv2.h5` and the label mapping to
  `models/class_indices.json`.
- Saves a training curves plot to `training_curves.png`.
- If a `test/` folder exists, evaluates final accuracy on it.

Typical expected accuracy on Food-11 with this setup: roughly 85-92%
validation accuracy after both phases, depending on epochs/augmentation
choices and hardware.

## 6. Predict on a single image (CLI)

```bash
python predict.py --image sample_data/my_food_photo.jpg --top_k 3
```

Example output:

```
Predicted: Dessert  (confidence: 94.32%)

Top predictions:
  Dessert              94.32%
  Bread                 3.10%
  Dairy product         1.58%
```

## 7. Get recipe recommendations (CLI)

```bash
python recommend.py --food_class "Meat" --veg non-veg --cuisine Indian
```

Filters:
- `--veg`: `veg` / `non-veg` (omit for no filter)
- `--cuisine`: e.g. `Indian`, `Italian`, `Chinese`, `French`, `Japanese`,
  `Korean`, `Thai`, `Spanish`, `Greek`, `American`, `Middle Eastern`,
  `Continental` (omit for no filter)
- `--top_n`: max number of recipes to return

## 8. Run the full web app

```bash
streamlit run app.py
```

This gives you:
- An image upload box
- The predicted food name + confidence (and top-3 breakdown)
- A sidebar with **Vegetarian / Non-Vegetarian / Any** and **Cuisine**
  filters
- A filtered, expandable list of matching recipes with ingredients and
  instructions

## 9. Extending the recipe database

`recipes_db.json` is a plain dictionary keyed by the 11 Food-11 class
names, each holding a list of recipe objects:

```json
{
  "name": "Butter Chicken",
  "cuisine": "Indian",
  "vegetarian": false,
  "prep_time_mins": 60,
  "ingredients": ["chicken", "tomato", "butter", "cream", "spices"],
  "instructions": "..."
}
```

Add as many recipes per class / cuisine as you like — no code changes
needed, `recommend.py` and `app.py` will pick them up automatically.

## 10. Notes & possible improvements

- Swap `MobileNetV2` in `train.py` for a heavier backbone (EfficientNetB0,
  ResNet50) if you want higher accuracy and have more compute.
- Add a "confidence threshold" in `app.py` to warn the user when the
  model is unsure (e.g. confidence < 50%).
- Connect a real recipe API (e.g. Spoonacular, Edamam) instead of the
  static `recipes_db.json` for a much larger recipe catalog — `recommend.py`
  is written so you can swap its data source without touching `app.py`.
- Add calorie/nutrition estimates per recipe for a more complete "food
  diary" style app.
