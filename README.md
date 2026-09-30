# 🍲 Food-11 Image Classification and Recipe Recommendation System

## 📌 Project Overview

The **Food-11 Image Classification and Recipe Recommendation System** is a deep learning-based application that identifies food items from images and recommends suitable recipes.

The system uses the **Food-11 dataset** and a **MobileNetV2 Convolutional Neural Network (CNN)** with Transfer Learning for image classification.

After identifying the food item, the system provides recipe recommendations based on the predicted food category. Users can also filter recipes based on **vegetarian/non-vegetarian preference** and **cuisine**.

The project is implemented as an interactive **Streamlit web application**.

---

## 🎯 Objectives

* Classify food images into 11 food categories.
* Use Transfer Learning to improve image classification.
* Display the predicted food and confidence score.
* Show the top-3 prediction results.
* Recommend recipes based on the predicted food.
* Filter recipes by dietary preference.
* Filter recipes by cuisine.
* Provide an easy-to-use web interface.

---

## 🍴 Food Categories

The model classifies images into the following 11 categories:

1. Apple Pie
2. Cheesecake
3. Chicken Curry
4. French Fries
5. Fried Rice
6. Hamburger
7. Hot Dog
8. Ice Cream
9. Omelette
10. Pizza
11. Sushi

---

## 🧠 Algorithm and Methods

### 1. Convolutional Neural Network (CNN)

CNNs are widely used for image classification. They automatically learn visual features such as edges, shapes, textures, and patterns from images.

### 2. MobileNetV2

**MobileNetV2** is a lightweight CNN architecture designed for efficient image classification.

It is suitable for this project because it provides good performance while requiring fewer computational resources than many larger CNN architectures.

### 3. Transfer Learning

A MobileNetV2 model pretrained on **ImageNet** is used as the base model.

The pretrained layers are initially frozen and a classification layer is trained for the Food-11 classes. Selected layers are then fine-tuned to improve classification performance.

### 4. Image Augmentation

Training images are augmented using techniques such as:

* Rotation
* Width and height shifting
* Shearing
* Zooming
* Horizontal flipping

This helps the model generalize better to different food images.

### 5. Softmax Classification

The final layer produces probabilities for the 11 food categories. The class with the highest probability is selected as the predicted food.

### 6. Rule-Based Recipe Recommendation

After classification, the predicted food category is matched with the recipe database.

Recipes can be filtered using:

* Vegetarian / Non-Vegetarian
* Cuisine

---

## 🛠️ Technologies Used

| Technology         | Purpose                          |
| ------------------ | -------------------------------- |
| Python             | Programming language             |
| TensorFlow / Keras | Deep learning and model training |
| MobileNetV2        | Image classification model       |
| NumPy              | Numerical operations             |
| PIL                | Image processing                 |
| JSON               | Recipe database                  |
| Streamlit          | Web application                  |
| VS Code            | Development environment          |

---

## 📊 Model Performance

The trained MobileNetV2 model achieved:

**Test Accuracy: 82.64%**

**Test Loss: 0.6050**

The model was trained using the Food-11 dataset with image augmentation and fine-tuning.

---

## 🔄 System Workflow

```text
User uploads food image
          ↓
Image preprocessing
          ↓
MobileNetV2 model
          ↓
Food classification
          ↓
Predicted food + confidence
          ↓
Recipe database
          ↓
Apply dietary/cuisine filters
          ↓
Display recommended recipes
```

---

## 📂 Project Structure

```text
food11_recipe_system/
│
├── app.py
├── predict.py
├── prepare_data.py
├── recommend.py
├── train.py
├── utils.py
├── recipes_db.json
├── requirements.txt
├── README.md
│
├── models/
│   ├── food11_mobilenetv2.h5
│   └── class_indices.json
│
└── sample_data/
```

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/food11-recipe-recommendation.git
```

### Step 2: Open the Project

```bash
cd food11-recipe-recommendation
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 🖥️ Application Features

### 📷 Food Image Upload

Users can upload a JPG, JPEG, or PNG food image.

### 🔎 Food Prediction

The system displays:

* Predicted food
* Confidence percentage
* Top-3 predictions

### 🍴 Recipe Recommendation

Recipes related to the predicted food are displayed automatically.

### 🥦 Dietary Filter

Users can select:

* Any
* Vegetarian
* Non-Vegetarian

### 🌎 Cuisine Filter

Users can filter recipes according to available cuisine types.

---

## 📸 Application Output

The Streamlit application provides:

* Uploaded food image
* Predicted food class
* Prediction confidence
* Top-3 predictions
* Recommended recipes
* Recipe ingredients
* Preparation time
* Cooking instructions
* Dietary and cuisine filters

---

## 🔮 Future Enhancements

* Add more food categories.
* Increase the size and diversity of the recipe database.
* Improve classification accuracy with a larger dataset.
* Add nutritional information for recipes.
* Add calorie estimation.
* Add personalized recipe recommendations.
* Deploy the application online.
* Add multilingual support.
* Add voice-based interaction.

---

## 👩‍💻 Developed By

**Sri Devi T.**

**B.E. Computer Science and Engineering**

**Vivekanandha College of Engineering for Women**

**Academic Year: 2023–2027**

---

## 📜 License

This project is developed for educational and academic purposes.
