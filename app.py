"""
app.py
------
Food-11 Image Classifier + Recipe Recommender
"""

import os

import streamlit as st
from PIL import Image

from utils import MODEL_PATH, CLASS_INDEX_PATH, load_recipes_db
from recommend import get_recommendations, get_available_cuisines


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Food-11 Classifier + Recipe Recommender",
    page_icon="🍲",
    layout="centered"
)


# ---------------------------------------------------------
# Load prediction function only when needed
# ---------------------------------------------------------

@st.cache_resource
def load_prediction_function():
    from predict import predict_image
    return predict_image


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🍲 Food Image Classifier & Recipe Recommender")

st.write(
    "Upload a food image. The AI model identifies the food "
    "and recommends matching recipes."
)


# ---------------------------------------------------------
# Check model
# ---------------------------------------------------------

model_ready = (
    os.path.exists(MODEL_PATH)
    and os.path.exists(CLASS_INDEX_PATH)
)

if not model_ready:
    st.error(
        "Trained model not found. "
        "Please make sure models/food11_mobilenetv2.h5 exists."
    )
    st.stop()


# ---------------------------------------------------------
# Load recipe database
# ---------------------------------------------------------

db = load_recipes_db()

all_cuisines = ["Any"] + get_available_cuisines(db)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("🍽️ Recipe Filters")

diet_choice = st.sidebar.radio(
    "Diet preference",
    ["Any", "Vegetarian", "Non-Vegetarian"]
)

cuisine_choice = st.sidebar.selectbox(
    "Cuisine",
    all_cuisines
)

top_n = st.sidebar.slider(
    "Number of recipes",
    min_value=1,
    max_value=10,
    value=4
)


# Convert UI choices into filters

veg_filter = None

if diet_choice == "Vegetarian":
    veg_filter = True

elif diet_choice == "Non-Vegetarian":
    veg_filter = False


cuisine_filter = (
    None
    if cuisine_choice == "Any"
    else cuisine_choice
)


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a food image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food Image",
        use_container_width=True
    )

    # Load prediction function only now
    predict_image = load_prediction_function()

    # Temporary image
    tmp_path = "temp_uploaded_image.jpg"

    image.save(tmp_path)

    try:

        with st.spinner("🔍 Analyzing food image..."):

            result = predict_image(
                tmp_path,
                top_k=3
            )

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


    # -----------------------------------------------------
    # Prediction result
    # -----------------------------------------------------

    st.success("Prediction completed!")

    st.subheader("🔎 Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Food",
            result["label"]
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )


    # -----------------------------------------------------
    # Top predictions
    # -----------------------------------------------------

    with st.expander("📊 Top-3 Predictions"):

        for name, probability in result["top_k"]:

            st.write(
                f"**{name}** — "
                f"{probability * 100:.2f}%"
            )


    st.divider()


    # -----------------------------------------------------
    # Recipe recommendations
    # -----------------------------------------------------

    st.subheader(
        f"🍴 Recommended {result['label']} Recipes"
    )

    recipes = get_recommendations(
        result["label"],
        vegetarian=veg_filter,
        cuisine=cuisine_filter,
        top_n=top_n,
        db=db
    )


    if not recipes:

        st.info(
            "No recipes match the selected filters. "
            "Try changing the diet or cuisine filter."
        )

    else:

        for recipe in recipes:

            if recipe["vegetarian"]:
                diet_tag = "🥦 Vegetarian"
            else:
                diet_tag = "🍗 Non-Vegetarian"


            with st.expander(
                f"🍽️ {recipe['name']} "
                f"— {recipe['cuisine']}"
            ):

                st.write(
                    f"**Diet:** {diet_tag}"
                )

                st.write(
                    f"**Preparation time:** "
                    f"{recipe['prep_time_mins']} minutes"
                )

                st.write(
                    "**Ingredients:**"
                )

                for ingredient in recipe["ingredients"]:
                    st.write(f"• {ingredient}")


                st.write(
                    "**Instructions:**"
                )

                st.write(
                    recipe["instructions"]
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

else:

    st.info(
        "👆 Upload a food image above to start classification."
    )


st.sidebar.markdown("---")

st.sidebar.caption(
    "Model: MobileNetV2 Transfer Learning"
)

st.sidebar.caption(
    "Dataset: Food-11"
)