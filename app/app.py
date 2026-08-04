"""
app.py

Streamlit application for the Student Health Risk predictor.

Loads the fitted preprocessing pipeline and the deployed LightGBM
model ONCE at startup (via @st.cache_resource — the Streamlit
equivalent of the "load once, not per request" pattern), collects the
13 lifestyle/physiological inputs through form widgets, applies the
SAME fitted preprocessing pipeline used during training (preventing
train/serve skew), and displays the predicted class alongside the
full 3-class probability distribution.

Run from the project root:
    streamlit run app/app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Allow importing from src/ when running `streamlit run app/app.py`
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import DEPLOYED_MODEL_PATH, FEATURE_ORDER, TARGET_CLASSES  # noqa: E402
from preprocessing import load_preprocessor, transform  # noqa: E402

st.set_page_config(page_title="Student Health Risk Predictor", page_icon="🩺", layout="centered")


# ---------------------------------------------------------------------
# Cached resources — loaded once per server process, not per request
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessor():
    preprocessor = load_preprocessor()
    model = joblib.load(DEPLOYED_MODEL_PATH)
    return preprocessor, model


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🩺 Student Health Risk Predictor")
st.caption(
    "Predicts the probability of at-risk / fit / unhealthy health status "
    "from lifestyle and physiological indicators, using a LightGBM model "
    "trained on 690,088 Kaggle records."
)

try:
    preprocessor, model = load_model_and_preprocessor()
except FileNotFoundError:
    st.error(
        "Model files not found. Train the models first "
        "(`python src/train_lightgbm.py`) so that "
        "`models/preprocessing_pipeline.pkl` and "
        "`models/lightgbm_model.pkl` exist."
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Lifestyle & Physiological Inputs")

    col1, col2 = st.columns(2)

    with col1:
        sleep_duration = st.slider("Sleep duration (hours)", 0.0, 12.0, 7.0, 0.1)
        heart_rate = st.slider("Resting heart rate (bpm)", 40, 150, 72)
        bmi = st.slider("BMI", 10.0, 45.0, 22.0, 0.1)
        calorie_expenditure = st.number_input("Calorie expenditure (kcal/day)", 800, 6000, 2200)
        step_count = st.number_input("Step count (per day)", 0, 30000, 6000)
        exercise_duration = st.slider("Exercise duration (minutes/day)", 0, 180, 30)
        water_intake = st.slider("Water intake (litres/day)", 0.0, 6.0, 2.0, 0.1)

    with col2:
        diet_type = st.selectbox(
            "Diet type",
            ["balanced", "veg", "non-veg"],
        )
        stress_level = st.selectbox("Stress level", ["low", "medium", "high"])
        sleep_quality = st.selectbox("Sleep quality", ["poor", "average", "good"])
        physical_activity_level = st.selectbox(
            "Physical activity level", ["sedentary", "moderate", "active"]
        )
        smoking_alcohol = st.selectbox(
            "Smoking / alcohol use", ["no", "occasional", "yes"]
        )
        gender = st.selectbox("Gender", ["male", "female", "other"])

    submitted = st.form_submit_button("Predict health risk")

if submitted:
    input_row = pd.DataFrame([{
        "sleep_duration": sleep_duration,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "calorie_expenditure": calorie_expenditure,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "physical_activity_level": physical_activity_level,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    }])[FEATURE_ORDER]

    X_enc = transform(input_row, preprocessor)
    predicted_class = model.predict(X_enc)[0]
    probabilities = model.predict_proba(X_enc)[0]

    class_order = list(model.classes_)
    proba_df = pd.DataFrame({
        "health_condition": class_order,
        "probability": probabilities,
    }).sort_values("probability", ascending=True)

    st.subheader("Prediction")
    label_colors = {"at-risk": "🔴", "unhealthy": "🟠", "fit": "🟢"}
    st.markdown(
        f"### {label_colors.get(predicted_class, '')} Predicted status: **{predicted_class.upper()}**"
    )

    st.bar_chart(proba_df.set_index("health_condition"), horizontal=True)

    st.subheader("Probability breakdown")
    for _, row in proba_df.sort_values("probability", ascending=False).iterrows():
        st.write(f"**{row['health_condition']}**: {row['probability'] * 100:.2f}%")

st.divider()
st.caption(
    "This tool is trained on a synthetic Kaggle competition dataset and is "
    "for educational/demonstration purposes only — it is not a medical "
    "diagnostic tool."
)
