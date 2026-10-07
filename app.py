import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# -----------------------------
# Load dataset and train model
# -----------------------------
@st.cache_resource
def load_model():
    heart_data = pd.read_csv("data.csv")

    X = heart_data.drop(columns="target")
    y = heart_data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X.values, y.values)

    return model


model = load_model()


# -----------------------------
# Page title
# -----------------------------
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's clinical information below to predict "
    "the likelihood of heart disease."
)

st.divider()


# -----------------------------
# Input fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=50,
        max_value=700,
        value=200,
        step=1
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

with col2:
    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        options=[0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[0, 1, 2, 3]
    )


# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    input_data = (
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    )

    input_array = np.asarray(input_data).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Heart Disease Detected")
    else:
        st.success("✅ Prediction: No Heart Disease Detected")

    st.caption(
        "This prediction is for educational purposes only and "
        "is not a medical diagnosis."
    )
