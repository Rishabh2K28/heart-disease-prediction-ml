import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression


# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# -----------------------------------
# Load dataset and train model
# -----------------------------------
@st.cache_resource
def load_model():
    heart_data = pd.read_csv("data.csv")

    X = heart_data.drop(columns="target")
    y = heart_data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X.values, y.values)

    return model


model = load_model()


# -----------------------------------
# Title
# -----------------------------------
st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's clinical information below to "
    "predict the presence of heart disease."
)

st.divider()


# -----------------------------------
# Input fields
# -----------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=150,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    chol = st.number_input(
        "Serum Cholesterol",
        min_value=0,
        max_value=1000,
        value=200,
        step=1
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )


with col2:

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=0,
        max_value=300,
        value=150,
        step=1
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia (thal)",
        [0, 1, 2, 3]
    )


# -----------------------------------
# Prediction
# -----------------------------------
st.divider()

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    invalid_inputs = []

    if age < 1 or age > 120:
        invalid_inputs.append(
            "Age must be between 1 and 120."
        )

    if trestbps < 50 or trestbps > 250:
        invalid_inputs.append(
            "Resting Blood Pressure must be between 50 and 250."
        )

    if chol < 50 or chol > 700:
        invalid_inputs.append(
            "Cholesterol must be between 50 and 700."
        )

    if thalach < 50 or thalach > 250:
        invalid_inputs.append(
            "Maximum Heart Rate must be between 50 and 250."
        )

    if oldpeak < 0 or oldpeak > 10:
        invalid_inputs.append(
            "ST Depression must be between 0 and 10."
        )

    # STOP HERE if anything is invalid
    if invalid_inputs:

        st.error("⚠️ Prediction cannot be performed.")

        st.warning(
            "Please enter values within the specified "
            "ranges to run the model."
        )

        for message in invalid_inputs:
            st.write(f"• {message}")

    else:

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
            "This prediction is for educational purposes only "
            "and is not a medical diagnosis."
        )
