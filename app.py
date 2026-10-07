import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


@st.cache_resource
def load_model():
    heart_data = pd.read_csv("data.csv")

    X = heart_data.drop(columns="target")
    y = heart_data["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X.values, y.values)

    return model


model = load_model()


st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's clinical information below to "
    "predict the presence of heart disease."
)

st.divider()


col1, col2 = st.columns(2)


with col1:

    age_input = st.text_input(
        "Age",
        value="50"
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

    trestbps_input = st.text_input(
        "Resting Blood Pressure",
        value="120"
    )

    chol_input = st.text_input(
        "Serum Cholesterol",
        value="200"
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

    thalach_input = st.text_input(
        "Maximum Heart Rate Achieved",
        value="150"
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )

    oldpeak_input = st.text_input(
        "ST Depression (oldpeak)",
        value="1.0"
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


st.divider()


if st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
):

    invalid_inputs = []

    # -----------------------------
    # Convert numeric inputs
    # -----------------------------

    try:
        age = float(age_input)
    except ValueError:
        invalid_inputs.append("Age must be a valid number.")
        age = None

    try:
        trestbps = float(trestbps_input)
    except ValueError:
        invalid_inputs.append(
            "Resting Blood Pressure must be a valid number."
        )
        trestbps = None

    try:
        chol = float(chol_input)
    except ValueError:
        invalid_inputs.append(
            "Cholesterol must be a valid number."
        )
        chol = None

    try:
        thalach = float(thalach_input)
    except ValueError:
        invalid_inputs.append(
            "Maximum Heart Rate must be a valid number."
        )
        thalach = None

    try:
        oldpeak = float(oldpeak_input)
    except ValueError:
        invalid_inputs.append(
            "ST Depression must be a valid number."
        )
        oldpeak = None


    # -----------------------------
    # Range validation
    # -----------------------------

    if age is not None and not (1 <= age <= 120):
        invalid_inputs.append(
            "Age must be between 1 and 120."
        )

    if trestbps is not None and not (50 <= trestbps <= 250):
        invalid_inputs.append(
            "Resting Blood Pressure must be between 50 and 250."
        )

    if chol is not None and not (50 <= chol <= 700):
        invalid_inputs.append(
            "Cholesterol must be between 50 and 700."
        )

    if thalach is not None and not (50 <= thalach <= 250):
        invalid_inputs.append(
            "Maximum Heart Rate must be between 50 and 250."
        )

    if oldpeak is not None and not (0 <= oldpeak <= 10):
        invalid_inputs.append(
            "ST Depression must be between 0 and 10."
        )


    # -----------------------------
    # STOP prediction if invalid
    # -----------------------------

    if invalid_inputs:

        st.error(
            "⚠️ Prediction cannot be performed."
        )

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

        input_array = np.asarray(
            input_data
        ).reshape(1, -1)

        prediction = model.predict(input_array)[0]

        if prediction == 1:
            st.error(
                "⚠️ Prediction: Heart Disease Detected"
            )
        else:
            st.success(
                "✅ Prediction: No Heart Disease Detected"
            )

        st.caption(
            "This prediction is for educational purposes only "
            "and is not a medical diagnosis."
        )
