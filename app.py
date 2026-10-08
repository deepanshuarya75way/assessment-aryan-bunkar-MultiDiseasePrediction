import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from pymongo import MongoClient 

from guidance.guidance_engine import GuidanceEngine


# =========================================================
# Page configuration
# =========================================================
client = MongoClient("mongodb://localhost:27017/")
db = client("MultidiseasePrediction")
feedback_collection = db("feedback")


st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# Load Heart Disease model
# =========================================================
@st.cache_resource
def load_heart_model():
    return joblib.load("models/heart_disease_model.pkl")


@st.cache_resource
def load_liver_model():
    return joblib.load("models/liver_disease_model.pkl")


@st.cache_resource
def load_kidney_model():
    return joblib.load("models/kidney_disease_model.pkl")


@st.cache_resource
def load_thyroid_model():
    return joblib.load("models/thyroid_disease_model.pkl")

@st.cache_resource
def load_lung_model():
    return tf.keras.models.load_model(
        "models/lung_disease_densenet121_repaired.keras",
        compile=False
    )


heart_model = load_heart_model()
liver_model = load_liver_model()
kidney_model = load_kidney_model()
thyroid_model = load_thyroid_model()
lung_model = load_lung_model()
# =========================================================
# Load Guidance Engine
# =========================================================

@st.cache_resource
def load_guidance_engine():
    return GuidanceEngine()


guidance_engine = load_guidance_engine()


# =========================================================
# Title
# =========================================================

st.title("🏥 Multi Disease Prediction")

st.write(
    "Machine-learning based disease screening with "
    "educational guidance."
)


# =========================================================
# Disease selection
# =========================================================

disease = st.selectbox(
    "Select Disease",
    [
        "Heart Disease",
        "Liver Disease",
        "Kidney Disease",
        "Thyroid Disease",
        "Lung Disease"
    ]
)

# =========================================================
# Heart Disease Inputs
# =========================================================

if disease == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=63
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0
                else "Male (1)"
        )

        cp = st.number_input(
            "Chest Pain Type (cp)",
            min_value=1,
            max_value=4,
            value=1
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=145
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=233
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0
                else "Yes (1)"
        )

        restecg = st.number_input(
            "Resting ECG (restecg)",
            min_value=0,
            max_value=2,
            value=2
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate (thalach)",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0
                else "Yes (1)"
        )

        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=2.3,
            step=0.1
        )

        slope = st.number_input(
            "Slope",
            min_value=1,
            max_value=3,
            value=3
        )

        ca = st.number_input(
            "Number of Major Vessels (ca)",
            min_value=0,
            max_value=4,
            value=0
        )

        thal = st.number_input(
            "Thal",
            min_value=0,
            max_value=7,
            value=6
        )


    # =====================================================
    # Create patient dataframe
    # =====================================================

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])


    # =====================================================
    # Prediction button
    # =====================================================

    if st.button(
        "🔍 Predict Heart Disease",
        type="primary"
    ):

        # Make prediction
        prediction = heart_model.predict(
            patient
        )[0]

        probabilities = heart_model.predict_proba(
            patient
        )[0]


        # Probability
        no_disease_probability = probabilities[0]

        disease_probability = probabilities[1]


        # Prediction text
        if prediction == 1:
            prediction_text = "Heart Disease"
        else:
            prediction_text = "No Heart Disease"


        # =================================================
        # Get Guidance
        # =================================================

        guidance = guidance_engine.get_guidance(
            disease_id=1,
            probability=disease_probability,
            inputs=patient
        )


        # =================================================
        # Display prediction
        # =================================================

        st.divider()

        st.subheader("Prediction Result")

        st.write(
            f"### {prediction_text}"
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "No Disease Probability",
                f"{no_disease_probability:.2%}"
            )

        with col2:

            st.metric(
                "Disease Probability",
                f"{disease_probability:.2%}"
            )


        # =================================================
        # Probability level
        # =================================================

        st.subheader(
            "Probability Level"
        )

        st.info(
            guidance["probability_level"]
        )


        # =================================================
        # Consume
        # =================================================

        st.subheader(
            "🥗 What to Consume"
        )

        for item in guidance["consume"]:

            st.write(
                f"• {item}"
            )


        # =================================================
        # Limit
        # =================================================

        st.subheader(
            "⚠️ What to Limit"
        )

        for item in guidance["limit"]:

            st.write(
                f"• {item}"
            )


        # =================================================
        # Lifestyle
        # =================================================

        st.subheader(
            "🏃 Lifestyle"
        )

        for item in guidance["lifestyle"]:

            st.write(
                f"• {item}"
            )


        # =================================================
        # Input-specific observations
        # =================================================

        if guidance["input_flags"]:

            st.subheader(
                "📋 Input-Specific Observations"
            )

            for item in guidance["input_flags"]:

                st.warning(
                    item["message"]
                )


        # =================================================
        # When to seek care
        # =================================================

        st.subheader(
            "🩺 When to Seek Care"
        )

        for item in guidance["seek_care"]:

            st.write(
                f"• {item}"
            )


        # =================================================
        # Sources
        # =================================================

        st.subheader(
            "📚 Sources"
        )

        for source in guidance["sources"]:

            st.write(
                f"• {source['organization']} - "
                f"{source['title']}"
            )

# =========================================================
# Liver Disease Inputs
# =========================================================

elif disease == "Liver Disease":

    st.header("🧪 Liver Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0
                else "Male (1)"
        )

        total_bilirubin = st.number_input(
            "Total Bilirubin",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            min_value=0.0,
            value=0.3,
            step=0.1
        )

        alkphos = st.number_input(
            "Alkaline Phosphatase (alkphos)",
            min_value=0.0,
            value=200.0,
            step=1.0
        )

    with col2:

        sgpt = st.number_input(
            "SGPT",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        sgot = st.number_input(
            "SGOT",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        total_proteins = st.number_input(
            "Total Proteins",
            min_value=0.0,
            value=6.5,
            step=0.1
        )

        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            value=3.5,
            step=0.1
        )

        ag_ratio = st.number_input(
            "A/G Ratio",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    # =====================================================
    # Create Liver patient dataframe
    # =====================================================

    liver_patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "total_bilirubin": total_bilirubin,
        "direct_bilirubin": direct_bilirubin,
        "alkphos": alkphos,
        "sgpt": sgpt,
        "sgot": sgot,
        "total_proteins": total_proteins,
        "albumin": albumin,
        "ag_ratio": ag_ratio
    }])

    # =====================================================
    # Prediction button
    # =====================================================

    if st.button(
        "🔍 Predict Liver Disease",
        type="primary"
    ):

        # Make prediction
        prediction = liver_model.predict(
            liver_patient
        )[0]

        probabilities = liver_model.predict_proba(
            liver_patient
        )[0]

        # Probability
        no_disease_probability = probabilities[0]

        disease_probability = probabilities[1]

        # Prediction text
        if prediction == 1:
            prediction_text = "Liver Disease"
        else:
            prediction_text = "No Liver Disease"

        # =================================================
        # Get Guidance
        # =================================================

        guidance = guidance_engine.get_guidance(
            disease_id=2,
            probability=disease_probability,
            inputs=liver_patient
        )

        # =================================================
        # Display prediction
        # =================================================

        st.divider()

        st.subheader("Prediction Result")

        st.write(
            f"### {prediction_text}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "No Disease Probability",
                f"{no_disease_probability:.2%}"
            )

        with col2:

            st.metric(
                "Disease Probability",
                f"{disease_probability:.2%}"
            )

        # =================================================
        # Probability level
        # =================================================

        st.subheader(
            "Probability Level"
        )

        st.info(
            guidance["probability_level"]
        )

        # =================================================
        # Consume
        # =================================================

        st.subheader(
            "🥗 What to Consume"
        )

        for item in guidance["consume"]:

            st.write(
                f"• {item}"
            )

        # =================================================
        # Limit
        # =================================================

        st.subheader(
            "⚠️ What to Limit"
        )

        for item in guidance["limit"]:

            st.write(
                f"• {item}"
            )

        # =================================================
        # Lifestyle
        # =================================================

        st.subheader(
            "🏃 Lifestyle"
        )

        for item in guidance["lifestyle"]:

            st.write(
                f"• {item}"
            )

        # =================================================
        # Input-specific observations
        # =================================================

        if guidance["input_flags"]:

            st.subheader(
                "📋 Input-Specific Observations"
            )

            for item in guidance["input_flags"]:

                st.warning(
                    item["message"]
                )

        # =================================================
        # When to seek care
        # =================================================

        st.subheader(
            "🩺 When to Seek Care"
        )

        for item in guidance["seek_care"]:

            st.write(
                f"• {item}"
            )

        # =================================================
        # Sources
        # =================================================

        st.subheader(
            "📚 Sources"
        )

        for source in guidance["sources"]:

            st.write(
                f"• {source['organization']} - "
                f"{source['title']}"
            )

elif disease == "Kidney Disease":

    st.header("🫘 Kidney Disease (CKD) Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"]
        )

        ethnicity = st.selectbox(
            "Ethnicity",
            options=["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Other"]
        )

        education_level = st.number_input(
            "Education Level",
            min_value=0.0,
            value=12.0,
            step=1.0
        )

        poverty_income_ratio = st.number_input(
            "Poverty Income Ratio",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0,
            step=0.1
        )

        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.0,
            value=70.0,
            step=0.1
        )

        height_cm = st.number_input(
            "Height (cm)",
            min_value=0.0,
            value=170.0,
            step=0.1
        )

        bp_systolic = st.number_input(
            "Systolic Blood Pressure",
            min_value=0.0,
            value=120.0,
            step=1.0
        )

        bp_diastolic = st.number_input(
            "Diastolic Blood Pressure",
            min_value=0.0,
            value=80.0,
            step=1.0
        )

        serum_creatinine = st.number_input(
            "Serum Creatinine",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        blood_urea_nitrogen = st.number_input(
            "Blood Urea Nitrogen",
            min_value=0.0,
            value=15.0,
            step=1.0
        )

    with col2:

        albumin_serum = st.number_input(
            "Serum Albumin",
            min_value=0.0,
            value=4.0,
            step=0.1
        )

        phosphorus = st.number_input(
            "Phosphorus",
            min_value=0.0,
            value=3.5,
            step=0.1
        )

        bicarbonate = st.number_input(
            "Bicarbonate",
            min_value=0.0,
            value=24.0,
            step=0.1
        )

        calcium = st.number_input(
            "Calcium",
            min_value=0.0,
            value=9.5,
            step=0.1
        )

        uric_acid = st.number_input(
            "Uric Acid",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        urine_creatinine = st.number_input(
            "Urine Creatinine",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        urine_albumin = st.number_input(
            "Urine Albumin",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        albumin_creatinine_ratio = st.number_input(
            "Albumin Creatinine Ratio",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        diabetes_diagnosed = st.selectbox(
            "Diabetes Diagnosed",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

        insulin_use = st.selectbox(
            "Insulin Use",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

        diabetes_pills = st.selectbox(
            "Diabetes Pills",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

        ever_smoked = st.selectbox(
            "Ever Smoked",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

        current_smoker = st.selectbox(
            "Current Smoker",
            options=[0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    kidney_patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "poverty_income_ratio": poverty_income_ratio,
        "bmi": bmi,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "serum_creatinine": serum_creatinine,
        "blood_urea_nitrogen": blood_urea_nitrogen,
        "albumin_serum": albumin_serum,
        "phosphorus": phosphorus,
        "bicarbonate": bicarbonate,
        "calcium": calcium,
        "uric_acid": uric_acid,
        "urine_creatinine": urine_creatinine,
        "urine_albumin": urine_albumin,
        "albumin_creatinine_ratio": albumin_creatinine_ratio,
        "diabetes_diagnosed": diabetes_diagnosed,
        "insulin_use": insulin_use,
        "diabetes_pills": diabetes_pills,
        "ever_smoked": ever_smoked,
        "current_smoker": current_smoker
    }])

    if st.button("🔍 Predict Kidney Disease", type="primary"):

        prediction = kidney_model.predict(kidney_patient)[0]

        probabilities = kidney_model.predict_proba(kidney_patient)[0]

        no_disease_probability = probabilities[0]
        disease_probability = probabilities[1]

        prediction_text = "CKD" if prediction == 1 else "No CKD"

        guidance = guidance_engine.get_guidance(
            disease_id=3,
            probability=disease_probability,
            inputs=kidney_patient
        )

        st.divider()

        st.subheader("Prediction Result")

        st.write(f"### {prediction_text}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "No CKD Probability",
                f"{no_disease_probability:.2%}"
            )

        with col2:
            st.metric(
                "CKD Probability",
                f"{disease_probability:.2%}"
            )

        st.subheader("Probability Level")

        st.info(guidance["probability_level"])

        st.subheader("🥗 What to Consume")

        for item in guidance["consume"]:
            st.write(f"• {item}")

        st.subheader("⚠️ What to Limit")

        for item in guidance["limit"]:
            st.write(f"• {item}")

        st.subheader("🏃 Lifestyle")

        for item in guidance["lifestyle"]:
            st.write(f"• {item}")

        if guidance["input_flags"]:

            st.subheader("📋 Input-Specific Observations")

            for item in guidance["input_flags"]:
                st.warning(item["message"])

        st.subheader("🩺 When to Seek Care")

        for item in guidance["seek_care"]:
            st.write(f"• {item}")

        st.subheader("📚 Sources")

        for source in guidance["sources"]:
            st.write(
                f"• {source['organization']} - {source['title']}"
            )
elif disease == "Thyroid Disease":

    st.header("🦋 Thyroid Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0 else "Male (1)"
        )

        on_thyroxine = st.selectbox(
            "Currently on Thyroxine",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        query_on_thyroxine = st.selectbox(
            "Query on Thyroxine",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        on_antithyroid_medication = st.selectbox(
            "On Antithyroid Medication",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        thyroid_surgery = st.selectbox(
            "Thyroid Surgery",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        query_hypothyroid = st.selectbox(
            "Query Hypothyroid",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        query_hyperthyroid = st.selectbox(
            "Query Hyperthyroid",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        pregnant = st.selectbox(
            "Pregnant",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        sick = st.selectbox(
            "Currently Sick",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        tumor = st.selectbox(
            "Tumor",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        lithium = st.selectbox(
            "Lithium Use",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

    with col2:

        goitre = st.selectbox(
            "Goitre",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        TSH_measured = st.selectbox(
            "TSH Measured",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        TSH = st.number_input(
            "TSH",
            min_value=0.0,
            value=2.0,
            step=0.01
        )

        T3_measured = st.selectbox(
            "T3 Measured",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        T3 = st.number_input(
            "T3",
            min_value=0.0,
            value=1.5,
            step=0.01
        )

        TT4_measured = st.selectbox(
            "TT4 Measured",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        TT4 = st.number_input(
            "TT4",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        T4U_measured = st.selectbox(
            "T4U Measured",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        T4U = st.number_input(
            "T4U",
            min_value=0.0,
            value=1.0,
            step=0.01
        )

        FTI_measured = st.selectbox(
            "FTI Measured",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        FTI = st.number_input(
            "FTI",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    thyroid_patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "on_thyroxine": on_thyroxine,
        "query_on_thyroxine": query_on_thyroxine,
        "on_antithyroid_medication": on_antithyroid_medication,
        "thyroid_surgery": thyroid_surgery,
        "query_hypothyroid": query_hypothyroid,
        "query_hyperthyroid": query_hyperthyroid,
        "pregnant": pregnant,
        "sick": sick,
        "tumor": tumor,
        "lithium": lithium,
        "goitre": goitre,
        "TSH_measured": TSH_measured,
        "TSH": TSH,
        "T3_measured": T3_measured,
        "T3": T3,
        "TT4_measured": TT4_measured,
        "TT4": TT4,
        "T4U_measured": T4U_measured,
        "T4U": T4U,
        "FTI_measured": FTI_measured,
        "FTI": FTI
    }])

    if st.button("🔍 Predict Thyroid Disease", type="primary"):

        prediction = thyroid_model.predict(thyroid_patient)[0]

        probabilities = thyroid_model.predict_proba(thyroid_patient)[0]

        classes = thyroid_model.classes_

        hypothyroid_probability = probabilities[
            list(classes).index("hypothyroid")
        ]

        negative_probability = probabilities[
            list(classes).index("negative")
        ]

        prediction_text = (
            "Hypothyroid"
            if prediction == "hypothyroid"
            else "No Hypothyroid"
        )

        guidance = guidance_engine.get_guidance(
            disease_id=4,
            probability=hypothyroid_probability,
            inputs=thyroid_patient
        )

        st.divider()

        st.subheader("Prediction Result")

        st.write(f"### {prediction_text}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "No Hypothyroid Probability",
                f"{negative_probability:.2%}"
            )

        with col2:
            st.metric(
                "Hypothyroid Probability",
                f"{hypothyroid_probability:.2%}"
            )

        st.subheader("Probability Level")

        st.info(guidance["probability_level"])

        st.subheader("🥗 What to Consume")

        for item in guidance["consume"]:
            st.write(f"• {item}")

        st.subheader("⚠️ What to Limit")

        for item in guidance["limit"]:
            st.write(f"• {item}")

        st.subheader("🏃 Lifestyle")

        for item in guidance["lifestyle"]:
            st.write(f"• {item}")

        if guidance["input_flags"]:

            st.subheader("📋 Input-Specific Observations")

            for item in guidance["input_flags"]:
                st.warning(item["message"])

        st.subheader("🩺 When to Seek Care")

        for item in guidance["seek_care"]:
            st.write(f"• {item}")

        st.subheader("📚 Sources")

        for source in guidance["sources"]:
            st.write(
                f"• {source['organization']} - {source['title']}"
            )
elif disease == "Lung Disease":

    st.header("🫁 Lung Disease / Pneumonia Prediction")

    st.write(
        "Upload a chest X-ray image to screen for pneumonia."
    )

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

        if st.button("🔍 Predict Lung Disease", type="primary"):

            # Resize image
            image_resized = image.resize((224, 224))

            # Convert to NumPy
            image_array = np.array(image_resized)

            # Normalize
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Prediction
            prediction_probability = lung_model.predict(
                image_array,
                verbose=0
            )[0][0]

            pneumonia_probability = float(prediction_probability)
            normal_probability = 1.0 - pneumonia_probability

            # Prediction
            if pneumonia_probability >= 0.5:
                prediction_text = "Pneumonia"
            else:
                prediction_text = "Normal"

            st.success(f"Prediction: {prediction_text}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Normal Probability",
                    f"{normal_probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Pneumonia Probability",
                    f"{pneumonia_probability * 100:.2f}%"
                )

            # Guidance
            guidance = guidance_engine.get_guidance(
                disease_id=5,
                probability=pneumonia_probability,
                inputs=None
            )

            st.subheader("📋 Guidance")

            if guidance.get("consume"):
                st.markdown("**🍎 Consume / Do:**")
                for item in guidance["consume"]:
                    st.write(f"• {item}")

            if guidance.get("avoid"):
                st.markdown("**⚠️ Limit / Avoid:**")
                for item in guidance["avoid"]:
                    st.write(f"• {item}")

            if guidance.get("lifestyle"):
                st.markdown("**🏃 Lifestyle:**")
                for item in guidance["lifestyle"]:
                    st.write(f"• {item}")

            if guidance.get("seek_care"):
                st.markdown("**🩺 Seek Care:**")
                for item in guidance["seek_care"]:
                    st.write(f"• {item}")