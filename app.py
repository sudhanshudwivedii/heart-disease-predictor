import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# Load Model, Scaler, Columns, Mappings
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("RandomForest_heaart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    mappings = joblib.load("mappings.pkl")
    return model, scaler, columns, mappings

try:
    model, scaler, columns, mappings = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}. "
        "Please make sure RandomForest_heaart.pkl, scaler.pkl, columns.pkl "
        "and mappings.pkl are in the same folder as app.py."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Category Mappings (readable label -> encoded value)
# ----------------------------------------------------------------------------
SEX_MAP = {"Female": 0, "Male": 1}
CHEST_PAIN_MAP = {
    "Asymptomatic (ASY)": 0,
    "Atypical Angina (ATA)": 1,
    "Non-Anginal Pain (NAP)": 2,
    "Typical Angina (TA)": 3,
}
RESTING_ECG_MAP = {
    "Left Ventricular Hypertrophy (LVH)": 0,
    "Normal": 1,
    "ST-T Wave Abnormality (ST)": 2,
}
EXERCISE_ANGINA_MAP = {"No": 0, "Yes": 1}
ST_SLOPE_MAP = {"Downsloping": 0, "Flat": 1, "Upsloping": 2}

# ----------------------------------------------------------------------------
# Custom CSS for a polished look
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #d7263d;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="main-title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine Learning based tool to estimate the risk of heart disease '
    'using patient clinical parameters. This is for educational purposes only and is '
    '<b>not</b> a substitute for professional medical advice.</div>',
    unsafe_allow_html=True
)
st.divider()

# ----------------------------------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------------------------------
st.sidebar.header("🩺 Patient Details")
st.sidebar.markdown("Enter the patient's clinical information below:")

st.sidebar.subheader("Numeric Parameters")
age = st.sidebar.slider("Age (years)", min_value=28, max_value=77, value=50, step=1)
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120, step=1)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=150, step=1)
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)

st.sidebar.subheader("Categorical Parameters")
fasting_bs_label = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
fasting_bs = 1 if fasting_bs_label == "Yes" else 0

sex_label = st.sidebar.selectbox("Sex", list(SEX_MAP.keys()))
chest_pain_label = st.sidebar.selectbox("Chest Pain Type", list(CHEST_PAIN_MAP.keys()))
resting_ecg_label = st.sidebar.selectbox("Resting ECG", list(RESTING_ECG_MAP.keys()))
exercise_angina_label = st.sidebar.selectbox("Exercise Induced Angina", list(EXERCISE_ANGINA_MAP.keys()))
st_slope_label = st.sidebar.selectbox("ST Slope", list(ST_SLOPE_MAP.keys()))

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True)

# ----------------------------------------------------------------------------
# Main Panel
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Prediction Logic
# ----------------------------------------------------------------------------
if predict_btn:
    # Map readable labels to encoded numeric values
    sex_val = SEX_MAP[sex_label]
    chest_pain_val = CHEST_PAIN_MAP[chest_pain_label]
    resting_ecg_val = RESTING_ECG_MAP[resting_ecg_label]
    exercise_angina_val = EXERCISE_ANGINA_MAP[exercise_angina_label]
    st_slope_val = ST_SLOPE_MAP[st_slope_label]

    # Build input dict matching required column order
    input_dict = {
        "Age": age,
        "Sex": sex_val,
        "ChestPainType": chest_pain_val,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg_val,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina_val,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope_val,
    }

    # Create DataFrame in exact column order expected by the model
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[columns]

    # Scale all columns
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    st.divider()
    st.subheader("📋 Prediction Result")

    prob_disease = probability[1] * 100
    prob_no_disease = probability[0] * 100

    if prediction == 1:
        st.markdown(
            f'<div class="result-box" style="background-color:#f8d7da; color:#842029; border:1px solid #f5c2c7;">'
            f'⚠️ High Risk: The patient is likely to have Heart Disease<br>'
            f'Probability: {prob_disease:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box" style="background-color:#d1e7dd; color:#0f5132; border:1px solid #badbcc;">'
            f'✅ Low Risk: The patient is unlikely to have Heart Disease<br>'
            f'Probability: {prob_no_disease:.2f}%</div>',
            unsafe_allow_html=True
        )

    st.write("")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.metric("Probability of No Disease", f"{prob_no_disease:.2f}%")
    with pcol2:
        st.metric("Probability of Disease", f"{prob_disease:.2f}%")

    st.progress(int(prob_disease))

    with st.expander("🔎 View Processed Input Data"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict** to see the result.")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit & RandomForestClassifier | "
    "⚠️ For educational/demo purposes only — always consult a certified doctor for medical decisions."
)