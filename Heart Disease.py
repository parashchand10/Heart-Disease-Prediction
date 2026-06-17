import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main Background */
.stApp {
    background-color: #f4f7fb;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

/* Form Card */
.form-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
}

/* Prediction Cards */
.success-card {
    background: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.danger-card {
    background: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    background: linear-gradient(135deg,#2563eb,#0ea5e9);
    color: white;
    border-radius: 10px;
    border: none;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(135deg,#1d4ed8,#0284c7);
    color: white;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.08);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_assets():
    with open("heartdisease_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("heartdisease_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("heartdisease_columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, scaler, columns

model, scaler, model_columns = load_assets()

# =========================
# HEADER
# =========================
st.markdown("""
<div class='hero'>
    <h1>Heart Health Predictor</h1>
</div>
""", unsafe_allow_html=True)

# =========================
# FORM
# =========================
st.markdown("<div class='form-card'>", unsafe_allow_html=True)

st.subheader("Patient Information")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=50
        )

        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar >120",
            ["No", "Yes"]
        )

    with col2:
        chol = st.number_input(
            "Cholesterol",
            min_value=0,
            value=250
        )

        maxhr = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )

    with col3:
        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

        cp_type = st.selectbox(
            "Chest Pain Type",
            ["ASY", "ATA", "NAP", "TA"]
        )

    submitted = st.form_submit_button("🔍 Predict Risk")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if submitted:

    data = {
        'Age': [age],
        'is_Female': [1 if sex == "Female" else 0],
        'Cholesterol': [chol],
        'FastingBS': [1 if fasting_bs == "Yes" else 0],
        'ST_Slope': [{"Up":0,"Flat":1,"Down":2}[st_slope]],
        'ExerciseAngina': [1 if exang == "Yes" else 0],
        'Oldpeak': [oldpeak],
        'MaxHR': [maxhr],

        'is_ChestPainType_ASY': [1 if cp_type == "ASY" else 0],
        'is_ChestPainType_ATA': [1 if cp_type == "ATA" else 0],
        'is_ChestPainType_NAP': [1 if cp_type == "NAP" else 0],
        'is_ChestPainType_TA': [1 if cp_type == "TA" else 0]
    }

    input_df = pd.DataFrame(data)

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    input_df[
        ['Age','Cholesterol','MaxHR','Oldpeak']
    ] = scaler.transform(
        input_df[
            ['Age','Cholesterol','MaxHR','Oldpeak']
        ]
    )

    prediction = model.predict(input_df)[0]

    try:
        probability = model.predict_proba(input_df)[0][1]
    except:
        probability = 0.5

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:

        if prediction == 1:
            st.markdown(f"""
            <div class='danger-card'>
                <h3>⚠ High Risk Detected</h3>
                <p>
                The model predicts a possibility of heart disease.
                Please consult a cardiologist for further evaluation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='success-card'>
                <h3>✅ Low Risk Detected</h3>
                <p>
                No significant signs of heart disease were detected.
                Maintain a healthy lifestyle and regular checkups.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class='metric-card'>
            <h4>Risk Score</h4>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(probability * 100))

        st.metric(
            "Heart Disease Risk",
            f"{probability*100:.1f}%"
        )

        if probability < 0.30:
            st.success("Low Risk")
        elif probability < 0.70:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")
            
