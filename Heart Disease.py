import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Heart Health Predictor", page_icon="❤️", layout="centered")

# --- Professional Styling ---
st.markdown("""
    <style>
    /* Center the subheader */
    .css-10tr3a {text-align: center;}
    /* Style the button to look professional */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
    /* Increase spacing in the form */
    .stForm {
        padding: 20px;
        border: 1px solid #E6E6E6;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    with open('heartdisease_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('heartdisease_scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('heartdisease_columns.pkl', 'rb') as f: model_columns = pickle.load(f)
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# --- Main Interface ---
st.title("Heart Disease Predictor")
st.write("Professional screening tool for assessing cardiovascular risk factors.")

with st.form("patient_form"):
    st.markdown("### Patient Information")
    
    # Using columns for better horizontal balance
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Age", min_value=1, max_value=100, value=50)
        sex = st.selectbox("Sex", [("Male", 0), ("Female", 1)], format_func=lambda x: x[0])
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        maxhr = st.number_input("Max Heart Rate (MaxHR)", min_value=60, max_value=220, value=140)
        
    with c2:
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        chol = st.number_input("Cholesterol", min_value=0, value=300)
        exang = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        st_slope = st.selectbox("ST Slope", [("Up", 0), ("Flat", 1), ("Down", 2)], format_func=lambda x: x[0])

    cp_type = st.selectbox("Chest Pain Type", [("ASY", 0), ("ATA", 1), ("NAP", 2), ("TA", 3)], format_func=lambda x: x[0])
    
    submit = st.form_submit_button("Run Prediction")

# --- Prediction Logic ---
if submit:
    # Extracting values from tuples
    data = {
        'Age': [age], 'is_Female': [sex[1]], 'Cholesterol': [chol],
        'FastingBS': [fasting_bs[1]], 'ST_Slope': [st_slope[1]], 'ExerciseAngina': [exang[1]],
        'Oldpeak': [oldpeak], 'MaxHR': [maxhr],
        'is_ChestPainType_ASY': [1 if cp_type[1] == 0 else 0],
        'is_ChestPainType_ATA': [1 if cp_type[1] == 1 else 0],
        'is_ChestPainType_NAP': [1 if cp_type[1] == 2 else 0],
        'is_ChestPainType_TA': [1 if cp_type[1] == 3 else 0]
    }
    
    input_df = pd.DataFrame(data).reindex(columns=model_columns, fill_value=0)
    input_df[['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(input_df[['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']])
    
    if model.predict(input_df)[0] == 1:
        st.error("Result: Heart disease. Please consult a cardiologist.")
    else:
        st.success("Result: No heart disease.")
