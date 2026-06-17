import streamlit as st
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Heart Health Predictor", page_icon="❤️", layout="centered")

# --- Custom Styling for Button ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Components ---
@st.cache_resource
def load_assets():
    with open('heartdisease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('heartdisease_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('heartdisease_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# --- Layout Wrapper ---
# Creating columns to constrain width for a cleaner, centered look
_, col_main, _ = st.columns([1, 6, 1])

with col_main:
    st.title("Heart Disease Predictor")
    st.markdown("This tool uses machine learning to assess potential heart disease risks.")

    with st.form("patient_form"):
        # Centering the subheader inside the form
        st.markdown("<h3 style='text-align: center;'>Patient Information</h3>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=100, value=50)
            is_female = st.selectbox("Sex", [("Male", 0), ("Female", 1)], format_func=lambda x: x[0])[1]
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            maxhr = st.number_input("Max Heart Rate (MaxHR)", min_value=60, max_value=220, value=140)
            
        with c2:
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            chol = st.number_input("Cholesterol", min_value=0, value=300)
            exang = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            st_slope = st.selectbox("ST Slope", [("Up", 0), ("Flat", 1), ("Down", 2)], format_func=lambda x: x[0])[1]
            cp_type = st.selectbox("Chest Pain Type", [("ASY", 0), ("ATA", 1), ("NAP", 2), ("TA", 3)], format_func=lambda x: x[0])[1]

        # Use an empty column trick to center the button if desired, 
        # or keep it full width for better mobile interaction.
        submit = st.form_submit_button("Analyze Patient Data", use_container_width=True)

    # --- Prediction Logic ---
    if submit:
        with st.spinner("Analyzing data..."):
            data = {
                'Age': [age], 'is_Female': [is_female], 'Cholesterol': [chol],
                'FastingBS': [fasting_bs], 'ST_Slope': [st_slope], 'ExerciseAngina': [exang],
                'Oldpeak': [oldpeak], 'MaxHR': [maxhr],
                'is_ChestPainType_ASY': [1 if cp_type == 0 else 0],
                'is_ChestPainType_ATA': [1 if cp_type == 1 else 0],
                'is_ChestPainType_NAP': [1 if cp_type == 2 else 0],
                'is_ChestPainType_TA': [1 if cp_type == 3 else 0]
            }
            input_df = pd.DataFrame(data)
            input_df = input_df.reindex(columns=model_columns, fill_value=0)
            input_df[['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.transform(input_df[['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']])
            
            if model.predict(input_df)[0] == 1:
                st.error("Result: Heart Disease Detected. Please consult a professional.")
            else:
                st.success("Result: No Heart Disease Detected.")
