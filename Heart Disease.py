import streamlit as st
import pandas as pd
import pickle

# Load the saved components
with open('heartdisease_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('heartdisease_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('heartdisease_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("Heart Disease Prediction App")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=50)
        is_female = st.selectbox("Sex", [("Male", 0), ("Female", 1)], format_func=lambda x: x[0])[1]
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        maxhr = st.number_input("Max Heart Rate (MaxHR)", min_value=60, max_value=220, value=140)
        
    with col2:
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0)
        chol = st.number_input("Cholesterol", min_value=0, value=300)
        exang = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        st_slope = st.selectbox("ST Slope", [("Up", 0), ("Flat", 1), ("Down", 2)], format_func=lambda x: x[0])[1]
        cp_type = st.selectbox("Chest Pain Type", [("ASY", 0), ("ATA", 1), ("NAP", 2), ("TA", 3)], format_func=lambda x: x[0])[1]

    submit = st.form_submit_button("Predict")

if submit:
    # Build dictionary (Fixed: Removed duplicate 'Cholesterol' key)
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
    # Ensure column order matches exactly what the model expects
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Scale (Ensure these columns exist in your scaler's training set)
    cols_to_scale = ['Age', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    # Predict
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
