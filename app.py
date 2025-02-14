import numpy as np
import streamlit as st
import pickle

# Page configuration
st.set_page_config(page_title="Heart Disease Prediction App", page_icon="\u2764\ufe0f", layout="wide")

# Custom styles
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6;}
    h1 {color: #d63384; text-align: center; font-family: 'Arial', sans-serif;}
    .stButton > button {background-color: #d63384; color: white; border-radius: 10px;}
    .stButton > button:hover {background-color: #a02364;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("\u2764\ufe0f Heart Disease Prediction App \u2764\ufe0f")
st.image("https://huggingface.co/spaces/Abhisikta-26201/Heart_Disease_Classification/resolve/main/IMG_20241212_011225.jpg", use_container_width=True)

# Input fields
with st.form("input_form"):
    st.header("Enter Patient Information")

    # Column layout for inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, step=1, format="%d", help="Enter the age of the patient (in Years)")
        gender = st.radio("Gender", ['Male', 'Female'], horizontal=True, help="Select the patient's gender")
        chestpain = st.selectbox(
            "Chest Pain Type",
            ['No Data Available','Non-anginal_pain', 'Typical_angina', 'Atypical_angina','Asymptomatic'],
            help = "Select the type of chest pain. 'No Data Available' corresponds to an unknown value. If selected, the 'most frequent' value of the category will be used for prediction."
        )
        chestpain = np.nan if chestpain == 'No Data Available' else chestpain
        restingBP = st.number_input("Resting Blood Pressure", min_value = (-1), max_value=300, step=1, format="%d", help = "Enter the resting blood pressure (in mmHg). '-1' corresponds to an unknown resting BP level. If selected, then mean value of the category will be used for prediction.")
        restingBP = np.nan if restingBP == (-1) else restingBP

    with col2:
        serum_cholesterol = st.number_input(
            "Serum Cholesterol", min_value=(-1), max_value=1000, step=1, format="%d", help = "Enter serum cholesterol level (in mg/dL). '-1' corresponds to an unknown serum choesterol level. If selected, then mean value of the category will be used for prediction."
        )
        serum_cholesterol = np.nan if serum_cholesterol == (-1) else serum_cholesterol
        fasting_blood_sugar = st.radio(
            "Fasting Blood Sugar > 120 mg/dL", ['Yes', 'No','No Data Available'], horizontal=True, help ="Indicate if fasting blood sugar is greater than 120 mg/dL. 'No Data Available' corresponds to an unknown fasting blood sugar level. If selected, the 'most frequent' value of the category will be used for prediction."
        )
        fasting_blood_sugar = np.nan if fasting_blood_sugar == 'No Data Available' else fasting_blood_sugar
        restingrelectro = st.selectbox(
            "Resting Electrocardiographic Results",
            ['No Data Available','ST-T_wave_abnormality', 'Normal', 'Left_ventricular_hypertrophy'],
            help = "Select the resting electrocardiographic result. 'No Data Available' corresponds to an unknown ECG result. If selected, the 'most frequent' value of the category will be used for prediction."
        )
        restingrelectro = np.nan if restingrelectro == 'No Data Available' else restingrelectro
        maxheartrate = st.number_input(
            "Maximum Heart Rate Achieved", min_value=(-1), max_value=250, step=1, format="%d", help = "Enter the maximum heart rate achieved. '-1' corresponds to an unknown maximum heart rate. If selected, then mean value of the category will be used for prediction."
        )
        maxheartrate = np.nan if maxheartrate == (-1) else maxheartrate


    # Second row of inputs
    exerciseangia = st.radio("Exercise-Induced Angina", ['Yes', 'No','No Data Available'], horizontal=True, help = "Indicate if exercise induced angina is present or not. 'No Data Available' corresponds to an unknown value. If selected, the 'most frequent' value of the category will be used for prediction.")
    exerciseangia = np.nan if exerciseangia == 'No Data Available' else exerciseangia
    oldpeak = st.number_input("ST Depression Induced by Exercise",min_value = -0.1, max_value = 10.0 ,step = 0.1, help = "Enter ST depression value induced by exercise relative to rest. '-0.1' corresponds to an unknown value. If selected, then mean value of the category will be used for prediction.")
    oldpeak = np.nan if oldpeak == -0.1 else oldpeak
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ['No Data Available','Downsloping', 'Upsloping', 'Flat'], help = "A normal ST segment during exercise slopes sharply upwards. The slope of the ST segment is the shift of the ST segment relative to the increase in heart rate caused by exercise. 'No Data Available' corresponds to an unknown slope value. If selected, the 'most frequent' value of the category will be used for prediction.")
    slope = np.nan if slope == 'No Data Available' else slope
    noofmajorvessels = st.selectbox(
        "Number of Major Vessels", ['No Data Available','Zero', 'One', 'Two', 'Three'],help = "Fluoroscopy is a procedure used during cardiac catheterization to help doctors see blood flow through the coronary arteries. This allows doctors to evaluate for arterial blockages.'No Data Available' corresponds to an unknown value. If selected, the 'most frequent' value of the category will be used for prediction.")
    noofmajorvessels = np.nan if noofmajorvessels == 'No Data Available' else noofmajorvessels

    # Submit button
    submitted = st.form_submit_button("Predict")

# Load model
model_1 = pickle.load(open(r"rfc.pkl", "rb"))

# Prediction
if submitted:
    with st.spinner("Analyzing the data..."):
        result = model_1.predict([[
            age, gender, chestpain, restingBP, serum_cholesterol,
            fasting_blood_sugar, restingrelectro, maxheartrate, exerciseangia,
            oldpeak, slope, noofmajorvessels
        ]])
        
    # Display result
    st.markdown(
        "### Prediction Result:", unsafe_allow_html=True
    )

    if result[0] == "Present":
        st.info("The patient is likely to have heart disease. Please consult a doctor for further evaluation.")
    else:
        st.success("The patient is unlikely to have heart disease. However, maintaining regular check-ups is recommended.")

# Footer
st.markdown(
    '''Developed by Abhisikta Moharana<br>    
    <a href="https://www.linkedin.com/in/abhisikta-moharana-983052270" target="_blank" style="text-decoration:none; color:#d63384;">
    LinkedIn
    </a><br>
    <a href="mailto:abhisikta.moharana2001@gmail.com" style="text-decoration:none; color:#d63384;">
    Email
    </a>'''
    ,
    unsafe_allow_html=True,
)