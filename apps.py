import streamlit as st
import pandas as pd
import joblib
model= joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")
st.title("Heart Disease Prediction App")
st.markdown("Please provide the following details:")
age=st.slider("Age",18,100,40)
Sex=st.selectbox("Sex",["M","F"])
chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp=st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
cholestrol=st.number_input("Cholestrol(mg/dl)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar >120 mg/dl",[0,1])
resting_ECG=st.selectbox("Resting ECG Results",["Normal","ST","LVH"])
max_hr=st.slider("Max Heart Rate Achieved",60,220,150)
exercise_angina=st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak=st.number_input("Oldpeak(ST depression)", 0.0, 10.0, 1.0)
st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])
if st.button("Predict Heart Disease"):
    raw_input= {
        "Age":age,
        "Sex" + Sex: 1,
        "ChestPainType" + chest_pain: 1,
        "RestingBP": resting_bp,
        "Cholestrol":cholestrol,
        "FastingBS":fasting_bs,
        "RestingECG"+resting_ECG: 1,
        "MaxHR":max_hr,
        "ExerciseAngina"+exercise_angina: 1,
        "Oldpeak":oldpeak,
        "ST_Slope" + st_slope: 1
    }
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns]
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]
    if prediction==1:
        st.error("⚠️High risk of heart disease. Please consult a doctor.")
    else:
        st.success("😊Low risk of heart disease.")

    


