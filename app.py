import streamlit as st
import numpy as np
import joblib

def load_model_and_scaler():
    """Load the trained model and scaler from files."""
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter patient details to predict the likelihood of diabetes.")
    
    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Plasma Glucose", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Triceps Skin Fold Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Serum Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    if st.button("Predict"):
        model, scaler = load_model_and_scaler()
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
        st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()