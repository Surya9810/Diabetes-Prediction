import numpy as np
import pandas as pd
import joblib

def load_model_and_scaler():
    """Load the trained model and scaler from files."""
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Ensure scaler.pkl exists
    return model, scaler

def get_user_input():
    """Prompt the user to enter values for prediction."""
    print("Enter the following values:")
    pregnancies = float(input("Pregnancies: "))
    glucose = float(input("Plasma Glucose: "))
    blood_pressure = float(input("Diastolic Blood Pressure: "))
    skin_thickness = float(input("Triceps Skin Fold Thickness: "))
    insulin = float(input("Serum Insulin: "))
    bmi = float(input("BMI: "))
    pedigree = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))

    # Convert user input to DataFrame with correct feature names
    feature_names = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]
    user_input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, 
                                    skin_thickness, insulin, bmi, pedigree, age]], 
                                  columns=feature_names)
    
    return user_input_df

def main():
    """Load model, get user input, and make a prediction."""
    model, scaler = load_model_and_scaler()
    user_input_df = get_user_input()
    
    # Ensure input is scaled correctly
    user_input_scaled = scaler.transform(user_input_df)  # No feature name warning now

    # Predict
    prediction = model.predict(user_input_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    print(f"\nPrediction: {result}")

if __name__ == "__main__":
    main()
