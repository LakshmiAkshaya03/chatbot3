import pandas as pd
import joblib

# Load the model
model_path = r"C:\Users\Hp\OneDrive\Desktop\trained_model_temp.joblib"
pickled_model = joblib.load(model_path)

# Define the function to get user input and predict the disease
def predict_disease(symptoms_input):
    # Convert input to a list of symptoms
    symptoms = symptoms_input.split(',')
    
    # Format the input correctly
    input_dict = {col: 0 for col in pickled_model.feature_names_in_}  # Use feature names from the model
    for symptom in symptoms:
        symptom = symptom.strip()  # Remove any leading/trailing whitespace
        if symptom in input_dict:
            input_dict[symptom] = 1

    input_df = pd.DataFrame([input_dict])

    # Predict the disease
    predicted_disease = pickled_model.predict(input_df)
    return predicted_disease[0]

# Example usage
if __name__ == "__main__":
    symptoms = input("Enter the symptoms, so we can predict the disease (comma-separated): ")
    predicted_disease = predict_disease(symptoms)
    print("Predicted Disease is:", predicted_disease)
