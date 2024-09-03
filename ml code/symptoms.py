import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\Hp\Desktop\project 12\project 9\ml code\Testing.csv", header=None)

# Extract the first row, which contains the symptoms
symptoms = data.iloc[0, :-1].tolist()  # Exclude the last column which is 'prognosis'

# Create a DataFrame from the list of symptoms
symptoms_df = pd.DataFrame(symptoms, columns=['symptom'])

# Save the DataFrame to a new CSV file
symptoms_df.to_csv('list_of_symptoms.csv', index=False)
