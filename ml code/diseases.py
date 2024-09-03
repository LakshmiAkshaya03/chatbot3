import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\Hp\Desktop\project 12\project 9\ml code\Testing.csv")

# Get the list of symptoms from the header (excluding the last column 'prognosis')
symptoms = data.columns[:-1]

# Create a list to store diseases and their associated symptoms
disease_symptoms_list = []

# Iterate through each row to populate the list
for index, row in data.iterrows():
    disease = row['prognosis']
    associated_symptoms = symptoms[row[symptoms] == 1].tolist()
    disease_symptoms_list.append({'disease': disease, 'symptoms': ', '.join(associated_symptoms)})

# Create a DataFrame from the list
disease_symptoms_df = pd.DataFrame(disease_symptoms_list)

# Save the DataFrame to a new CSV file
disease_symptoms_df.to_csv('disease_symptoms.csv', index=False)
