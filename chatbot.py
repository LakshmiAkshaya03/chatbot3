from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from markupsafe import Markup
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import re
import firebase_admin
from firebase_admin import auth, credentials
import google.generativeai as genai
import csv
import secrets
import markdown2
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__, template_folder=r'C:\Users\Hp\Desktop\project 12\project 9\templates', static_folder=r'C:\Users\Hp\Desktop\project 12\project 9\static')

# Set your Google API key here
GOOGLE_API_KEY = 'AIzaSyCXceCtbVBxgmwf1j4PfdXTHQHWdIutmK0'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r'C:\Users\Hp\Desktop\project 12\project 9\health-chatbot-c0564-firebase-adminsdk-d26xv-ca5c6b7735.json')
firebase_admin.initialize_app(cred)

# Generate random secret key
secret_key = secrets.token_hex(16)
app.secret_key = secret_key

# Load the trained ML model
model_path = r"C:\Users\Hp\OneDrive\Desktop\trained_model_temp.joblib"
pickled_model = joblib.load(model_path)

def read_google_sheet(sheet_key, credentials_file):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_key).sheet1
    records = sheet.get_all_records()
    data = {}
    for row in records:
        hospital = row['Hospital Name']
        location = row['Location']  # New line to read location information
        doctor = {
            'name': row['Doctor Name'],
            'specialization': row['Specialization'],
            'experience': row['Experience'],
            'link': row['Slot URL'],
            'location': location  # Include location in doctor data
        }
        if hospital not in data:
            data[hospital] = []
        data[hospital].append(doctor)
    return data

def authenticate_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return True
    except firebase_admin.auth.UserNotFoundError:
        return False

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

# New function to read the list of symptoms from the CSV file
def read_symptoms(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        symptoms = [row[0] for row in reader]
    return symptoms

# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'email' in session:
        error = "You are already signed in. Please log out to create a new account."
        return render_template('signup.html', error=error)

    error = None  # Initialize error variable

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Regular expression pattern to match alphanumeric characters only
        alphanumericRegex = re.compile(r'^[a-zA-Z0-9]+$')

        # Check if the password meets the criteria
        if len(password) < 6 or len(password) > 8 or not alphanumericRegex.match(password):
            error = "Invalid password. Password must be between 6 and 8 characters long and consist of alphabets and numbers only."
        # Check if password contains both alphabets and numbers
        elif not any(char.isdigit() for char in password) or not any(char.isalpha() for char in password):
            error = "Invalid password. Password must contain both alphabets and numbers."
        # Check if password contains only alphabets or only numbers
        elif password.isalpha() or password.isdigit():
            error = "Invalid password. Password must contain both alphabets and numbers."
        else:
            try:
                user = auth.create_user(email=email, password=password)
                return redirect(url_for('login'))
            except firebase_admin.auth.EmailAlreadyExistsError:
                error = 'Email already exists. Please use a different email address.'
            except Exception as e:
                error = str(e)

    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if not authenticate_user(email, password):
            return render_template('login.html', error='Invalid email or password')
        else:
            session['email'] = email
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    # Path to the symptoms CSV file
    symptoms_csv_file = r'C:\Users\Hp\Desktop\project 12\project 9\ml code\list_of_symptoms.csv'
    symptoms_list = read_symptoms(symptoms_csv_file)

    return render_template('dashboard.html', symptoms=symptoms_list)

@app.route('/appointment')
def appointment():
    if 'email' not in session:
        return redirect(url_for('login'))
    sheet_key = "1i1wvJSI6eQ2oMCkC1X36iq35CayMrVK-t7XHztYPx0M"
    credentials_file =r"/Users/akshaya/Downloads/chatbot-419505-68988d9df9fa.json"
    data = read_google_sheet(sheet_key, credentials_file)
    return render_template('appointment.html', data=data)

@app.route('/hospital')
def hospital():
    sheet_key = "1i1wvJSI6eQ2oMCkC1X36iq35CayMrVK-t7XHztYPx0M"
    credentials_file =r"C:\Users\Hp\Desktop\project 12\project 9\chatbot-419505-68988d9df9fa.json"
    data = read_google_sheet(sheet_key, credentials_file)

    hospitals = {}
    for hospital, doctors in data.items():
        if hospital not in hospitals:
            hospitals[hospital] = []
        hospitals[hospital].extend(doctors)

    return render_template('csv_content.html', data=data)

@app.route('/api/chatbot', methods=['GET', 'POST'])
def chatbot():
    # Path to the symptoms CSV file
    symptoms_csv_file = r'C:\Users\Hp\Desktop\project 12\project 9\ml code\list_of_symptoms.csv'
    symptoms_list = read_symptoms(symptoms_csv_file)

    if request.method == 'GET':
        return render_template('chatbot.html', symptoms=symptoms_list)

    if request.method == 'POST':
        data = request.json
        symptoms = data.get('symptoms', '')

        # Predict the disease using the trained model
        predicted_disease = predict_disease(symptoms)

        print("Predicted Disease is:", predicted_disease)
        # Generate the response using generative AI
        try:
            prompt = f"The disease predicted based on the symptoms is {predicted_disease}. Please provide only precautions to take. dont give me the any medication and last advice better to contact the doctor.followed by that give all possible disease  for symptoms {symptoms}and give its precaution and tell which is more closely macthing  "
            response = model.generate_content(prompt)

            # Clean the response text to remove "*" and "**"
            clean_response = re.sub(r'\*+', '', response.text)

            # Use Markup to mark the HTML as safe
            safe_html =f"predicted disease is {predicted_disease} \n \n"+ Markup(clean_response)

            return jsonify({'message': str(safe_html)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
