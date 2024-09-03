import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib as jb

# Set display option
pd.set_option('display.max_columns', None)

# Load data
train = pd.read_csv(r'C:\Users\Hp\Desktop\Gagan\disease_prediction_and_doctor_recommendation_system-t5tkju\Dataset and Training Script\Training.csv')
test = pd.read_csv(r'C:\Users\Hp\Desktop\Gagan\disease_prediction_and_doctor_recommendation_system-t5tkju\Dataset and Training Script\Testing.csv')
data = pd.concat([train, test])

# Analyze data
print(sorted(data.columns.tolist()[:-1]))
print(data.head(20))
print(data.columns)
print(data.prognosis.value_counts())
print(data.info())
print(data.describe())

data_X = {'Symptoms': [], 'Prognosis': [], 'length': []}
table = pd.DataFrame(data_X)
table = table.astype({"Symptoms": str, "Prognosis": object, 'length': int})

# Use pd.concat instead of append
rows = []
for symp in sorted(data.columns.tolist()[:-1]):
    prognosis = data[data[symp] == 1].prognosis.unique().tolist()
    row = {'Symptoms': symp, 'Prognosis': prognosis, 'length': len(prognosis)}
    rows.append(row)

table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)

print(table.sort_values(by='length', ascending=False).head(10))

# Preprocessing function
def preprocess_inputs(df):
    df = df.copy()
    y = df['prognosis']
    X = df.drop('prognosis', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

# Training (Original Data)
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, y_train)
LR_score = LR.score(X_test, y_test)
print("LR Accuracy: {:.2f}%".format(LR_score*100))

svc = SVC(gamma="auto", kernel="rbf")
svc.fit(X_train, y_train)
SVC_score = svc.score(X_test, y_test)
print("SVC Accuracy: {:.2f}%".format(SVC_score * 100))

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)
DT_score = DT.score(X_test, y_test)
print("DT Accuracy: {:.2f}%".format(DT_score * 100))

rfc = RandomForestClassifier(random_state=42, n_estimators=100)
rfc.fit(X_train, y_train)
RFC_score = rfc.score(X_test, y_test)
print("RFC Accuracy: {:.2f}%".format(RFC_score * 100))

# Algorithm comparison
algorithms = {
    "DecisionTree": DecisionTreeClassifier(max_depth=100),
    "RandomForest": RandomForestClassifier(n_estimators=10),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50),
    "K-Nearest": KNeighborsClassifier(n_neighbors=5),
    "GNB": GaussianNB()
}

results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

# Feature Selection
coefficients = np.mean(LR.coef_, axis=0)
importance_threshold = np.quantile(np.abs(coefficients), q=0.28)

fig = px.bar(
    x=coefficients,
    y=X_train.columns,
    orientation='h',
    color=coefficients,
    color_continuous_scale=[(0, 'red'), (1, 'blue')],
    labels={'x': "Coefficient Value", 'y': "Feature"},
    title="Feature Importance From Model Weights"
)

fig.add_vline(x=importance_threshold, line_color='yellow')
fig.add_vline(x=-importance_threshold, line_color='yellow')
fig.add_vrect(x0=importance_threshold, x1=-importance_threshold, line_width=0, fillcolor='yellow', opacity=0.2)

fig.show()

low_importance_features = X_train.columns[np.abs(coefficients) < importance_threshold]
reduced_data = data.drop(low_importance_features, axis=1).copy()

X_train, X_test, y_train, y_test = preprocess_inputs(reduced_data)

# Algorithm comparison on reduced data
print("\nNow testing algorithms on reduced data")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

reduced_data_model = LogisticRegression(max_iter=1000)
reduced_data_model.fit(X_train, y_train)

print("Test Accuracy: {:.2f}%".format(reduced_data_model.score(X_test, y_test) * 100))
y_pred = reduced_data_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(np.arange(41) + 0.5, reduced_data_model.classes_, rotation=90)
plt.yticks(np.arange(41) + 0.5, reduced_data_model.classes_, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

winner = max(results, key=results.get)
print('\nAlgorithm with highest accuracy on train/test is %s with a %f %% success' % (winner, results[winner]*100))

# Save the best model
jb.dump(algorithms[winner], 'trained_model_temp.joblib')

import joblib
# Load the model
pickled_model = joblib.load(r"C:\Users\Hp\OneDrive\Desktop\trained_model_temp.joblib")

# Input handling
new_input = input("Enter the symptoms, so we can predict the disease (comma-separated): ").split(',')

# Format the input correctly
new_input_dict = {col: 0 for col in X_train.columns}
for symptom in new_input:
    symptom = symptom.strip()  # Remove any leading/trailing whitespace
    if symptom in new_input_dict:
        new_input_dict[symptom] = 1

new_input_df = pd.DataFrame([new_input_dict])

# Predict the disease
predicted_disease_type = pickled_model.predict(new_input_df)
print("Predicted Disease is:", predicted_disease_type[0])
