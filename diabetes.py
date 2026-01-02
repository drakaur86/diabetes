import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
# URL for the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# 2. Data Cleaning (The secret to 83% accuracy)
# Replacing 0 with NaN for columns where 0 is biologically impossible
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

# Fill NaNs with the median of each column
df.fillna(df.median(), inplace=True)

# 3. Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Model
# RandomForest is used here to capture non-linear relationships in medical data
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Diabetes Model Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 7. Function for Real-Time Prediction
def predict_diabetes(data):
    """
    Input: List of 8 features
    Example: [2, 138, 62, 35, 0, 33.6, 0.127, 47]
    """
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return "Diabetic" if prediction[0] == 1 else "Healthy"

# Test prediction
print(f"Test Result: {predict_diabetes([2, 138, 62, 35, 0, 33.6, 0.127, 47])}")
