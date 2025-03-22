# %%
# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert 'TotalCharges' to numeric (handle spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median value
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not relevant for prediction)
df.drop(['customerID'], axis=1, inplace=True)

# %%
# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Encode categorical variables
for col in categorical_cols:
    if df[col].nunique() == 2:  # Binary categorical values
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# %%
# Define features (X) and target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform training data
X_train = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test = scaler.transform(X_test)

# %%
# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# %%
# Print evaluation metrics
print("🔹 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# %%
# Function to make predictions for a new customer
def predict_churn(tenure, monthly_charges, total_charges):
    # Create a DataFrame with user input
    new_data = pd.DataFrame(columns=X.columns)

    # Assign input values
    new_data.loc[0] = 0  # Initialize with 0 for categorical features
    new_data["tenure"] = tenure
    new_data["MonthlyCharges"] = monthly_charges
    new_data["TotalCharges"] = total_charges

    # Scale new input
    input_data = scaler.transform(new_data)

    # Predict churn
    prediction = model.predict(input_data)[0]

    return "Churn" if prediction == 1 else "No Churn"

# Example: Predict for a new customer
print("🟢 New Customer Prediction:", predict_churn(tenure=24, monthly_charges=70, total_charges=1500))
