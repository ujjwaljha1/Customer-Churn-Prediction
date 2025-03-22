import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# 🎨 Streamlit Page Config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# 📌 Sidebar - Project Introduction
st.sidebar.title("📚 About the Project")
st.sidebar.markdown("""
### Customer Churn Prediction
🔹 **Objective:** Predict whether a customer will leave a service (churn) or continue.  
🔹 **Dataset:** Uses telecom customer data.  
🔹 **Goal:** Help businesses retain customers by identifying potential churn risks.
""")

st.sidebar.write("💡 **Developed by:** Ujjwal Jha")
st.sidebar.write("📧 Email: ujjwaljha744@email.com")

# 📌 Load and Preprocess Data
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df.drop(['customerID'], axis=1, inplace=True)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df

df = load_data()

# 📌 Split dataset
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# 📌 Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Train XGBoost Model
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Main Title
st.title("📊 Customer Churn Prediction")
st.markdown("### 🌟 Predict if a customer will stay or leave based on their data!")

# 🎨 UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 📌 Enter Customer Details:")

    # 📌 User Input Fields
    tenure = st.slider("📅 Tenure (Months)", 0, 100, 24)
    monthly_charges = st.slider("💰 Monthly Charges ($)", 0.0, 200.0, 50.0)
    total_charges = st.slider("💳 Total Charges ($)", 0.0, 10000.0, 1000.0)

    # 📌 Prediction Function
    def predict_churn(tenure, monthly_charges, total_charges):
        new_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
        new_data["tenure"] = tenure
        new_data["MonthlyCharges"] = monthly_charges
        new_data["TotalCharges"] = total_charges
        input_data = scaler.transform(new_data)
        prediction = model.predict(input_data)[0]
        return "❌ Churn" if prediction == 1 else "✅ No Churn"

    # 📌 Predict Button
    if st.button("🔮 Predict Churn"):
        result = predict_churn(tenure, monthly_charges, total_charges)
        if result == "✅ No Churn":
            st.success(f"Prediction: {result}")
        else:
            st.error(f"Prediction: {result}")

with col2:
    # 📌 Display Model Accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.metric(label="📊 Model Accuracy", value=f"{accuracy:.2%}")

    # 📌 Confusion Matrix
    st.write("### 🔍 Model Performance")
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    st.pyplot(plt.gcf())  # Show confusion matrix

# 📌 Footer
st.markdown("---")
st.markdown("💡 *This AI-powered application predicts customer churn using XGBoost.*")

