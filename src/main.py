# app.py

import streamlit as st
import pandas as pd
from data_loader import load_data, preprocess_data
from model import train_model
from shap_explainer import plot_summary_shap, explain_single_prediction
from utils import column_mapping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Credit Risk Prediction App")

uploaded_file = st.sidebar.file_uploader("Upload German Credit Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.rename(columns=column_mapping, inplace=True)
else:
    df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X, y, feature_names, scaler, label_encoders = preprocess_data(df)
model = train_model(X, y)

# Evaluation
st.subheader("Model Evaluation")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# SHAP Summary
st.subheader("Feature Importance (SHAP)")
plot_summary_shap(model, X_test, feature_names)

# Prediction Section
st.markdown("---")
st.header("Make a Prediction")
input_data = {}
for feature in feature_names:
    if df[feature].dtype == 'object':
        options = list(label_encoders[feature].classes_)
        val = st.selectbox(f"{feature}", options)
        val = label_encoders[feature].transform([val])[0]
    else:
        val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_data[feature] = val

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
st.write("Predicted Credit Risk:", "Good" if prediction == 1 else "Bad")
st.write("Probability of Good Credit Risk:", f"{probability:.2f}")

if st.button("Explain this prediction"):
    st.subheader("Feature Impact on Prediction")
    explain_single_prediction(model, input_scaled, feature_names)
