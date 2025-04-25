import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data
from model_training import train_model
from visualizations import plot_roc_curve, plot_shap_summary, plot_feature_importance
from insights import display_insights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap

st.set_page_config(layout="wide")
st.title("Credit Risk Prediction App")

# Sidebar for data upload
# Load and preprocess data
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocess data
X, y, feature_names, scaler, label_encoders = preprocess_data(df)

# Train model
model = train_model(X, y)

# Model evaluation
st.subheader("Model Evaluation")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
st.write(pd.DataFrame(confusion_matrix(y_test, y_pred)))

# Plot ROC curve
plot_roc_curve(y_test, y_prob)

# SHAP feature importance
st.subheader("Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plot_shap_summary(shap_values, X_test, feature_names)

# Credit risk insights
st.markdown("---")
st.header("Credit Risk Insights")
plot_feature_importance(model, feature_names)
display_insights(model, X, feature_names, explainer)

# Prediction section
st.markdown("---")
st.header("Make a Prediction")
input_data = {}
for feature in feature_names:
    if df[feature].dtype == 'object':
        options = list(label_encoders[feature].classes_)
        val = st.selectbox(f"{feature}", options)
        val = label_encoders[feature].transform([val])[0]
    else:
        val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()),
                              float(df[feature].mean()))
    input_data[feature] = val

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
st.write(f"Predicted Risk: {'Good' if prediction == 1 else 'Bad'}")
st.write(f"Probability of Good Credit: {probability:.2f}")

# Button to show insights again
if st.button("Click here to improve your account"):
    st.markdown("---")
    st.header("Credit Risk Insights")
    plot_feature_importance(model, feature_names)
    display_insights(model, X, feature_names, explainer)