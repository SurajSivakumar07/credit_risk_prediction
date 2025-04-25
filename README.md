# Credit Risk Prediction App

This is an interactive Streamlit web application that predicts the credit risk (Good or Bad) of individuals based on the German Credit dataset. It includes data preprocessing, model training using a Random Forest Classifier, model evaluation, and SHAP-based explainability for predictions.


---

## 🚀 Features

- Upload your own German Credit dataset or use the default one
- View a preview of the dataset
- Model training using GridSearchCV for hyperparameter tuning
- Visualize classification metrics (Report, Confusion Matrix, ROC Curve)
- SHAP-based feature importance visualization
- Predict credit risk for a new user input
- Explain predictions using SHAP force and bar plots

---
## Dataset
    https://www.kaggle.com/datasets/ashrafkhan94/german-credit-history/data
## 📦 Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt

streamlit run main.py
