import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    plt.clf()

def plot_shap_summary(shap_values, X_test, feature_names):
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values[0], pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_feature_importance(model, feature_names):
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Features Influencing Credit Risk')
    ax.set_xlabel('Relative Importance')
    st.pyplot(fig)
    plt.clf()

def plot_shap_analysis(X, feature_names, explainer):
    sample_size = min(100, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    sample_shap_values = explainer.shap_values(X_sample)
    if isinstance(sample_shap_values, list) and len(sample_shap_values) > 1:
        sample_shap_to_analyze = sample_shap_values[1]
    else:
        sample_shap_to_analyze = sample_shap_values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sample_shap_to_analyze, X_sample, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())
    plt.clf()