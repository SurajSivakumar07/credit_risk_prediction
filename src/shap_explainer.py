# shap_explainer.py

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def plot_summary_shap(model, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.text(f"SHAP values shape: {np.array(shap_values).shape}")
    st.text(f"X_test shape: {X_test.shape}")

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names), plot_type="dot", show=False)
    else:
        shap.summary_plot(shap_values if not isinstance(shap_values, list) else shap_values[0],
                          pd.DataFrame(X_test, columns=feature_names), plot_type="dot", show=False)
    st.pyplot(plt.gcf())

def explain_single_prediction(model, input_scaled, feature_names):
    explainer = shap.TreeExplainer(model)
    input_shap = explainer.shap_values(input_scaled)

    shap_values_to_plot = input_shap[1] if isinstance(input_shap, list) and len(input_shap) > 1 else input_shap

    plt.figure(figsize=(10, 6))
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values_to_plot,
        pd.DataFrame(input_scaled, columns=feature_names),
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())

    plt.figure(figsize=(10, 6))
    shap_values_flat = shap_values_to_plot.flatten() if shap_values_to_plot.ndim > 1 else shap_values_to_plot
    indices = np.argsort(np.abs(shap_values_flat))
    plt.barh(
        [feature_names[i] for i in indices],
        [shap_values_flat[i] for i in indices]
    )
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title('Feature Impact on Prediction')
    st.pyplot(plt.gcf())
