import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import joblib

from sta import some

st.set_page_config(layout="wide")
st.title("Credit Risk Prediction App")

column_mapping = {
    'A11': 'Status of existing checking account',
    '6': 'Duration in month',
    'A34': 'Credit history',
    'A43': 'Purpose',
    '1169': 'Credit amount',
    'A65': 'Savings account',
    'A75': 'Present employment since',
    '4': 'Installment rate in percentage of disposable income',
    'A93': 'Personal status and sex',
    'A101': 'Other debtors',
    '4.1': 'Present residence since',
    'A121': 'Property',
    '67': 'Age in years',
    'A143': 'Other installment plans',
    'A152': 'Housing',
    '2': 'Number of existing credits at this bank',
    'A173': 'Job',
    '1': 'Number of dependents',
    'A192': 'Telephone',
    'A201': 'Foreign worker',
    '1.1': 'Risk'
}


@st.cache_data
def load_data():
    df = pd.read_csv("./data/german.csv")

    # Apply value mappings
    value_mappings = {
        'A11': {'A11': 'Person has a negative balance', 'A12': 'Person has a small balance (0 <= ... < 200 DM)', 'A13': 'Person has a large balance (>= 200 DM or salary assignments)', 'A14': 'person does not have a checking account.'},
        'A34': {'A30': 'no credits taken', 'A31': 'all paid back', 'A32': 'existing credits paid duly', 'A33': 'delay in past', 'A34': 'critical/other loans'},
        'A43': {'A40': 'radio/TV', 'A41': 'education', 'A42': 'furniture', 'A43': 'new car', 'A44': 'used car', 'A45': 'business',
                'A46': 'domestic appliances', 'A47': 'repairs', 'A48': 'vacation', 'A49': 'retraining', 'A410': 'others'},
        'A65': {'A61': 'unemployed', 'A62': '< 1 year', 'A63': '1 <= ... < 4 years', 'A64': '4 <= ... < 7 years', 'A65': '>= 7 years'},
        'A75': {'A71': 'male single', 'A72': 'male married/widowed', 'A73': 'male divorced', 'A74': 'female divorced/married', 'A75': 'female single'},
        'A93': {'A91': 'own', 'A92': 'for free', 'A93': 'rent'},
        'A101': {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
        'A121': {'A121': 'real estate', 'A122': 'building society savings/life insurance', 'A123': 'car or other', 'A124': 'unknown/no property'},
        'A143': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
        'A152': {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
        'A173': {'A171': 'unskilled non-resident', 'A172': 'unskilled resident', 'A173': 'skilled employee', 'A174': 'highly qualified/self-employed'},
        'A192': {'A191': 'yes', 'A192': 'no'},
        'A201': {'A201': 'good', 'A202': 'bad'}
    }

    # Apply value mappings
    for col, mapping in value_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # Rename column headers
    df.rename(columns=column_mapping, inplace=True)
    return df


@st.cache_data
def preprocess_data(df):
    df = df.copy()
    df['Risk'] = df['Risk'].replace({1: 1, 2: 0})  # 1: Good, 2: Bad -> Convert to binary
    categorical_cols = df.select_dtypes(include='object').columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, df.columns[:-1], scaler, label_encoders


@st.cache_data
def train_model(X, y):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_


# Load and preprocess
st.sidebar.header("Upload Dataset")
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

# Metrics
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

# SHAP Explanation
st.subheader("Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Debug information to understand shapes
st.text(f"SHAP values shape: {np.array(shap_values).shape}")
st.text(f"X_test shape: {X_test.shape}")

# Create SHAP plot with proper handling of shape
if isinstance(shap_values, list):
    # If shap_values is a list of arrays (one per class)
    if len(shap_values) > 1:
        # Binary classification typically returns two arrays
        shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
    else:
        # Single array in a list
        shap.summary_plot(shap_values[0], pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)
else:
    # If shap_values is a single array
    shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names), plot_type="bar", show=False)

# Display the plot in Streamlit
st.pyplot(plt.gcf())
plt.clf()  # Clear the current figure to avoid overlapping plots
# Add after the model evaluation section but before the prediction section

# New section for credit risk insights

st.markdown("---")
st.header("Credit Risk Insights")
input_data = {}


# Calculate global feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Display feature importance
st.subheader("Key Factors Influencing Credit Risk")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
ax.set_title('Top 10 Features Influencing Credit Risk')
ax.set_xlabel('Relative Importance')
st.pyplot(fig)

# Analyze global SHAP values for more nuanced insights
st.subheader("In-Depth SHAP Analysis")
# Sample a smaller subset for SHAP analysis to improve performance
sample_size = min(100, X.shape[0])
sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X[sample_indices]

# Calculate SHAP values for sample
sample_shap_values = explainer.shap_values(X_sample)
if isinstance(sample_shap_values, list) and len(sample_shap_values) > 1:
    # For binary classification, use values for "good credit" class
    sample_shap_to_analyze = sample_shap_values[1]
else:
    sample_shap_to_analyze = sample_shap_values

# Show SHAP summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(sample_shap_to_analyze, X_sample, feature_names=feature_names, show=False)
st.pyplot(plt.gcf())
plt.clf()

# Provide text insights based on feature importance and SHAP analysis
st.subheader("Key Insights")

# Get top features from both analyses
top_features = feature_importance['Feature'].head(5).tolist()
top_features_text = ", ".join(top_features[:-1]) + f", and {top_features[-1]}"

st.write(f"""
Based on our analysis, the most significant factors influencing credit risk in this dataset are {top_features_text}. Let's explore what these mean for credit evaluation:
""")

# Define insights for common important features in German credit data
feature_insights = {
    'Status of existing checking account': """
    **Checking Account Status**: A customer's checking account status strongly correlates with credit risk. Those with no checking account or negative balances typically represent higher risk, while substantial positive balances indicate financial stability and lower risk.

    **Improvement Strategy**: Consider offering secured credit products for customers with negative balances, and implement tiered interest rates based on checking account status.
    """,

    'Duration in month': """
    **Loan Duration**: Longer loan terms generally correlate with higher risk. This could be due to increased uncertainty over extended periods or reflect riskier borrowers seeking to minimize monthly payments.

    **Improvement Strategy**: For longer-term loans, consider implementing stricter qualification criteria or requiring additional collateral. Offer incentives for shorter loan terms.
    """,

    'Credit history': """
    **Credit History**: Past repayment behavior strongly predicts future behavior. Customers with spotless payment histories represent lower risk than those with missed payments.

    **Improvement Strategy**: Develop more nuanced credit history scoring that weighs recent behavior more heavily than older incidents. Consider specialized products for those with limited credit history.
    """,

    'Purpose': """
    **Loan Purpose**: The reason for borrowing significantly impacts risk. Business loans and certain consumer purchases may have different risk profiles.

    **Improvement Strategy**: Adjust risk assessment based on the specific purpose, with more favorable terms for historically lower-risk purposes. Develop specialized evaluation criteria for different loan types.
    """,

    'Credit amount': """
    **Loan Amount**: Higher credit amounts often correlate with increased risk, possibly because they represent a greater financial burden relative to income.

    **Improvement Strategy**: Implement progressive loan-to-income ratio limits and offer stepped lending programs that allow borrowers to qualify for larger amounts after demonstrating repayment ability.
    """,

    'Age in years': """
    **Age**: Age can correlate with financial stability and repayment behavior, with middle-aged borrowers often representing lower risk than very young or elderly applicants.

    **Improvement Strategy**: Develop age-appropriate financial education programs and tailor product offerings based on life stage needs while maintaining age compliance regulations.
    """,

    'Present employment since': """
    **Employment Duration**: Longer employment history generally indicates stability and lower credit risk.

    **Improvement Strategy**: For newer employees, consider additional factors like education, industry, and career progression. Offer credit-building products for those new to the workforce.
    """,

    'Property': """
    **Property Ownership**: Owning property, especially real estate, typically correlates with lower credit risk as it indicates financial stability and provides potential collateral.

    **Improvement Strategy**: Develop differentiated offerings for property owners vs. non-owners, potentially with secured options for the latter group.
    """,

    'Personal status and sex': """
    **Personal Status**: Marital status and household structure can impact financial stability and risk profiles.

    **Improvement Strategy**: Focus on household income and expenses rather than status itself, ensuring fair evaluation while recognizing household financial dynamics.
    """,

    'Housing': """
    **Housing Situation**: Homeowners often represent lower credit risk than renters, potentially due to demonstrated financial responsibility and stability.

    **Improvement Strategy**: Consider rent payment history as a positive factor for renters, and develop housing-specific risk assessment models.
    """
}

# Display insights for top features
for feature in top_features[:3]:  # Show insights for top 3 features
    if feature in feature_insights:
        st.markdown(feature_insights[feature])
    else:
        st.write(f"**{feature}**: This feature shows significant impact on credit risk assessment.")

# Add general improvement strategies section
st.subheader("Strategies for Improving Credit Evaluation")
st.write("""
Based on our analysis, here are recommendations to enhance your credit risk evaluation process:

1. **Implement Multi-Factor Scoring**: Rather than relying heavily on a few features, develop a balanced scorecard that considers diverse aspects of an applicant's financial profile.

2. **Segment-Specific Models**: Create specialized evaluation models for different customer segments (e.g., young professionals, retirees, self-employed) that account for their unique circumstances.

3. **Behavioral Indicators**: Incorporate transaction patterns and financial behaviors from checking and savings accounts into risk assessment.

4. **Progressive Lending**: Establish a stepped approach that allows customers to access higher credit limits after demonstrating responsible usage.

5. **Alternative Data Sources**: Consider non-traditional data sources like utility payments, rent history, and telecom payment records, especially for thin-file customers.

6. **Regular Model Retraining**: Credit risk factors change over time due to economic conditions and demographic shifts. Implement a schedule to retrain models with fresh data.

7. **Explainable AI Approach**: Ensure credit decisions can be explained to customers, which improves transparency and helps applicants understand how to improve their creditworthiness.

8. **Economic Adjustments**: Incorporate macroeconomic indicators into your models to adjust risk thresholds during different economic cycles.
""")

# Add a section for personalized risk mitigation
# Add a section for personalized risk mitigation
# st.subheader("Personalized Risk Mitigation Tool")
# st.write("""
# Explore how changing certain factors could potentially improve an applicant's credit risk profile.
# Adjust the sliders below to see how modifications might affect the prediction.
# """)
#
# # Create a copy of the input data for modification
# if 'input_data' in locals():
#     modified_data = input_data.copy()
#     modified_df = pd.DataFrame([modified_data])
#
#     # Select top numerical features for modification
#     numerical_features = [f for f in feature_names if df[f].dtype != 'object']
#     top_numerical = [f for f in top_features if f in numerical_features]
#
#     # Debug info
#     st.text(f"Available features in input data: {list(modified_data.keys())}")
#     st.text(f"Top numerical features identified: {top_numerical}")
#
#     # Mapping between different naming conventions
#     feature_mapping = {
#         'Credit amount': ['Credit amount', 'Credit Amount', '1169', 'Credit_amount'],
#         'Duration in month': ['Duration in month', 'Duration', '6', 'Duration_in_month'],
#         'Age in years': ['Age in years', 'Age', '67', 'Age_in_years']
#     }
#
#     st.write("Adjust key factors:")
#     modified = False
#
#     for feature in top_numerical[:3]:
#         actual_key = None
#
#         # Try all aliases to find actual key in input_data
#         for alias in feature_mapping.get(feature, [feature]):
#             if alias in modified_data:
#                 actual_key = alias
#                 break
#
#         if actual_key is None:
#             st.warning(f"Could not find feature '{feature}' in input data.")
#             continue
#
#         try:
#             min_val = float(df[feature].min())
#             max_val = float(df[feature].max())
#             current_val = float(modified_data[actual_key])
#
#             new_val = st.slider(
#                 f"Adjust {feature}",
#                 min_value=min_val,
#                 max_value=max_val,
#                 value=current_val,
#                 step=(max_val - min_val) / 100
#             )
#
#             if new_val != current_val:
#                 modified_data[actual_key] = new_val
#                 modified = True
#
#         except Exception as e:
#             st.error(f"Error processing feature '{feature}': {str(e)}")
#             continue
#
#     if modified:
#         modified_df = pd.DataFrame([modified_data])
#         modified_scaled = scaler.transform(modified_df)
#         new_prediction = model.predict(modified_scaled)[0]
#         new_probability = model.predict_proba(modified_scaled)[0][1]
#
#         # Show comparison
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("Original Prediction:")
#             st.write(f"Credit Risk: {'Good' if prediction == 1 else 'Bad'}")
#             st.write(f"Probability of Good Credit: {probability:.2f}")
#         with col2:
#             st.write("Modified Prediction:")
#             st.write(f"Credit Risk: {'Good' if new_prediction == 1 else 'Bad'}")
#             st.write(f"Probability of Good Credit: {new_probability:.2f}")
#
#         # Show change
#         prob_diff = new_probability - probability
#         if abs(prob_diff) > 0.001:
#             if prob_diff > 0:
#                 st.success(f"The modifications improved the credit score by {prob_diff:.2%}!")
#             else:
#                 st.error(f"The modifications decreased the credit score by {abs(prob_diff):.2%}.")
#         else:
#             st.info("The modifications had minimal impact on the credit score.")
#

# Alternative SHAP visualization - feature importance plot
plt.figure(figsize=(10, 6))
if isinstance(shap_values, list) and len(shap_values) > 1:
    # For binary classification
    shap.summary_plot(shap_values[1], pd.DataFrame(X_test, columns=feature_names), plot_type="dot", show=False)
else:
    # For single array
    shap.summary_plot(shap_values if not isinstance(shap_values, list) else shap_values[0],
                      pd.DataFrame(X_test, columns=feature_names), plot_type="dot", show=False)
st.pyplot(plt.gcf())

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
st.write("Predicted Credit Risk:", "Good" if prediction == 1 else "Bad")
st.write("Probability of Good Credit Risk:", f"{probability:.2f}")

# Add individual SHAP values for the current prediction
# Add individual SHAP values for the current prediction
# Add individual SHAP values for the current prediction
# Add individual SHAP values for the current prediction
if st.button("Explain this prediction"):

     some()