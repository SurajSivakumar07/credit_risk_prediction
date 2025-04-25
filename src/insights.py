import streamlit as st
import pandas as pd
import numpy as np
from visualizations import plot_shap_analysis

def display_insights(model, X, feature_names, explainer):
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # SHAP analysis
    st.subheader("In-Depth SHAP Analysis")
    plot_shap_analysis(X, feature_names, explainer)

    # Key insights
    st.subheader("Key Insights")
    top_features = feature_importance['Feature'].head(5).tolist()
    top_features_text = ", ".join(top_features[:-1]) + f", and {top_features[-1]}"

    st.write(f"""
    Based on our analysis, the most significant factors influencing credit risk in this dataset are {top_features_text}. Let's explore what these mean for credit evaluation:
    """)

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

    for feature in top_features[:3]:
        if feature in feature_insights:
            st.markdown(feature_insights[feature])
        else:
            st.write(f"**{feature}**: This feature shows significant impact on credit risk assessment.")

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