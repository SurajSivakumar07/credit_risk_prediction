import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data():
    df = pd.read_csv("../data/german.csv")
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
    for col, mapping in value_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    df.rename(columns={
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
    }, inplace=True)
    return df

def preprocess_data(df):
    df = df.copy()
    df['Risk'] = df['Risk'].replace({1: 1, 2: 0})
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