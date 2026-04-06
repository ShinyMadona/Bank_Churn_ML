
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_and_train_model():

    # Load both sheets
    customer_df = pd.read_excel('Bank_churn_Customer_info.xlsx')
    account_df = pd.read_excel('Bank_churn_Account_info.xlsx')

    # Clean EstimatedSalary — remove € symbol
    customer_df['EstimatedSalary'] = customer_df['EstimatedSalary'].astype(str).str.replace('€','').str.replace(',','').str.strip()
    customer_df['EstimatedSalary'] = pd.to_numeric(customer_df['EstimatedSalary'], errors='coerce')

    # Clean Balance — remove € symbol
    account_df['Balance'] = account_df['Balance'].astype(str).str.replace('€','').str.replace(',','').str.strip()
    account_df['Balance'] = pd.to_numeric(account_df['Balance'], errors='coerce')

    # Clean HasCrCard and IsActiveMember
    account_df['HasCrCard'] = account_df['HasCrCard'].map({'Yes': 1, 'No': 0})
    account_df['IsActiveMember'] = account_df['IsActiveMember'].map({'Yes': 1, 'No': 0})

    # Drop duplicates
    account_df = account_df.drop_duplicates(subset='CustomerId')
    customer_df = customer_df.drop_duplicates(subset='CustomerId')

    # Merge
    df = pd.merge(customer_df, account_df, on='CustomerId', how='inner')

    # Drop unnecessary columns
    df = df.drop(['CustomerId', 'Surname'], axis=1)

    # Drop rows with missing values
    df = df.dropna()

    # Clean Geography — standardise
    df['Geography'] = df['Geography'].str.strip().str.upper()
    geo_map = {'FRA': 'France', 'FRANCE': 'France',
               'SPAIN': 'Spain', 'ESP': 'Spain',
               'GERMANY': 'Germany', 'GER': 'Germany', 'DEU': 'Germany'}
    df['Geography'] = df['Geography'].map(geo_map).fillna(df['Geography'])

    # Clean Gender
    df['Gender'] = df['Gender'].str.strip().str.title()

    # Drop Tenure_y if duplicate after merge
    if 'Tenure_x' in df.columns:
        df = df.rename(columns={'Tenure_x': 'Tenure'})
    if 'Tenure_y' in df.columns:
        df = df.drop('Tenure_y', axis=1)

    # Encode categorical
    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    df['Geography_encoded'] = le_geo.fit_transform(df['Geography'])
    df['Gender_encoded'] = le_gen.fit_transform(df['Gender'])

    # Store unique values for dropdowns
    geo_classes = list(le_geo.classes_)
    gen_classes = list(le_gen.classes_)

    # Features and target
    features = ['CreditScore', 'Geography_encoded', 'Gender_encoded',
                'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    X = df[features]
    y = df['Exited']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=3.78,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_scaled, y)

    return model, scaler, le_geo, le_gen, geo_classes, gen_classes

# Load model
with st.spinner("Training model... please wait"):
    model, scaler, le_geo, le_gen, geo_classes, gen_classes = load_and_train_model()

# UI
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter customer details to predict whether they are likely to leave the bank.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    geography = st.selectbox("Geography", geo_classes)
    gender = st.selectbox("Gender", gen_classes)
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=500000.0, step=10000.0)

has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

st.divider()

if st.button("Predict Churn", use_container_width=True, type="primary"):

    geo_encoded = le_geo.transform([geography])[0]
    gen_encoded = le_gen.transform([gender])[0]
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_val = 1 if is_active == "Yes" else 0

    input_data = np.array([[
        credit_score, geo_encoded, gen_encoded, age,
        tenure, balance, num_products,
        has_cr_card_val, is_active_val, estimated_salary
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        churn_prob = round(probability[1] * 100, 1)
        st.error(f"⚠️ This customer is likely to CHURN")
        st.metric("Churn Probability", f"{churn_prob}%")
        st.markdown("**Recommended Action:** Contact this customer with a retention offer immediately.")
    else:
        stay_prob = round(probability[0] * 100, 1)
        st.success(f"✅ This customer is likely to STAY")
        st.metric("Retention Probability", f"{stay_prob}%")
        st.markdown("**Status:** Low churn risk. No immediate action required.")

    with st.expander("View Input Summary"):
        summary = pd.DataFrame({
            'Feature': ['Credit Score', 'Geography', 'Gender', 'Age',
                       'Tenure', 'Balance', 'Products', 'Credit Card',
                       'Active Member', 'Salary'],
            'Value': [credit_score, geography, gender, age,
                     tenure, balance, num_products, has_cr_card,
                     is_active, estimated_salary]
        })
        st.dataframe(summary, use_container_width=True)

st.divider()
st.caption("Built by Shiny Madona Arockiasamy | IBM Certified Data Scientist | github.com/shinymadona")

