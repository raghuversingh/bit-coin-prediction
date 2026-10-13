import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(page_title="Bitcoin Predictor", layout="wide")

# --- SESSION STATE FOR AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- AUTHENTICATION SECTION ---
def login_signup():
    st.title("Welcome to Bitcoin Predictor")
    auth_mode = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if auth_mode == "Sign Up":
        if st.button("Create Account"):
            st.success("Account created successfully! Please login.")
    
    elif auth_mode == "Login":
        if st.button("Login"):
            if email and password: # Simple check for demo purposes
                st.session_state['logged_in'] = True
                st.rerun()
            else:     
                st.error("Please enter valid credentials")

# --- BACKEND LOGIC (from your .ipynb) ---
def load_and_train_model():
    try:
        df = pd.read_csv('bitcoin.csv')
        
        # Feature Engineering (as per notebook logic)
        df['open-close'] = df['Open'] - df['Close']
        df['low-high'] = df['Low'] - df['High']
        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        features = df[['open-close', 'low-high', 'Close']]
        target = df['target']
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, target, test_size=0.1, random_state=2022)
        
        model = XGBClassifier()
        model.fit(X_train, Y_train)
        
        return df, model, scaler, X_test, Y_test
    except FileNotFoundError:
        st.error("Dataset 'bitcoin.csv' not found. Please upload it to the project folder.")
        return None, None, None, None, None

# --- MAIN APP INTERFACE ---
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard & Prediction", "Visualizations", "Logout"])
    
    if page == "Logout":
        st.session_state['logged_in'] = False
        st.rerun()

    df, model, scaler, X_test, Y_test = load_and_train_model()
    
    if df is not None:
        if page == "Dashboard & Prediction":
            st.title("📈 Bitcoin Prediction Dashboard")
            st.write("### Dataset Preview")
            st.dataframe(df.head())

            st.write("### Make a Prediction")
            col1, col2, col3 = st.columns(3)
            with col1: open_p = st.number_input("Open Price", value=float(df['Open'].iloc[-1]))
            with col2: high_p = st.number_input("High Price", value=float(df['High'].iloc[-1]))
            with col3: low_p = st.number_input("Low Price", value=float(df['Low'].iloc[-1]))
            
            close_p = st.number_input("Current Close Price", value=float(df['Close'].iloc[-1]))

            if st.button("Predict Movement"):
                # Calculate features same as training
                oc = open_p - close_p
                lh = low_p - high_p
                input_data = scaler.transform([[oc, lh, close_p]])
                prediction = model.predict(input_data)
                
                if prediction[0] == 1:
                    st.success("The model predicts the price will GO UP tomorrow! 🚀")
                else:
                    st.warning("The model predicts the price will GO DOWN tomorrow! 📉")

        elif page == "Visualizations":
            st.title("📊 Analysis & Charts")
            
            st.write("### Close Price Trend")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Close'])
            ax.set_title('Bitcoin Close Price')
            st.pyplot(fig)

            st.write("### Distribution of Features")
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
            sns.distplot(df['open-close'], ax=ax2[0])
            sns.distplot(df['low-high'], ax=ax2[1])
            st.pyplot(fig2)

# --- EXECUTION ---
if st.session_state['logged_in']:
    main_app()
else:
    login_signup()
    




