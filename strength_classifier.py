import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import string
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
import time

# Load rockyou passwords into a set for fast lookup
@st.cache_resource
def load_rockyou(filepath):
    rockyou_passwords = set()
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                password = line.strip()
                if password:  # skip empty lines
                    rockyou_passwords.add(password)
        st.success(f"Loaded {len(rockyou_passwords)} passwords from rockyou.txt")
    except Exception as e:
        st.error(f"Error loading rockyou.txt: {e}")
    return rockyou_passwords

# Check if password is combination of 2-3 rockyou passwords
def is_rockyou_combo(password, rockyou_set, min_len=3):
    password = str(password)
    # Check all possible 2-splits
    for i in range(min_len, len(password)-min_len+1):
        part1 = password[:i]
        part2 = password[i:]
        if part1 in rockyou_set and part2 in rockyou_set:
            return True
    
    # Check all possible 3-splits
    for i in range(min_len, len(password)-min_len*2+1):
        for j in range(i+min_len, len(password)-min_len+1):
            part1 = password[:i]
            part2 = password[i:j]
            part3 = password[j:]
            if part1 in rockyou_set and part2 in rockyou_set and part3 in rockyou_set:
                return True
    return False

# Feature Engineering Functions
def count_uppercase(password):
    password = str(password)
    return sum(1 for char in password if char.isupper())

def count_lowercase(password):
    password = str(password)
    return sum(1 for char in password if char.islower())

def count_numbers(password):
    password = str(password)
    return sum(1 for char in password if char.isdigit())

def count_special_chars(password):
    password = str(password)
    special_chars = string.punctuation
    return sum(1 for char in password if char in special_chars)

def count_repeated_chars(password):
    password = str(password)
    repeats = re.findall(r'(.)\1+', password)
    return len(repeats)

def get_password_length(password):
    password = str(password)
    return len(password)

def extract_features(df, rockyou_set):
    df['password'] = df['password'].astype(str)
    df['length'] = df['password'].apply(get_password_length)
    df['uppercase'] = df['password'].apply(count_uppercase)
    df['lowercase'] = df['password'].apply(count_lowercase)
    df['numbers'] = df['password'].apply(count_numbers)
    df['special_chars'] = df['password'].apply(count_special_chars)
    df['repeated_chars'] = df['password'].apply(count_repeated_chars)
    df['has_upper'] = df['uppercase'] > 0
    df['has_lower'] = df['lowercase'] > 0
    df['has_number'] = df['numbers'] > 0
    df['has_special'] = df['special_chars'] > 0
    
    # New features
    df['in_rockyou'] = df['password'].apply(lambda x: x in rockyou_set)
    df['is_rockyou_combo'] = df['password'].apply(lambda x: is_rockyou_combo(x, rockyou_set))
    
    return df

@st.cache_data
def load_data(rockyou_set):
    try:
        file_path = Path(r'C:\Users\omlat\Downloads\archive (3)\data.csv')
        df = pd.read_csv(file_path, 
                        header=None,
                        names=['password', 'strength'],
                        skiprows=1,
                        dtype={'password': str, 'strength': int},
                        on_bad_lines='skip')
        
        # Modify strength based on rockyou rules
        df['strength'] = df.apply(lambda row: 
                                0 if row['password'] in rockyou_set 
                                else (min(row['strength'], 1) if is_rockyou_combo(row['password'], rockyou_set) 
                                      else row['strength']), axis=1)
        
        df['strength'] = df['strength'].clip(0, 2)
        st.success("Successfully loaded and modified dataset")
        return df
    except Exception as e:
        st.error(f"Error loading local file: {e}")
        st.warning("Using fallback minimal dataset")
        return pd.DataFrame({
            'password': ['123456', 'password', 'Password123', 'StrongPass!2023'],
            'strength': [0, 0, 1, 2]
        })

@st.cache_resource
def train_model(rockyou_set):
    df = load_data(rockyou_set)
    
    df = df.dropna()
    df = extract_features(df, rockyou_set)
    
    X = df.drop(['password', 'strength'], axis=1)
    y = df['strength']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_strength(password, model, rockyou_set):
    features = {'password': [str(password)]}
    df = pd.DataFrame(features)
    df = extract_features(df, rockyou_set)
    features_df = df.drop('password', axis=1)
    prediction = model.predict(features_df)[0]
    strength_map = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    probabilities = model.predict_proba(features_df)[0]
    risk_factor = probabilities[0] * 100 + probabilities[1] * 50
    return strength_map[prediction], risk_factor, df.iloc[0]

def batch_predict(uploaded_file, model, rockyou_set):
    results = []
    try:
        for line in uploaded_file:
            password = line.decode('utf-8').strip()
            if password:
                strength, risk_factor, features = predict_strength(password, model, rockyou_set)
                results.append({
                    'Password': password,
                    'Strength': strength,
                    'Risk Factor': f"{risk_factor:.1f}%",
                    'In RockYou': 'Yes' if features['in_rockyou'] else 'No',
                    'Is RockYou Combo': 'Yes' if features['is_rockyou_combo'] else 'No',
                    'Length': features['length'],
                    'Uppercase': features['uppercase'],
                    'Lowercase': features['lowercase'],
                    'Numbers': features['numbers'],
                    'Special Chars': features['special_chars']
                })
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error during batch processing: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Password Strength Classifier", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    .weak-password {
        color: red;
        font-weight: bold;
    }
    .medium-password {
        color: orange;
        font-weight: bold;
    }
    .strong-password {
        color: green;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔒 Password Strength Classifier")
    st.markdown("Analyze the strength of your passwords using machine learning")
    
    # Load rockyou passwords
    rockyou_path = Path('rockyou.txt')  # Update path if needed
    rockyou_set = load_rockyou(rockyou_path)
    
    # Load or train model
    try:
        model = joblib.load('password_strength_model.pkl')
        st.success("Loaded pre-trained model")
    except:
        st.warning("No model found. Training new model...")
        with st.spinner('Training model... This may take a few minutes'):
            model = train_model(rockyou_set)
            joblib.dump(model, 'password_strength_model.pkl')
            st.success("Model trained and saved!")
    
    # Main interface
    option = st.radio("Select an option:", 
                     ("Check a single password", "Process a file with multiple passwords"),
                     horizontal=True)
    
    st.markdown("---")
    
    if option == "Check a single password":
        password = st.text_input("Enter a password to analyze:", type="password")
        
        if password:
            with st.spinner('Analyzing password...'):
                time.sleep(0.5)  # Simulate processing time
                strength, risk_factor, features = predict_strength(password, model, rockyou_set)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Password Strength", strength)
                
                with col2:
                    st.metric("Risk Factor", f"{risk_factor:.1f}%")
                
                with col3:
                    st.metric("Password Length", features['length'])
                
                # Strength indicator
                if strength == 'Weak':
                    st.error("⚠️ This password is weak and easily guessable")
                    st.progress(risk_factor/100)
                elif strength == 'Medium':
                    st.warning("⚠️ This password is moderately strong but could be improved")
                    st.progress(risk_factor/100)
                else:
                    st.success("✅ This is a strong password!")
                    st.progress(risk_factor/100)
                
                # Additional details
                with st.expander("Detailed Analysis"):
                    st.write(f"**Password:** `{password}`")
                    st.write(f"**Contains uppercase letters:** {'Yes' if features['has_upper'] else 'No'} ({features['uppercase']} characters)")
                    st.write(f"**Contains lowercase letters:** {'Yes' if features['has_lower'] else 'No'} ({features['lowercase']} characters)")
                    st.write(f"**Contains numbers:** {'Yes' if features['has_number'] else 'No'} ({features['numbers']} characters)")
                    st.write(f"**Contains special characters:** {'Yes' if features['has_special'] else 'No'} ({features['special_chars']} characters)")
                    st.write(f"**Repeated characters:** {features['repeated_chars']}")
                    
                    if features['in_rockyou']:
                        st.error("🚨 This password appears in the rockyou.txt breach database")
                    elif features['is_rockyou_combo']:
                        st.error("🚨 This password is a combination of known weak passwords")
                
                # Recommendations
                with st.expander("Recommendations for stronger passwords"):
                    st.markdown("""
                    - Use at least 12 characters
                    - Combine uppercase, lowercase, numbers and special characters
                    - Avoid common words or phrases
                    - Don't use personal information
                    - Consider using a passphrase (e.g., "CorrectHorseBatteryStaple")
                    - Use a password manager to generate and store strong passwords
                    """)
    
    else:  # Process file
        st.subheader("Upload a text file with passwords")
        st.markdown("The file should contain one password per line.")
        
        uploaded_file = st.file_uploader("Choose a file", type=['txt'])
        
        if uploaded_file is not None:
            with st.spinner('Processing passwords...'):
                results_df = batch_predict(uploaded_file, model, rockyou_set)
                
                if results_df is not None:
                    st.success(f"Processed {len(results_df)} passwords")
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        weak_count = len(results_df[results_df['Strength'] == 'Weak'])
                        st.metric("Weak Passwords", weak_count)
                    
                    with col2:
                        medium_count = len(results_df[results_df['Strength'] == 'Medium'])
                        st.metric("Medium Passwords", medium_count)
                    
                    with col3:
                        strong_count = len(results_df[results_df['Strength'] == 'Strong'])
                        st.metric("Strong Passwords", strong_count)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(results_df.style.applymap(
                        lambda x: 'color: red' if x == 'Weak' else (
                            'color: orange' if x == 'Medium' else 'color: green'),
                        subset=['Strength']
                    ))
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name='password_analysis_results.csv',
                        mime='text/csv'
                    )
                    
                    # Show weak passwords separately
                    weak_passwords = results_df[results_df['Strength'] == 'Weak']
                    if not weak_passwords.empty:
                        st.subheader("⚠️ Weak Passwords Detected")
                        st.dataframe(weak_passwords)

if __name__ == "__main__":
    main()