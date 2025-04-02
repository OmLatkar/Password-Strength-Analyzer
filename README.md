🔒 Password Strength Analyzer
A machine learning-based tool to classify password strength (Weak/Medium/Strong) using RandomForestClassifier and checks against breached passwords (rockyou.txt).

🚀 Features
Strength Prediction: Classifies passwords as Weak, Medium, or Strong.

Breach Check: Flags passwords found in rockyou.txt (common breached passwords).

Combo Detection: Identifies passwords made by combining weak passwords (e.g., password123).

Batch Processing: Analyze multiple passwords from a .txt file.

Risk Scoring: Provides a risk percentage for each password.

⚙️ Setup
Prerequisites
Python 3.8+

Required libraries:

bash
Copy
pip install pandas scikit-learn streamlit joblib
Download rockyou.txt
Download from Kaggle ([Link](https://www.kaggle.com/datasets/wjburns/common-password-list-rockyoutxt))

You need a Kaggle account.

After downloading, place rockyou.txt in the project root folder.

Alternative Download (if Kaggle is unavailable):

bash
Copy
wget https://downloads.skullsecurity.org/passwords/rockyou.txt.bz2
bunzip2 rockyou.txt.bz2
🏃‍♂️ How to Run
Clone the repository

bash
Copy
git clone https://github.com/OmLatkar/Password-Strength-Analyzer.git
cd Password-Strength-Analyzer
Run the Streamlit app

bash
Copy
streamlit run strength_classifier.py
Access the app
Open http://localhost:8501 in your browser.

🖥️ Usage
1. Single Password Check
Enter a password in the input box.

Get instant feedback:

Strength (Weak/Medium/Strong)

Risk factor (%)

Detailed analysis (length, character types, etc.)

2. Batch Processing
Upload a .txt file with one password per line.

Download results as a CSV file.

📁 File Structure
Copy
Password-Strength-Analyzer/
├── strength_classifier.py   # Main Streamlit app
├── rockyou.txt             # Breached passwords list (download separately)
├── data.csv                # Training dataset
├── password_strength_model.pkl  # Pre-trained model
├── passwords.txt           # Example input file
└── results.txt             # Example output
🔧 Troubleshooting
Error: rockyou.txt not found
Ensure the file is in the project folder and named exactly rockyou.txt.

Kaggle Download Issues
If you can’t access Kaggle, use the wget alternative above.

Model Training
If password_strength_model.pkl is missing, the app will train a new model (may take a few minutes).
