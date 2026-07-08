🔮 Customer Churn Prediction

A machine learning web app that predicts whether a bank customer is likely to churn (leave the bank) based on their profile and account information. Built with an Artificial Neural Network (ANN) and deployed using Streamlit.

🚀 Live App: customerchurnprediction-pbdyvv9mszdvauultgp6lu.streamlit.app


📖 Overview

Customer churn — when a customer stops doing business with a company — is a critical metric for banks and subscription-based businesses. Retaining an existing customer is far cheaper than acquiring a new one, so being able to flag at-risk customers early lets a business step in with retention offers before it's too late.

This project trains a deep learning model on the classic bank customer dataset (Churn_Modelling.csv) to estimate the probability that a given customer will exit (churn), and serves the model through an interactive Streamlit interface.

🖼️ App Preview

The app takes in customer details through a simple form and returns a churn probability along with a plain-language verdict:


Geography — customer's country (France, Germany, Spain)
Gender — Male / Female
Age — customer's age
Balance — account balance
Credit Score
Estimated Salary
Tenure — years with the bank
Number of Products — number of bank products used
Has Credit Card — Yes / No
Is Active Member — Yes / No


Based on these inputs, the app outputs something like:

Churn Probability: 0.02
The customer is expected to stay with us.

🧠 How It Works


Data preprocessing — categorical features (Geography, Gender) are encoded using a saved OneHotEncoder and LabelEncoder, while numerical features are scaled with a saved StandardScaler.
Model — a trained Artificial Neural Network (Keras/TensorFlow, saved as model.h5) takes the processed features and outputs a churn probability between 0 and 1.
Interface — Streamlit collects user input, transforms it using the saved encoders/scaler, feeds it to the model, and displays the predicted probability along with a human-readable message.


📁 Repository Structure

Customer_Churn_Prediction/
├── app.py                      # Streamlit web application
├── Churn_Modelling.csv         # Training dataset
├── experiments.ipynb           # Model building & training experiments
├── prediction.ipynb            # Notebook for testing predictions
├── model.h5                    # Trained ANN model
├── scaler.pkl                  # Fitted StandardScaler
├── label_encoder_gender.pkl    # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for Geography
├── requirements.txt            # Python dependencies
└── README.md

🛠️ Tech Stack


Python
TensorFlow / Keras — building and training the ANN
Scikit-learn — preprocessing (encoding & scaling)
Pandas / NumPy — data handling
Streamlit — web app framework & deployment


⚙️ Getting Started Locally

Clone the repository:

bashgit clone https://github.com/Rohit8022/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

Install the dependencies:

bashpip install -r requirements.txt

Run the Streamlit app:

bashstreamlit run app.py

The app will open in your browser at http://localhost:8501.

📊 Dataset

The model is trained on the Churn_Modelling.csv dataset, which contains ~10,000 bank customer records with features such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership status, estimated salary, and whether the customer exited (churned).

📌 Future Improvements


Add model explainability (e.g., SHAP values) to show which features drove a prediction
Experiment with additional models (XGBoost, Random Forest) for comparison
Add batch prediction support via CSV upload


👤 Author

Rohit Prajapati


GitHub: @Rohit8022


📄 License

This project is open source and available for learning and personal use.
