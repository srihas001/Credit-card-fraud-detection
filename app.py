import streamlit as st
import pickle
import pandas as pd

st.title("Credit Card Fraud Detection")

# Model selection
model_choice = st.selectbox(
    "Select a Model:",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

# Load models once
models = {
    "Logistic Regression": pickle.load(open("logistic_model.pkl", "rb")),
    "Random Forest": pickle.load(open("randomforest.pkl", "rb")),
    "XGBoost": pickle.load(open("xgboost_model.pkl", "rb"))
}
model = models[model_choice]

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with transaction data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Drop target column if exists
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df

    # Prediction
    predictions = model.predict(X)

    df["Prediction"] = predictions
    df["Prediction"] = df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

    st.write("Results with Predictions:")
    st.dataframe(df)

    # Download option
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv")
