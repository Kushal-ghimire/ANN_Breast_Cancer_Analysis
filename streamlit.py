# streamlit.py

import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
df = pd.read_csv('breast_cancer_data.csv')  # Load the prepared dataset from CSV
X = df.drop(columns=['target'])  # Separate features from the target variable
y = df['target']  # Extract the target variable

# Train model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000)  # Initialize and configure the MLPClassifier
model.fit(X, y)  # Fit the model to the entire dataset
joblib.dump(model, 'breast_cancer_model.pkl')  # Save the trained model to a file

# Streamlit app
st.title('Breast Cancer Prediction App')  # Set the title of the Streamlit app
st.write("This app uses a neural network to predict if a tumor is malignant or benign.")  # Add a description to the app

# Input features
user_input = {}  # Initialize a dictionary to store user inputs
for feature in X.columns:  # Loop through each feature
    user_input[feature] = st.number_input(f"{feature}")  # Create a number input field for each feature

# Predict button
if st.button('Predict'):  # When the user clicks the Predict button
    input_data = pd.DataFrame([user_input])  # Convert user inputs to a DataFrame
    model = joblib.load('breast_cancer_model.pkl')  # Load the trained model
    prediction = model.predict(input_data)  # Make a prediction based on user input
    result = 'Malignant' if prediction[0] == 0 else 'Benign'  # Interpret the prediction result
    st.write(f'The predicted tumor type is: {result}')  # Display the prediction result