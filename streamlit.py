# app.py
import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
df = pd.read_csv('breast_cancer_data.csv')
X = df.drop(columns=['target'])
y = df['target']

# Train model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000)
model.fit(X, y)
joblib.dump(model, 'breast_cancer_model.pkl')

# Streamlit app
st.title('Breast Cancer Prediction App')
st.write("This app uses a neural network to predict if a tumor is malignant or benign.")

# Input features
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"{feature}")

# Predict button
if st.button('Predict'):
    input_data = pd.DataFrame([user_input])
    model = joblib.load('breast_cancer_model.pkl')
    prediction = model.predict(input_data)
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    st.write(f'The predicted tumor type is: {result}')