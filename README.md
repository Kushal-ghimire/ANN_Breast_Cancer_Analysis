# ANN_Breast_Cancer_Analysis

This project involves data preprocessing, feature selection, ANN model building, and creating a Streamlit app for predicting breast cancer.

## Setup

- Clone the repository. (git clone <repository_link>)
- Create a virtual environment and install required packages.(On Windows use venv\Scripts\activate)
- python -m venv venv
- pip install -r requirements.txt


## Usage

- Run the Streamlit app locally: ( streamlit run streamlit.py

## Features

- **Data Preprocessing and Feature Selection:** The project includes loading the breast cancer dataset, handling missing values, and selecting the most relevant features using `SelectKBest`.
- **ANN Model Building and Evaluation:** A neural network model is built using `MLPClassifier` from `sklearn`. The model's hyperparameters are optimized using Grid Search Cross-Validation to improve performance.
- **Streamlit App for User Interaction and Predictions:** An interactive web application using Streamlit allows users to input feature values and get predictions about whether a tumor is malignant or benign.

## Project Structure

- `data_preparation.py`: Script for loading and preparing the dataset.
- `feature_selection.py`: Script for feature selection.
- `model_selection.py`: Script for tuning ANN model hyperparameters using Grid Search,Script for creating and training the ANN model
- `streamlit.py`: Streamlit app for user interaction and predictions.
- `breast_cancer_data.csv`: Preprocessed dataset.
- `README.md`: Documentation of the project.

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- pandas
- joblib


