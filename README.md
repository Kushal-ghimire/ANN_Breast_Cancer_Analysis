# ANN_Breast_Cancer_Analysis

This project involves data preprocessing, feature selection, ANN model building, and creating a Streamlit app for predicting breast cancer.

## Setup

- Create venv
  **python -m venv Name*
- Activate environment
  **source Name/Scripts/activate*
- Install requirements.txt
  **pip install -r requirements.txt*
- Run streamlit app
  **streamlit run streamlit.py*

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- pandas
- joblib
  
## Usage

- Run the Streamlit app locally: ( streamlit run streamlit.py

## Dataset

- The dataset is the Breast Cancer dataset provided by scikit-learn, containing the following:

**30 features** such as mean radius, texture, perimeter, area, etc.*
- 2 target classes:
   - 0: Malignant (cancerous)
   - 1: Benign (non-cancerous)

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

## Output
![Output](https://github.com/user-attachments/assets/08125357-2fcb-4add-b16d-1dd9d993440e)


