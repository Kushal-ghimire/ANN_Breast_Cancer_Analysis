# data_preparation.py
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display basic information
print(df.info())
print(df.head())

# Handle missing values if any
df = df.dropna()

# Save the prepared dataset
df.to_csv('breast_cancer_data.csv', index=False)