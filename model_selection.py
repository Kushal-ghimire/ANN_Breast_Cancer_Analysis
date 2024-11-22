# grid_search.py
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load dataset
df = pd.read_csv('breast_cancer_data.csv')  # Load the prepared dataset from CSV
X = df.drop(columns=['target'])  # Separate features from the target variable
y = df['target']  # Extract the target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets

# Define ANN model
model = MLPClassifier(max_iter=1000)  # Initialize the MLPClassifier with a maximum of 1000 iterations

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],  # Define different hidden layer sizes to try
    'activation': ['relu', 'tanh'],  # Define activation functions to try
    'solver': ['adam', 'sgd'],  # Define solvers to try
    'alpha': [0.0001, 0.001, 0.01]  # Define regularization parameter values to try
}

# Set up GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)  # Set up grid search with 5-fold cross-validation
grid_search.fit(X_train, y_train)  # Fit the grid search to the training data

# Best parameters
print('Best Parameters:', grid_search.best_params_)  # Print the best parameters found by grid search

# Evaluate model
y_pred = grid_search.best_estimator_.predict(X_test)  # Make predictions on the test set
print(classification_report(y_test, y_pred))  # Print the classification report

# ann_model.py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load dataset
df = pd.read_csv('breast_cancer_data.csv')  # Load the prepared dataset from CSV
X = df.drop(columns=['target'])  # Separate features from the target variable
y = df['target']  # Extract the target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets

# Create and train ANN model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000)  # Initialize and configure the MLPClassifier
model.fit(X_train, y_train)  # Fit the model to the training data

# Evaluate model
y_pred = model.predict(X_test)  # Make predictions on the test set
print(classification_report(y_test, y_pred))  # Print the classification report