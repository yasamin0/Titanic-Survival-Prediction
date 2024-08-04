import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the test data
test_data_path = 'titanic-test.txt'
test_df = pd.read_csv(test_data_path, delim_whitespace=True, header=None, 
                      names=['Pclass', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Survived'])

# Load model parameters and scaler
model_coefficients = np.load('model_coefficients.npy')
model_intercept = np.load('model_intercept.npy')
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

# Recreate the scaler with training parameters
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Prepare test data
X_test = scaler.transform(test_df.drop('Survived', axis=1))
y_test = test_df['Survived']

# Define function to predict using loaded parameters
def predict(X):
    z = np.dot(X, model_coefficients.T) + model_intercept
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)

# Load training accuracy
training_accuracy = np.load('training_accuracy.npy')

# Make predictions on test set
y_pred_test = predict(X_test)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test accuracy: {test_accuracy:.2%}")

# Determine if the model is overfitting or underfitting
if training_accuracy > test_accuracy + 0.05:  # Check if the model is overfitting
    print("The model might be overfitting the training set.")
elif training_accuracy < 0.7 and test_accuracy < 0.7:  # Assume 0.7 as a threshold for poor performance
    print("The model might be underfitting the training set.")
else:
    print("The model is neither significantly overfitting nor underfitting.")

# Suggestions to increase the performance of the model:
print("1. Adjust the complexity of the model by adding/removing features or using polynomial features.")
print("2. Use regularization techniques to reduce overfitting.")
print("3. Optimize hyperparameters such as the learning rate and number of iterations.")
print("4. Increase the quality and quantity of data.")
print("5. Try different algorithms suited for the problem type and size.")
