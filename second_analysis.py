import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

# Load the data
train_data_path = 'titanic-train.txt'
train_df = pd.read_csv(train_data_path, delim_whitespace=True, skiprows=1, header=None, 
                       names=['Pclass', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Survived'])

# Add new feature 'FamilySize'
train_df['FamilySize'] = train_df['Siblings/Spouses'] + train_df['Parents/Children']

# Scale features and apply polynomial features
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train = scaler.fit_transform(train_df.drop('Survived', axis=1))
X_train_poly = poly.fit_transform(X_train)  # Apply polynomial transformation
y_train = train_df['Survived']

# Initialize and configure SGDClassifier
model = SGDClassifier(loss='log_loss', max_iter=100, learning_rate='constant', eta0=0.01, random_state=42)

# Train the model
model.fit(X_train_poly, y_train)

# Calculate training accuracy
y_pred = model.predict(X_train_poly)
training_accuracy = accuracy_score(y_train, y_pred)
print(f"Training accuracy: {training_accuracy:.2%}")

# Calculate the log loss for all predictions
probabilities = model.predict_proba(X_train_poly)
total_log_loss = log_loss(y_train, probabilities)
print(f"Total Log Loss: {total_log_loss:.4f}")

# Save model parameters and training accuracy
np.save('model_coefficients.npy', model.coef_)
np.save('model_intercept.npy', model.intercept_)
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)
np.save('training_accuracy.npy', training_accuracy)

# Plotting the probability distribution
plt.figure(figsize=(10, 6))
plt.hist(probabilities[:, 1], bins=50, alpha=0.75)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Function to predict the probability of survival for a hypothetical passenger
def predict_survival(model, scaler, poly, pclass, sex, age, siblings_spouses, parents_children, fare, family_size):
    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],  # 0 for male, 1 for female
        'Age': [age],
        'Siblings/Spouses': [siblings_spouses],
        'Parents/Children': [parents_children],
        'Fare': [fare],
        'FamilySize': [family_size]
    })
    passenger_scaled = scaler.transform(passenger)
    passenger_poly = poly.transform(passenger_scaled)
    survival_probability = model.predict_proba(passenger_poly)[0, 1]
    return survival_probability

# Example usage of the prediction function
my_survival_probability = predict_survival(model, scaler, poly, 1, 1, 25, 0, 0, 50, 0)
print(f"My probability of survival: {my_survival_probability:.2f}")
