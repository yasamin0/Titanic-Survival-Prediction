import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

# Load the data
train_data_path = 'titanic-train.txt'
train_df = pd.read_csv(train_data_path, delim_whitespace=True, skiprows=1, header=None, 
                       names=['Pclass', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Survived'])

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop('Survived', axis=1))
y_train = train_df['Survived']

# Initialize SGDClassifier
model = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='constant', eta0=0.01, tol=None, random_state=42, verbose=0, warm_start=True)

# Manually train the model to capture loss curve
loss_curve = []
n_epochs = 50  # Number of passes over the data
for epoch in range(n_epochs):
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_pred_proba)
    loss_curve.append(loss)
    print(f'Epoch {epoch + 1}, Loss: {loss}')

# Plotting the training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), loss_curve, marker='o')
plt.title('Training Curve (Iterations vs. Loss)')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.grid(True)
plt.show()

# Calculate training accuracy
y_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_pred)
print(f"Training accuracy: {training_accuracy:.2%}")

# Save training accuracy, model parameters, and scaler
np.save('training_accuracy.npy', training_accuracy)  # Save training accuracy
np.save('model_coefficients.npy', model.coef_)       # Save model coefficients
np.save('model_intercept.npy', model.intercept_)     # Save model intercept
np.save('scaler_mean.npy', scaler.mean_)             # Save the means used by the scaler
np.save('scaler_scale.npy', scaler.scale_)           # Save the scales used by the scaler

# Feature influence
print("Feature coefficients:", model.coef_[0])

# Predict the probability of survival for a hypothetical passenger
def predict_survival(model, scaler, pclass, sex, age, siblings_spouses, parents_children, fare):
    passenger = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],  # 0 for male, 1 for female
        'Age': [age],
        'Siblings/Spouses': [siblings_spouses],
        'Parents/Children': [parents_children],
        'Fare': [fare]
    })
    passenger_scaled = scaler.transform(passenger)
    survival_probability = model.predict_proba(passenger_scaled)[0, 1]
    return survival_probability

my_survival_probability = predict_survival(model, scaler, 1, 1, 25, 0, 0, 50)
print(f"My probability of survival: {my_survival_probability:.2f}")

# Scatter plot showing the distribution of the two most influential features (assuming 'Age' and 'Fare')
plt.figure(figsize=(10, 6))
plt.scatter(train_df['Age'], train_df['Fare'], c=y_train, cmap='bwr', alpha=0.5)
plt.title('Scatter Plot by Age and Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.colorbar(label='Survived')
plt.show()
