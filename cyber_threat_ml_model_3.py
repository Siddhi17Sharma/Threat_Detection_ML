import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the synthetic dataset
df = pd.read_csv('cyber_threat_dataset.csv')

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier with further hyperparameter tuning
rf_model = RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_split=4, min_samples_leaf=1, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test data with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)

print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')
print(f'Random Forest F1 Score: {rf_f1:.2f}')
print(f'Random Forest Precision: {rf_precision:.2f}')
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))

# Train a Support Vector Machine (SVM) with further optimized hyperparameters
svm_model = SVC(kernel='rbf', C=50, gamma='auto', probability=True)
svm_model.fit(X_train, y_train)

# Predict on the test data with SVM
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)

print(f'SVM Accuracy (Non-Adam): {svm_accuracy * 100:.2f}%')
print(f'SVM F1 Score (Non-Adam): {svm_f1:.2f}')
print(f'SVM Precision (Non-Adam): {svm_precision:.2f}')
print('SVM Classification Report (Non-Adam):')
print(classification_report(y_test, y_pred_svm))

# Implementing an optimized Neural Network with Adam optimizer
adam_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

adam_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
adam_model.fit(X_train, y_train, epochs=80, batch_size=128, validation_split=0.2, verbose=1)

# Predict using the Adam-optimized model
y_pred_adam = (adam_model.predict(X_test) > 0.5).astype(int)

# Evaluate Adam-optimized model
adam_accuracy = accuracy_score(y_test, y_pred_adam)
adam_f1 = f1_score(y_test, y_pred_adam)
adam_precision = precision_score(y_test, y_pred_adam)

print(f'Adam Optimized Model Accuracy: {adam_accuracy * 100:.2f}%')
print(f'Adam Optimized Model F1 Score: {adam_f1:.2f}')
print(f'Adam Optimized Model Precision: {adam_precision:.2f}')

# Save the models
import joblib
joblib.dump(rf_model, 'cyber_threat_rf_model.pkl')
joblib.dump(svm_model, 'cyber_threat_svm_model.pkl')
adam_model.save('cyber_threat_adam_model.h5')
print("Models saved as 'cyber_threat_rf_model.pkl', 'cyber_threat_svm_model.pkl', and 'cyber_threat_adam_model.h5'")
