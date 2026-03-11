import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

DATA_FILE = 'hand_data.csv'
MODEL_FILE = 'sign_language_model.p'

# Load data
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please run process_kaggle.py first to gather data!")
    exit(1)

X = data.iloc[:, 1:].values # Coordinates (normalized)
y = data.iloc[:, 0].values  # Labels

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using more estimators for the larger vocabulary (26 letters + 20 words = 46 classes)
model = RandomForestClassifier(n_estimators=200, random_state=42)
print("Training new Random Forest Model...")
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test) * 100
print(f"Model Accuracy on Test Set: {accuracy:.2f}%")

if accuracy < 90.0:
    print("Warning: Accuracy is below 90%. You may want to collect more data using collect.py.")
else:
    print("Excellent! Model accuracy is above 90%.")

# Save model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print(f"Model successfully saved as '{MODEL_FILE}'.")
print("You can now restart your ASL server to use the new predictions.")
