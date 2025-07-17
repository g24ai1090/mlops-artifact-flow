import pickle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load model
with open("model_train.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Scale data if scaling was done during training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Print evaluation
print("Classification Report:\n")
print(classification_report(y, y_pred))
