import pickle
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss

# Load model
with open("model_train.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']

digits = load_digits()
X, y = digits.data, digits.target
X_scaled = scaler.transform(X)  # ✅ use same scaler

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

print("Classification Report:\n")
print(classification_report(y, y_pred))

loss = log_loss(y, y_proba)
print(f"\nLog Loss: {loss:.4f}")

