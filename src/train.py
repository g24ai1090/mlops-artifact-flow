import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def train_model(X, y, config):
    model = LogisticRegression(C=config['C'], solver=config['solver'], max_iter=config['max_iter'])
    model.fit(X, y)
    return model

def main():
    config = load_config('config/config.json')
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_model(X_scaled, y, config)

    # Save both model and scaler
    with open('model_train.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

if __name__ == '__main__':
    main()
