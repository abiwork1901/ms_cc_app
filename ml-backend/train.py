# train.py
# -------------------------------------------
# Synthetic training pipeline for Risk Scorer
# -------------------------------------------
# Usage:
#   pip install numpy pandas scikit-learn joblib
#   python train.py --samples 5000 --seed 42 --outdir artifacts
# -------------------------------------------
import argparse
import json
import pathlib
import random
from typing import List, Tuple
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---- Luhn helpers -----------------------------------------------------------

def _luhn_checksum(num: str) -> int:
    digits = [int(d) for d in num]
    odd_sum = sum(digits[-1::-2])
    even_sum = sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
    return (odd_sum + even_sum) % 10


def passes_luhn(num: str) -> bool:
    return _luhn_checksum(num) == 0


def gen_valid_card(length: int) -> str:
    while True:
        base = ''.join(random.choice('0123456789') for _ in range(length - 1))
        cd = (10 - _luhn_checksum(base + '0')) % 10
        card = base + str(cd)
        if passes_luhn(card):
            return card

# ---- Synthetic data ---------------------------------------------------------

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic data for training."""
    data = []
    
    # Common first names and last names
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'James', 'Lisa', 'Robert', 'Mary']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    
    for _ in range(n_samples):
        # Generate a random name with some patterns
        if random.random() < 0.3:  # 30% chance of a suspicious name
            name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 15)))
        else:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Generate a card number
        card_number = ''.join(random.choices('0123456789', k=random.randint(13, 19)))
        
        # Generate a credit limit with some patterns
        if random.random() < 0.2:  # 20% chance of a suspicious amount
            limit = random.randint(100000, 1000000)  # Very high limit
        else:
            limit = random.randint(1000, 50000)  # Normal limit
        
        # Calculate risk score based on name and limit patterns
        risk_score = calculate_risk_score(name, limit)
        
        data.append({
            'name': name,
            'card_number': card_number,
            'limit': limit,
            'risk_score': risk_score
        })
    
    return pd.DataFrame(data)

def calculate_risk_score(name: str, limit: float) -> float:
    """Calculate risk score based on name and limit patterns."""
    score = 50.0  # Base score
    
    # Name-based scoring
    if not re.match(r'^[A-Za-z\s]+$', name):  # Non-alphabetic characters
        score += 20
    if len(name.split()) != 2:  # Not exactly two words
        score += 15
    if len(name) < 5 or len(name) > 30:  # Suspicious length
        score += 10
    if re.search(r'\d', name):  # Contains numbers
        score += 25
    
    # Limit-based scoring
    if limit > 50000:  # High limit
        score += 15
    if limit < 1000:  # Very low limit
        score += 10
    if limit % 1000 == 0:  # Round number
        score += 5
    
    # Normalize score to 0-100 range
    return min(100, max(0, score))

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the data."""
    features = pd.DataFrame()
    
    # Name-based features
    features['name_length'] = df['name'].str.len()
    features['name_word_count'] = df['name'].str.split().str.len()
    features['name_has_numbers'] = df['name'].str.contains(r'\d').astype(int)
    features['name_has_special_chars'] = ~df['name'].str.match(r'^[A-Za-z\s]+$').astype(int)
    
    # Card number features
    features['card_number_length'] = df['card_number'].str.len()
    
    # Limit features
    features['limit'] = df['limit']
    features['limit_is_round'] = (df['limit'] % 1000 == 0).astype(int)
    features['limit_is_high'] = (df['limit'] > 50000).astype(int)
    features['limit_is_low'] = (df['limit'] < 1000).astype(int)
    
    return features

def train_model():
    """Train the risk scoring model."""
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=5000)
    
    print("Extracting features...")
    X = extract_features(df)
    y = df['risk_score']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Save the model
    os.makedirs('artifacts', exist_ok=True)
    model_path = os.path.join('artifacts', 'risk_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names
    feature_names = X.columns.tolist()
    feature_path = os.path.join('artifacts', 'feature_names.txt')
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"Feature names saved to {feature_path}")

# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("artifacts"))
    args = ap.parse_args()

    train_model()


if __name__ == "__main__":
    main()
