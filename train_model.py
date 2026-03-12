import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Generate Synthetic Data
np.random.seed(42)
num_samples = 5000

# Normal transactions
amount_normal = np.random.normal(loc=50, scale=20, size=num_samples)
amount_normal = np.clip(amount_normal, a_min=1, a_max=None) # Ensure positive
location_options = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
location_normal = np.random.choice(location_options, size=num_samples)
merchant_options = ['Retail', 'Restaurant', 'Electronics', 'Grocery', 'Online']
merchant_normal = np.random.choice(merchant_options, size=num_samples)
device_options = ['Mobile', 'Desktop', 'Tablet']
device_normal = np.random.choice(device_options, p=[0.7, 0.2, 0.1], size=num_samples)
time_normal = np.random.randint(6, 23, size=num_samples) # Mostly daytime (6 AM to 11 PM)

# Fraudulent (Anomalous) transactions (around 5% of data)
num_fraud = int(num_samples * 0.05)
amount_fraud = np.random.uniform(high=5000, low=1000, size=num_fraud) # Unusually high amounts
location_fraud = np.random.choice(['International_A', 'International_B'], size=num_fraud) # Unusual locations
merchant_fraud = np.random.choice(['Luxury', 'Crypto'], size=num_fraud)
device_fraud = np.random.choice(['Desktop', 'Unknown'], size=num_fraud)
time_fraud = np.random.randint(0, 5, size=num_fraud) # Mostly nighttime (Midnight to 5 AM)

# Combine
amount = np.concatenate([amount_normal, amount_fraud])
location = np.concatenate([location_normal, location_fraud])
merchant = np.concatenate([merchant_normal, merchant_fraud])
device = np.concatenate([device_normal, device_fraud])
time = np.concatenate([time_normal, time_fraud])

# Labels (0 for normal, 1 for anomaly - conceptually, though IsolationForest doesn't need labels for training)
labels = np.concatenate([np.zeros(num_samples), np.ones(num_fraud)])

# Create DataFrame
df = pd.DataFrame({
    'amount': amount,
    'location': location,
    'merchant_type': merchant,
    'device_type': device,
    'time_of_transaction': time,
    'is_fraud': labels # Adding just for testing/validation later if needed
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/synthetic_transactions.csv', index=False)
print("Synthetic data generated.")

# 2. Preprocessing Setup
categorical_features = ['location', 'merchant_type', 'device_type']
numerical_features = ['amount', 'time_of_transaction']

# We need to fit the preprocessor on the expected real-world data distribution
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit preprocessor
X = df.drop(columns=['is_fraud'])
X_processed = preprocessor.fit_transform(X)

# 3. Train Isolation Forest
print("Training Isolation Forest Model...")
# contamination is the expected proportion of outliers in the dataset
model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
model.fit(X_processed)

# 4. Save the Model and Preprocessor
print("Saving models...")
joblib.dump(model, 'models/isolation_forest_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("Training complete! Model and pipelines saved.")
