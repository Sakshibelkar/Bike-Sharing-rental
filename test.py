import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('Dataset.csv')

# Convert '?' to NaN across entire dataframe
df.replace('?', np.nan, inplace=True)

# Identify and clean ALL numeric columns (including target components)
num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'yr', 'mnth', 'hr']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df[num_cols] = df[num_cols].fillna(df[num_cols].median())  # Use median for robustness

# Feature engineering (hr is now safe)
df['is_peak_hour'] = df['hr'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

# Categorical encoding
cat_cols = ['season', 'weathersit', 'weekday', 'workingday', 'holiday']
# Fill any remaining NaNs in cats with mode before encoding
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Verify no NaNs before modeling
print("Final NaN check:\n", df.isna().sum().sum())  # Should be 0

# Rest of your code unchanged...

# Prepare features/target (drop non-features)
X = df.drop(['cnt', 'dteday', 'instant'], axis=1)  # Exclude ID/date/target
y = df['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train best model (LinearRegression from notebook)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.4f}")  # Should be ~1.0 as in notebook[file:1]

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved!")