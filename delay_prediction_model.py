import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the CSV data
df = pd.read_csv("sample_data.csv")

# Encode Weather_Condition separately
weather_enc = LabelEncoder()
df["Weather_Condition"] = weather_enc.fit_transform(df["Weather_Condition"])

# Encode Status separately
status_enc = LabelEncoder()
df["Status"] = status_enc.fit_transform(df["Status"])  # 0 = Delayed, 1 = On-Time

# Define features and target
X = df[["Supplier_Performance", "Past_Delays", "Weather_Condition", "Distance_km"]]
y = df["Status"]

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and label encoders separately
joblib.dump(model, "shipping_model.pkl")
joblib.dump(weather_enc, "weather_label_encoder.pkl")  # Save weather encoder
joblib.dump(status_enc, "status_label_encoder.pkl")  # Save status encoder

print("✅ Model trained and saved successfully!")
print("Weather Encoder Classes:", weather_enc.classes_)  # Debugging
print("Status Encoder Classes:", status_enc.classes_)  # Debugging
