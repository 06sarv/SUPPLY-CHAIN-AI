import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = r"E:\coding\CODING PROJECTS\PROJECTS\SUPPLY CHAIN AI\supply_chain_data.csv"
data = pd.read_csv(file_path)

# Convert categorical columns to numerical
label_enc = LabelEncoder()
categorical_cols = ["Inspection results", "Supplier name", "Transportation modes", "Routes"]
for col in categorical_cols:
    data[col] = label_enc.fit_transform(data[col])

# Select features for fraud detection
fraud_features = ["Defect rates", "Manufacturing costs", "Inspection results", "Shipping times", 
                  "Shipping costs", "Supplier name", "Transportation modes", "Routes"]
fraud_data = data[fraud_features]

# Select features for demand prediction
demand_features = ["Number of products sold", "Stock levels", "Order quantities", "Lead times", 
                   "Production volumes", "Manufacturing lead time", "Price"]
demand_data = data[demand_features]

# Normalize numerical data
scaler = StandardScaler()
fraud_data_scaled = scaler.fit_transform(fraud_data)
demand_data_scaled = scaler.fit_transform(demand_data)

# Convert back to DataFrame
fraud_data = pd.DataFrame(fraud_data_scaled, columns=fraud_features)
demand_data = pd.DataFrame(demand_data_scaled, columns=demand_features)

# Save processed data
fraud_data.to_csv("fraud_preprocessed.csv", index=False)
demand_data.to_csv("demand_preprocessed.csv", index=False)

print("Preprocessing completed! Preprocessed data saved.")
