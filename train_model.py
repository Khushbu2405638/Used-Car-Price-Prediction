import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nDataset loading started...")

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("car_data.csv")

print("\nDataset loaded successfully!")
print("Original shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates().copy()
print("Shape after removing duplicates:", df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum().to_string())

# Select required target column
if "selling_price" not in df.columns:
    raise ValueError("Dataset must contain a 'selling_price' column.")

# Convert target to numeric
df["selling_price"] = pd.to_numeric(
    df["selling_price"], errors="coerce"
)

# Remove rows with missing target values
df = df.dropna(subset=["selling_price"]).copy()

# Separate input features and target
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Fill missing values in input features
for column in categorical_columns:
    X[column] = X[column].fillna("Unknown").astype(str)

for column in numerical_columns:
    X[column] = X[column].fillna(X[column].median())

print("\nMissing values after preprocessing:")
print(X.isnull().sum().sum())

print("\nProcessed dataset shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

print("\nPreprocessing categorical and numerical features...")
print("Training Random Forest model...")

# Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save model pipeline
joblib.dump(model, "models/car_price_model.pkl")

# Save original feature names
joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

print("\nModel saved successfully!")

# Create actual vs predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Graph saved as actual_vs_predicted.png")
print("\nDATA PROCESSING AND MODEL TRAINING COMPLETED!")