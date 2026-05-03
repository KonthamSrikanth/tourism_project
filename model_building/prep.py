import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Initialize API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from HF using hf:// protocol (NOT the browser URL)
DATASET_PATH = "hf://datasets/SrikanthKontham/tourism_project/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Drop unnecessary identifier columns
df.drop(columns=['CustomerID'], inplace=True)
print("Dropped: 'CustomerID'")

# Handle missing values
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# Target column
target_col = 'ProdTaken'

# Split into X and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv",  index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv",  index=False)
print("Saved locally.")

# Upload to HF dataset space inside 'data/' folder
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"data/{file_path}",
        repo_id="SrikanthKontham/tourism_project",
        repo_type="dataset",
    )
    print(f"Uploaded: {file_path} → data/{file_path}")

print("All files uploaded successfully!")
