import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor 
import pickle
import os
from sklearn.metrics import r2_score


# Paths
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","job_salary_prediction_dataset.csv")
save_path = "model/salary_pred.pkl"
os.makedirs("model", exist_ok=True)


# Load data
df = pd.read_csv(data_path)

# Encode categorical columns
job_title_encoder=LabelEncoder()
education_level_encoder=LabelEncoder()
industry_encoder=LabelEncoder()
company_size_encoder=LabelEncoder()
location_encoder=LabelEncoder()
remote_work_encoder=LabelEncoder()


df['job_title']=job_title_encoder.fit_transform(df['job_title'])
df['education_level']=education_level_encoder.fit_transform(df['education_level'])
df['industry']=industry_encoder.fit_transform(df['industry'])
df['company_size']=company_size_encoder.fit_transform(df['company_size'])
df['location']=location_encoder.fit_transform(df['location'])
df['remote_work']=remote_work_encoder.fit_transform(df['remote_work'])


# Split X and y
X = df.drop("salary",axis=1)
y = df["salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)


print("Training R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))


# Save model + scaler + encoders
with open(save_path, "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "job_title_encoder": job_title_encoder,
        "education_level_encoder": education_level_encoder,
        "industry_encoder": industry_encoder,
        "company_size_encoder": company_size_encoder,
        "location_encoder": location_encoder,
        "remote_work_encoder": remote_work_encoder,
    }, f)

print("Model saved successfully!")