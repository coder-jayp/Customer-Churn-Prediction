import os
import pandas as pd

df = pd.read_csv(r"D:\Customer-Churn-Project\data\processed\churn_processed.csv")

df['TenureGroup'] = pd.cut(
    df['Tenure Months'],
    bins=[0, 12, 24, 48, 100],
    labels=['0-12', '12-24', '24-48', '48+']
)

service_cols = [
    'Phone Service', 'Multiple Lines', 'Internet Service',
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies'
]

df['ServiceCount'] = df[service_cols].sum(axis=1)

monthly_median = df['Monthly Charges'].median()
df['HighSpender'] = (df['Monthly Charges'] > monthly_median).astype(int)

os.makedirs(r"D:\Customer-Churn-Project\data\features", exist_ok=True)
df.to_csv(r"D:\Customer-Churn-Project\data\features\churn_features.csv", index=False)

print("Feature engineering complete. Saved to data/features/churn_features.csv")