import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv(r"D:\Customer-Churn-Project\data\raw\telco_customer_churn.csv")

df.ffill(inplace=True)

categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    if col != "CustomerID": 
        df[col] = LabelEncoder().fit_transform(df[col])

num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv(r"D:\Customer-Churn-Project\data\processed\churn_processed.csv", index=False)
print("Data preprocessing complete. Processed file saved to data/processed/churn_processed.csv")
