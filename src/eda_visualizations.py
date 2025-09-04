import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_churn_distribution(df: pd.DataFrame, target_col: str = "Churn") -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, palette="viridis")
    plt.title("Churn Distribution")
    plt.show()

def plot_monthly_charges(df: pd.DataFrame, col: str = "MonthlyCharges", target_col: str = "Churn") -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=col, data=df, palette="viridis")
    plt.title("Monthly Charges by Churn Status")
    plt.show()

def plot_tenure_distribution(df: pd.DataFrame, col: str = "tenure") -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title("Distribution of Customer Tenure")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Count")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()