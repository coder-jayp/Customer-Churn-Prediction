# 📊 Customer Churn Prediction (GPU-Accelerated ML Pipeline)

A **production-ready machine learning pipeline** for predicting customer churn.  
This project demonstrates **EDA, feature engineering, model training, hyperparameter tuning, model explainability (SHAP), and GPU-accelerated training** using **XGBoost** and **LightGBM**.

## 📂 Dataset
This project uses the [Telco Customer Churn dataset (IBM, Kaggle)]

### Setup
1. Download the dataset from Kaggle.
2. Place the CSV file in `data/raw/`.
3. Run preprocessing and model training pipelines.


## 🚀 Key Features
- 📊 **Exploratory Data Analysis (EDA)** with reusable visualization utilities.  
- ⚙️ **Automated preprocessing** (categorical encoding, scaling, missing values).  
- 🤖 **Multiple ML models**:  
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost (CUDA-accelerated)  
  - LightGBM (GPU-accelerated)  
- 🔍 **Hyperparameter tuning** with GridSearchCV.  
- 📈 **Model benchmarking**: Accuracy, Precision, Recall, F1, ROC-AUC.  
- 🧠 **Explainability** with SHAP + feature importance plots.  
- 💾 **Artifact management**:  
  - Best model → `.pkl`  
  - Results → `.csv`  
  - Visualizations → `.png`  


## 🗂 Project Structure

Customer-Churn-Project/

│── data/                     # Raw and processed datasets
│── models/                   # Saved models + metrics + plots
│── notebooks/                # Jupyter notebooks for experimentation
│── src/
│   ├── data_preprocessing.py # Data cleaning & preprocessing
│   ├── eda_visualizations.py # Reusable EDA/visualization functions
│   ├── model_training.py     # Training pipeline (multi-model + GPU)
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies

# Clone 

git clone https://github.com/coder-jayp/Customer-Churn-Prediction.git

cd Customer-Churn-Project

# Create virtual environment

python -m venv venv

source venv/bin/activate   # (Linux/Mac)

venv\Scripts\activate      # (Windows)

# Install dependencies

pip install -r requirements.txt


🛠 Usage
Run EDA

python -m src.eda_visualizations


Train Models

python -m src.model_training

Outputs:

models/model_results.csv → performance comparison

models/best_model.pkl → serialized best model

models/feature_importance.png → feature importance visualization

models/shap_summary.png → SHAP explainability

🏆 Highlights

✅ End-to-end ML pipeline (ready for production).

✅ GPU-accelerated training (XGBoost & LightGBM).

✅ Model explainability with SHAP.

✅ Clean, modular, and extensible project structure.