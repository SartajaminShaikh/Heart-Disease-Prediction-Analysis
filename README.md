# 🫀 CardioInsight – Heart Disease Prediction & Analytics System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

CardioInsight is a *Heart Disease Prediction and Analytics System* built using *Python and Machine Learning*.  
The project analyzes clinical and demographic patient data to predict the *presence or absence of heart disease* and to identify key medical risk factors.

It combines *Exploratory Data Analysis (EDA), **data visualization, and **machine learning models* to provide accurate predictions and meaningful healthcare insights.

---

## 🛠 Technology Stack

- *Programming Language:* Python  
- *Environment:* Jupyter Notebook  

### Libraries
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

### Machine Learning Algorithms
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

## 🚀 Features

### Core Features
- Heart disease prediction (Presence / Absence)
- Cleaned and preprocessed medical dataset
- Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Statistical insights from patient data
- Machine learning model comparison

### Analytical Features
- Age and gender-based risk analysis
- Chest pain type evaluation
- Cholesterol and blood pressure analysis
- Maximum heart rate analysis
- Exercise-induced angina impact
- Thallium stress test analysis
- Correlation heatmaps and visual plots

### Machine Learning Features
- Model training and testing
- Accuracy evaluation
- Confusion matrix
- Classification report (Precision, Recall, F1-score)
- Feature importance analysis

---

## 📦 Project Structure

---

Heart-Disease-Prediction/
│
├── Heart Disease Anlysis.ipynb
├── Heart disease Prediction Analysis.pdf
├── dataset/
│ └── heart_disease.csv
├── images/
│ └── plots/
├── README.md
└── requirements.txt



---

## 📊 Dataset Information

- *Total Records:* 270 patients  
- *Total Features:* 14  
- *Target Variable:* Heart Disease (Presence / Absence)

### Key Attributes
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- ECG Results  
- Maximum Heart Rate  
- Exercise-Induced Angina  
- ST Depression  
- Slope of ST Segment  
- Number of Major Vessels  
- Thallium Stress Test  

---

## 🔍 Exploratory Data Analysis (EDA)

### Steps Performed
- Checked for missing and duplicate values
- Feature-wise distribution analysis
- Correlation analysis using heatmaps
- Comparison between disease and non-disease groups

### Key Insights
- Asymptomatic chest pain is a strong indicator of heart disease
- Higher number of major vessels increases disease risk
- Lower maximum heart rate strongly correlates with disease presence
- Reversible thallium defects are strong predictors
- Older age groups show higher disease prevalence

---

## 🤖 Machine Learning Models

| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline classification |
| Decision Tree | Rule-based prediction |
| Random Forest | Ensemble model with improved accuracy |

### Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Precision  
- Recall  
- F1-Score  

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8 or higher  
- Jupyter Notebook  

### Steps
```bash
git clone https://github.com/your-username/cardioinsight-heart-disease.git
cd cardioinsight-heart-disease
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook
Open Heart Disease Anlysis.ipynb.

📈 Results
Machine learning models successfully predict heart disease

Random Forest achieved the best overall performance

Results align with real-world clinical risk factors

Dataset is suitable for healthcare predictive analytics

🎯 Key Resume Highlights
Performed exploratory data analysis on medical datasets

Built and evaluated machine learning classification models

Implemented end-to-end ML workflow

Visualized healthcare trends using Python libraries

Evaluated models using industry-standard metrics

🔮 Future Enhancements
Deep learning models (ANN)

Web app using Flask or Streamlit

Model deployment

Explainable AI (SHAP / LIME)

Real-time patient input system

Mobile healthcare application

📝 License
This project is licensed under the MIT License.

👤 Author
Sartajamin Musa Shaikh

📞 Contact
For questions or collaboration, please open an issue in the repository.
