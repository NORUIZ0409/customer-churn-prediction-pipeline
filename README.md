# 🔮 E-commerce Customer Churn Prediction

An end-to-end machine learning project to **predict customer churn** for a subscription-based e-commerce service.  
This repository contains a full pipeline from data ingestion and cleaning to model training, evaluation, and deployment via a multi-page, interactive web application.

---

## 🎬 Demo
Below is a demonstration of the final **Streamlit web application**, which uses the best-trained model to make live predictions.

> *(A GIF of your `app.py` running would be perfect here. You can use a tool like [ScreenToGIF](https://www.screentogif.com/) or [LICEcap](https://www.cockos.com/licecap/) to record your screen.)*

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [How to Run](#-how-to-run)
- [Results & Evaluation](#-results--evaluation)
- [License](#-license)

---

## 📝 Project Overview
The primary goal of this project is to proactively identify customers who are at a high risk of **churning** (canceling their subscription).  
By leveraging customer data, we build a **classification model** that provides a **probability of churn for each customer**.  

This allows the business to take targeted actions, such as offering discounts or support, to retain valuable customers.

The project is built as a **modular pipeline**, ensuring that each step of the machine learning lifecycle is separated and maintainable. It uses **MLflow** for robust experiment tracking and a comprehensive **Streamlit suite** to provide a user-friendly interface for non-technical users to interact with the final model and explore the data.

---

## ✨ Features
- **End-to-End Modular Pipeline**: Distinct, reusable scripts for each stage (data_ingestion, feature_engineering, model_training, etc.).
- **Data Cleaning & Preprocessing**: Handles missing values, data type conversions, and prepares the data for modeling.
- **Advanced Feature Engineering**: Creates new, insightful features (e.g., tenure groups, service counts).
- **Multi-Model Training**: Trains and compares Logistic Regression, Random Forest, and XGBoost to find the best performer.
- **Experiment Tracking with MLflow**: Logs parameters, metrics, models, and artifacts for every run.
- **Model Evaluation**: Generates performance reports including confusion matrix and ROC curve.
- **Multi-Page Interactive Web UI** (Streamlit):
  - 📈 Real-time prediction dashboard  
  - 📂 Batch prediction for CSV files  
  - 🔬 Model performance analysis  
  - 📊 Interactive data explorer  

---

## 🛠️ Tech Stack & Libraries
**Programming Language**: 🐍 Python 3.11+  

**Core Libraries**  
- Pandas → Data manipulation & analysis  
- Scikit-learn → Preprocessing, modeling, and evaluation  
- XGBoost → Gradient boosting model  

**Experiment Tracking**  
- MLflow → Logging experiments, parameters, metrics, and models  

**Web Application & Visualization**  
- Streamlit → Interactive user interface  
- Plotly & Plotly Express → Dynamic, interactive charts  

**Utilities**  
- PyYAML → Configuration management  

---

## 📂 Project Structure
```bash
├── config/
│   └── config.yaml                 # Configuration file for paths, parameters, etc.
├── data/
│   ├── raw/
│   │   └── ecommerce_churn_data.csv  # Raw dataset
│   └── processed/                    # Cleaned & processed data (train/test sets)
├── reports/                          # Evaluation artifacts like plots
├── saved_models/                     # Trained model and preprocessor files (.pkl)
├── src/                              # Source code for the pipeline
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── pipeline.py                   # Run the entire training pipeline
│   └── utils.py                      # Utility functions
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
