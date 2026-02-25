# Credit Risk Analysis & Assessment Tool

[![Python package](https://github.com/arko-saha/credit-risk-analysis/actions/workflows/python-app.yml/badge.svg)](https://github.com/arko-saha/credit-risk-analysis/actions/workflows/python-app.yml)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade machine learning solution for credit risk assessment. This project transforms raw financial data into actionable risk insights, calculating Probability of Default (PD), Risk Classification, and Expected Loss (EL) using an optimized LightGBM model.

## 🚀 Overview

This repository demonstrates an end-to-end MLOps pipeline for credit risk modeling. It transitions from a research-oriented Jupyter Notebook to a modular, testable, and scalable Python library.

### Key Features
- **Production-Ready Architecture**: Modular implementation with clear separation of concerns (data loading, preprocessing, modeling, calculation).
- **Advanced Feature Engineering**: Automatic calculation of critical metrics like Debt-to-Income (DTI) ratio.
- **Handling Class Imbalance**: Integrated SMOTE (Synthetic Minority Over-sampling Technique) to ensure robust performance on default predictions.
- **Interactive Risk Tool**: A CLI-based assessment tool for real-time customer risk evaluation.
- **Enterprise Standards**: Configuration management via Pydantic, automated CI/CD with GitHub Actions, and comprehensive unit testing.

## 🛠 Tech Stack
- **Modeling**: LightGBM, Scikit-learn
- **Data Engineering**: Pandas, NumPy, Imbalanced-learn (SMOTE)
- **Infrastructure**: Pydantic (Settings), Pytest, GitHub Actions
- **Visuals**: Matplotlib, Seaborn

## 📂 Project Structure
```text
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Raw datasets (Git ignored in production)
├── research/             # Original research & exploratory analysis
├── src/credit_risk/      # Core package logic
│   ├── config.py         # Configuration management
│   ├── data_loader.py    # Schema validation & loading
│   ├── preprocessor.py   # Feature engineering & scaling
│   ├── model.py          # Model wrapper & calibration
│   ├── risk_calculator.py# Financial risk logic (PD, EL)
│   └── cli.py            # Assessment entry point
├── tests/                # Unit & integration tests
└── requirements.txt      # Dependency management
```

## ⚙️ Installation & Usage

### 1. Setup
```bash
git clone https://github.com/arko-saha/credit-risk-analysis.git
cd credit-risk-analysis
pip install -r requirements.txt
```

### 2. Run Risk Assessment CLI
The CLI tool performs a full pipeline execution (trains the model on historical data) and then opens an interactive session for new customer assessments.
```bash
$env:PYTHONPATH += ";src" # On Windows (PowerShell)
python src/credit_risk/cli.py
```

### 3. Run Tests
```bash
pytest
```

## 📊 Risk Metrics Explained

- **Probability of Default (PD)**: The likelihood that a borrower will fail to meet their debt obligations.
- **Expected Loss (EL)**: Calculated as `PD * Exposure at Default * (1 - Recovery Rate)`. This helps financial institutions set aside appropriate capital reserves.
- **Risk Classification**: Automated bucketing into Low, Medium, and High risk based on PD thresholds.

## 🧪 CI/CD
Every commit is automatically validated via GitHub Actions, ensuring:
- **Linting**: Consistent code style (flake8).
- **Automated Testing**: Verification of core logic and financial calculations.

---

