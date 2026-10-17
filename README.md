# Salary Predictor

A machine learning project that explores salary patterns, compares seven regression models, and provides salary estimates through a Flask web application.

## Overview

The project combines exploratory data analysis, data preprocessing, model evaluation, and a web interface. Users enter their professional background and workplace details to generate an estimate.

## Features

- Nine inputs covering experience, skills, education, and workplace details
- Dropdown options loaded from the saved encoders
- XGBoost predictions through a Flask application
- Responsive form and results pages
- Jupyter notebook with EDA and model comparisons

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Flask, HTML, and CSS.

## Dataset and Analysis

The notebook contains 250,000 records with nine input features and salary as the target.

Inputs:
- Job title
- Experience in years
- Education level
- Number of skills
- Industry
- Company size
- Location
- Remote work
- Number of certifications

The saved analysis reports no missing values or duplicate rows. Visualizations explore salary distributions, experience, education, industry, and numeric correlations.

## Model Comparison

The notebook uses a 70% training and 30% test split with `random_state=42`.

| Model | Test R² | MAE |
|---|---:|---:|
| Linear Regression | 0.4576 | 21,708.66 |
| Decision Tree | 0.9280 | 7,771.00 |
| Random Forest | 0.9673 | 5,342.18 |
| K-Nearest Neighbors | 0.6579 | 17,545.71 |
| AdaBoost | 0.7452 | 15,391.38 |
| Gradient Boosting | 0.8378 | 12,168.89 |
| XGBoost | **0.9752** | **4,682.02** |

Results above are taken from the notebook’s saved outputs, not a new evaluation. MAE is expressed in the dataset’s salary units; R² is not classification accuracy.

The notebook evaluates XGBoost with `n_estimators=50` and `random_state=1`. The separate app training script uses `XGBRegressor()` defaults, so these scores should not be treated as verified scores for the saved app model.

## Project Files

| Path | Purpose |
|---|---|
| `app.py` | Flask routes and salary prediction |
| `train_test.py` | Model training and saving preprocessing objects |
| `ML_Project.ipynb` | EDA and regression model comparisons |
| `data/` | Training dataset |
| `model/salary_pred.pkl` | Saved model, scaler, and encoders |
| `templates/` | Form and results pages |
| `requirements.txt` | App and training dependencies |

## Run Locally

Use Python 3.11.

1. Clone the repository:

   ```bash
   git clone https://github.com/NabeelOmer/Salary-Predictor.git
   cd Salary-Predictor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies and start the app:

   ```bash
   python -m pip install -r requirements.txt
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

The saved model is included, so retraining is not required to try the app.

## Explore the Notebook

Install the additional analysis tools:

```bash
python -m pip install matplotlib seaborn jupyter
jupyter notebook ML_Project.ipynb
```

## Limitations

This is a portfolio and learning project, not a live salary benchmark. Results depend on the training dataset and do not guarantee actual compensation.

The app currently displays a rupee symbol. The dataset’s source, currency, and salary period need confirmation before interpreting predictions as real compensation figures.

## Author

**Mohammed Nabeel P U**

[LinkedIn](https://www.linkedin.com/in/mohammed-nabeel-p-u)
