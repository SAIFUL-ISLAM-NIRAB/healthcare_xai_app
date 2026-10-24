# 🏥 Explainable AI (XAI) & LLM Diabetes Risk Predictor

An end-to-end health data pipeline that predicts diabetes risk using Machine Learning, opens the "black box" of the model using SHAP (Explainable AI), and instantly translates complex clinical metrics into a patient-friendly summary using a Large Language Model (LLM).

This project focuses on solving a critical challenge in digital health: **Trust and Interpretability**. Instead of just giving a blind prediction score, it explains _why_ the prediction was made and delivers an empathetic, actionable clinical report.

---

## How the Pipeline Works

1. **Prediction Engine:** A trained **XGBoost Classifier** evaluates patient biometrics and lifestyle indicators to compute the exact percentage risk of diabetes.
2. **Explainability Layer:** **SHAP** dynamically calculates the localized mathematical feature importances for the specific patient and renders a real-time waterfall plot.
3. **Clinical Translation:** The engine extracts the top risk factors from the SHAP values and pushes them into **Qwen-3.8-27B via Groq Cloud API**, which auto-generates an easy-to-read medical breakdown.
4. **User Interface:** A full-stack **Streamlit** dashboard built with integrated interactive clinical tooltip guides for seamless patient assessments.

---

## Setup & Installation

### Prerequisites

Make sure you have **Python 3.8+** installed on your computer.

### 1. Clone the repository & install dependencies

```bash
git clone https://github.com
cd healthcare_xai_app
pip install streamlit pandas xgboost shap groq matplotlib
```

### 2. Download the Dataset

The model is trained on the CDC's behavioral risk factor survey data.

1. Download `diabetes_binary_health_indicators_BRFSS2015.csv` from [Kaggle](https://kaggle.com).
2. Place the `.csv` file directly inside your `healthcare_xai_app` project folder.

### 3. Train and Save the AI Model

Run the training script once to generate your localized serialized model file (`diabetes_model.json`):

```bash
python train_model.py
```

### 4. Configure API Authorization & Launch the App

1. Get a free API Key from the [Groq Console](https://groq.com).
2. Open `app.py` in VS Code, find the line `client = Groq(api_key="YOUR_GROQ_API_KEY")`, and paste your live key string inside the quotes.
3. Launch your local web server:

```bash
streamlit run app.py
```

---

## CDC Features Used for Clinical Evaluation

The model evaluates the following health indicators from the patient sidebar:

- **Biometrics:** Body Mass Index (BMI).
- **Vitals & Medical History:** High Blood Pressure (Hypertension), High Cholesterol (Hypercholesterolemia), History of Stroke, and History of Heart Disease/Attack.
- **Lifestyle Factors:** Nicotine intake (Smoking status) and Physical activity frequency.
- **Demographics:** Age cohorts (grouped into 5-year intervals) and General Health Self-Evaluation (Scale 1-5).

---

_Disclaimer: This repository outlines an engineering prototype meant for research validation and academic benchmarking. It is not intended for real-world clinical deployment or medical diagnosis._
