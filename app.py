import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import shap
from groq import Groq
import matplotlib.pyplot as plt

# ==========================================
# 1. APPLICATION PAGE SETUP
# ==========================================

# Configure layout settings for the Streamlit web dashboard
st.set_page_config(page_title="XAI Healthcare Predictor", layout="wide")

# Render primary dashboard headers and brief subtext
st.title("Explainable AI (XAI) & LLM Diabetes Risk Predictor")
st.write("This pipeline predicts diabetes risk and explains the clinical reasoning using SHAP and Qwen LLM.")

# ==========================================
# 2. MACHINE LEARNING RESOURCES LOADER
# ==========================================

# Cache resources globally so the model doesn't re-load on every webpage toggle


@st.cache_resource
def load_resources():
    # Instantiate an empty XGBoost classifier structure
    model = XGBClassifier()
    # Serialize and load the pre-trained JSON model from the directory
    model.load_model('diabetes_model.json')

    # Declare explicit feature names matching the exact schema of the CDC dataset
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
        'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    return model, feature_names


# Unpack the trained model and feature maps into global state Variables
model, feature_names = load_resources()

# ==========================================
# 3. SIDEBAR PATIENT METRICS INTERFACE
# ==========================================

# Render the sidebar header outside the function loop to avoid duplication bugs
st.sidebar.header("Patient Clinical Metrics")


def get_user_input():
    # Render dropdown selector for Hypertension status
    high_bp = st.sidebar.selectbox(
        "High Blood Pressure?",
        ["No", "Yes"],
        help="Select 'Yes' if the patient has hypertension or blood pressure regularly above 130/80 mmHg."
    )

    # Render dropdown selector for Hypercholesterolemia status
    high_chol = st.sidebar.selectbox(
        "High Cholesterol?",
        ["No", "Yes"],
        help="Select 'Yes' if the patient has been diagnosed with hypercholesterolemia (high cholesterol levels)."
    )

    # Render localized interactive slider for Body Mass Index bounds
    bmi = st.sidebar.slider(
        "Body Mass Index (BMI)",
        10.0, 50.0, 25.0,
        help="""
        **BMI Scale Meanings:**
        * **10.0 - 18.4:** Underweight (Malnutrition risk)
        * **18.5 - 24.9:** Normal Weight (Ideal healthy range)
        * **25.0 - 29.9:** Overweight (Pre-obesity stage)
        * **30.0 - 39.9:** Obese (High metabolic/diabetes risk)
        * **40.0 - 50.0:** Severely Obese (Morbid obesity - extreme clinical risk)
        """
    )

    # Render dropdown selector for Lifelong Nicotine intake indicators
    smoker = st.sidebar.selectbox(
        "Smoked at least 100 cigarettes?",
        ["No", "Yes"],
        help="Select 'Yes' if the patient has smoked 100+ cigarettes (approx. 5 packs) in their entire lifetime."
    )

    # Render dropdown selector for Cardiovascular events history
    heart_disease = st.sidebar.selectbox(
        "History of Heart Disease/Attack?",
        ["No", "Yes"],
        help="Select 'Yes' if the patient has a known history of coronary artery disease or myocardial infarction (heart attack)."
    )

    # Render dropdown selector for Recent Physical Exercise indicators
    phys_act = st.sidebar.selectbox(
        "Physical Activity in past 30 days?",
        ["Yes", "No"],
        help="Select 'Yes' if the patient engaged in any physical exercise outside of regular job duties in the past month."
    )

    # Render localized interactive slider for General Health condition metrics
    gen_hlth = st.sidebar.slider(
        "General Health Rating",
        1, 5, 3,
        help="""
        **General Health Self-Evaluation Scale:**
        * **1 (Excellent):** No physical ailments or clinical symptoms
        * **2 (Very Good):** Minor/negligible health concerns
        * **3 (Good):** Average health status
        * **4 (Fair):** Noticeable physical frailty or chronic fatigue
        * **5 (Poor):** Severely compromised health status/low immunity
        """
    )

    # Render localized interactive slider for Demographic age cohort arrays
    age = st.sidebar.slider(
        "Age Category (1-13 Scale)",
        1, 13, 6,
        help="""
        **Age Category Scale (5-Year Cohorts):**
        * **1:** 18 - 24 years (Young Adult)
        * **2:** 25 - 29 years
        * **3:** 30 - 34 years
        * **4:** 35 - 39 years
        * **5:** 40 - 44 years (Middle-Aged)
        * **6:** 45 - 49 years
        * **7:** 50 - 54 years
        * **8:** 55 - 59 years
        * **9:** 60 - 64 years (Senior Citizen)
        * **10:** 65 - 69 years (Elderly)
        * **11:** 70 - 74 years
        * **12:** 75 - 79 years
        * **13:** 80 years or older (Geriatric)
        """
    )

    # Map the captured raw string/float values into the structured dictionary schema
    data = {
        'HighBP': 1 if high_bp == "Yes" else 0,
        'HighChol': 1 if high_chol == "Yes" else 0,
        'CholCheck': 1, 'BMI': bmi,
        'Smoker': 1 if smoker == "Yes" else 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 1 if heart_disease == "Yes" else 0,
        'PhysActivity': 1 if phys_act == "Yes" else 0,
        'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0, 'AnyHealthcare': 1, 'NoDocbcCost': 0,
        'GenHlth': gen_hlth, 'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 1, 'Age': age,
        'Education': 5, 'Income': 5
    }

    # Cast the formatted payload dictionary object directly into a Pandas DataFrame structure
    return pd.DataFrame([data])


# Instantiate the user metrics payload object reference
input_df = get_user_input()

# ==========================================
# 4. PIPELINE EXECUTION LAYER (CORE LOGIC)
# ==========================================

# Divide the display interface layout into two discrete equal column elements
col1, col2 = st.columns(2)

# Column 1 Block: Machine Learning Predictions and Mathematical Interpretations
with col1:
    st.subheader("AI Model Prediction")

    # Generate assessment conditions when the trigger button is engaged
    if st.button("Run Clinical Assessment"):
        # Run the prediction (returns 0 or 1)
        prediction = model.predict(input_df)[0]

        # Extract the exact probability matrix row safely
        prob_matrix = model.predict_proba(input_df)

        # Get the probability score for class 1 (risk of diabetes)
        probability = float(prob_matrix[0][1])

        # Display responsive warning banners mapping directly to the computed boolean flag
        if prediction == 1:
            st.error(
                f"🚨 High Risk Detected! Probability: {probability * 100:.2f}%")
        else:
            st.success(f"✅ Low Risk. Probability: {probability * 100:.2f}%")

        # Initialize Explainable AI plotting matrix
        st.subheader("🔍 SHAP Explanation (Why?)")

        # Instantiate a native additive shapley attribution engine from the booster data
        explainer = shap.Explainer(model)
        # Calculate real-time mathematical feature contribution score tracking matrices
        shap_values = explainer(input_df)

        # Set up a Matplotlib plotting frame to render the vector chart asset smoothly
        fig, ax = plt.subplots(figsize=(6, 4))
        # Draw a custom localized waterfall plot tracking feature impact weights
        shap.plots.waterfall(shap_values[0], max_display=6, show=False)
        # Feed the Matplotlib memory buffer stack map directly into the UI canvas
        st.pyplot(plt.gcf())
        # Clear the graph context arrays to prevent pipeline rendering leak issues
        plt.clf()

        # Extract and rank the absolute array index values for the top driving parameters
        top_indices = np.argsort(np.abs(shap_values.values[0]))[-3:]
        # Map indices back into explicit string arrays using the pre-declared schema
        top_features = [feature_names[i] for i in top_indices]

        # Column 2 Block: Asynchronous Language Model Context Processing
        with col2:
            st.subheader("📝 LLM Generated Clinical Summary")

            # Open a live loading spinner component during remote server processing loops
            with st.spinner("Qwen is analyzing the XAI features..."):
                try:
                    # Initialize the authorized remote cloud service provider connection framework
                    client = Groq(
                        api_key="Your_Groq_API_Key")

                    # Generate a clean, instruction-optimized dynamic runtime execution prompt string
                    # Note: probability is now a clean float, so we use it directly without [0][1] brackets!
                    prompt = f"""
                    You are an expert digital health assistant. 
                    The XGBoost model predicted a diabetes probability of {probability * 100:.2f}%.
                    The top 3 driving features computed via SHAP values are: {', '.join(top_features)}.
                    
                    Provide an empathetic, plain-language clinical interpretation of these results for the patient. 
                    Explain why these specific features matter and give 3 tailored preventative health tips.
                    """

                    # Package and dispatch the runtime prompt structure over authorized cloud streams
                    completion = client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )

                    # Access the first element of the choices list safely using index 0
                    st.info(completion.choices[0].message.content)

                except Exception as e:
                    # Provide automated safe fallback messaging to log real-time stack trace messages
                    st.error(f"Actual Error: {e}")
