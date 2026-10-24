import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# =====================================================================
# 1. DATA INGESTION & PIPELINE SETUP
# =====================================================================

# Load the raw behavioral health dataset from the local directory path
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# =====================================================================
# 2. FEATURE ENGINEERING & EXTRACTION
# =====================================================================

# Isolate independent features (clinical metrics) by dropping the target column
X = df.drop('Diabetes_binary', axis=1)

# Isolate the dependent target variable vector (Diabetes status: 0 or 1)
y = df['Diabetes_binary']

# Partition rows into 80% for optimization modeling and 20% for testing validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 3. AI MODEL TRAINING & SERIALIZATION
# =====================================================================

# Instantiate an XGBoost Classifier with 100 estimators and bounded tree depth
model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)

# Fit the machine learning booster algorithm on the training matrices
model.fit(X_train, y_train)

# Evaluate and display localized performance validation metrics
accuracy = model.score(X_test, y_test)
print(f"Model validation completed. Baseline accuracy score: {accuracy * 100:.2f}%")

# Serialize the mathematically trained parameters down into a structural JSON artifact
model.save_model('diabetes_model.json')
print("Model artifacts successfully serialized and saved as 'diabetes_model.json'!")
