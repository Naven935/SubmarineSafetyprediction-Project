# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------------------- Preprocessing Data -------------------------

# Read the input CSV file
sonar_data = pd.read_csv("SonarData.csv", header=None)

# Ensure all feature columns (except the last) are numeric
sonar_features = sonar_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Keep the last column (labels) unchanged
sonar_labels = sonar_data.iloc[:, -1]

# Merge features & labels again
sonar_data = pd.concat([sonar_features, sonar_labels], axis=1)

# Drop rows where all feature values are NaN (but keep labels)
sonar_data.dropna(subset=sonar_features.columns, inplace=True)

# Ensure there is data to process
if sonar_data.shape[0] == 0:
    raise ValueError("❌ ERROR: SonarData.csv contains no valid numeric data after processing.")

# Split data into features (X) and labels (y)
X = sonar_data.iloc[:, :-1].values
y = sonar_data.iloc[:, -1].values

# Standardize features (ensures better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------- Hyperparameter Tuning -------------------------

# Optimized SVM Model
param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_

# Optimized Random Forest Model
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Define optimized models
LRModel = LogisticRegression()
KnnModel = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Train models
LRModel.fit(X_train, y_train)
KnnModel.fit(X_train, y_train)

# ----------------------- Ensemble Learning (Voting Classifier) -------------------------

ensemble_model = VotingClassifier(estimators=[
    ('svm', best_svm),
    ('knn', KnnModel),
    ('rf', rf)
], voting='hard')

# Train ensemble model
ensemble_model.fit(X_train, y_train)

# ----------------------- Function to Predict New Samples -------------------------

def Predict_New_Class(fresh_sample):
    st.subheader("📊 Model Accuracy Scores")

    # Calculate accuracy scores
    accuracy_LR = accuracy_score(y_test, LRModel.predict(X_test))
    accuracy_knn = accuracy_score(y_test, KnnModel.predict(X_test))
    accuracy_svm = accuracy_score(y_test, best_svm.predict(X_test))
    accuracy_rf = accuracy_score(y_test, rf.predict(X_test))
    accuracy_ensemble = accuracy_score(y_test, ensemble_model.predict(X_test))

    # Display accuracy in Streamlit
    st.write(f"✅ Logistic Regression Accuracy: {accuracy_LR:.2f}")
    st.write(f"✅ KNN Accuracy: {accuracy_knn:.2f}")
    st.write(f"✅ SVM (Optimized) Accuracy: {accuracy_svm:.2f}")
    st.write(f"✅ Random Forest Accuracy: {accuracy_rf:.2f}")
    st.write(f"🔥 Ensemble Model Accuracy: {accuracy_ensemble:.2f} 🚀")

    # Convert all values to numeric and handle NaN values
    fresh_sample = fresh_sample.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with column mean
    fresh_sample.fillna(fresh_sample.mean(), inplace=True)

    # Ensure correct number of features (handle extra columns)
    if fresh_sample.shape[1] > X_train.shape[1]:
        fresh_sample = fresh_sample.iloc[:, :X_train.shape[1]]  # Remove extra columns automatically
    elif fresh_sample.shape[1] < X_train.shape[1]:
        st.error(f"❌ ERROR: Uploaded file has {fresh_sample.shape[1]} features, expected {X_train.shape[1]}. Please check the data format.")
        return

    # Standardize new data
    fresh_sample_scaled = scaler.transform(fresh_sample)

    st.subheader("📢 Prediction Results")

    # Predicting the class of new samples
    for index, sample in enumerate(fresh_sample_scaled):
        try:
            # Make predictions
            prediction_of_ensemble = ensemble_model.predict([sample])[0]

            # Determine final prediction
            final_prediction = "🪨 The given sample is a Rock!!" if prediction_of_ensemble == 'R' else "💎 The given sample is a Mine!!"

            # Display results
            st.write(f"**Sample {index}: {final_prediction}**")

        except Exception as e:
            st.error(f"❌ ERROR processing sample {index}: {e}")

# ----------------------- Streamlit UI -------------------------

# Hide menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("🔍 Rock Vs Mine Detection using Machine Learning")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Your Input CSV File for Prediction", type=["csv"])

if uploaded_file is not None:
    uploaded_file_as_dataFrame = pd.read_csv(uploaded_file, header=None)

    # Display the uploaded data
    st.dataframe(uploaded_file_as_dataFrame)

    # Call the prediction function
    Predict_New_Class(uploaded_file_as_dataFrame)
