
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

import joblib
# from tensorflow.keras.layers import Dense
# Load the trained ANN model
from tensorflow.keras.layers import Dense

model = tf.keras.models.load_model(
    "heart_disease_ann_model.h5",
    compile=False,
    custom_objects={"Dense": Dense}
)

# Load the StandardScaler
scaler = joblib.load('scaler.pkl')

# Load feature names
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Prediction using ANN")
st.markdown("This application predicts the likelihood of heart disease based on various patient parameters.")

st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex', options=[('Male', 1), ('Female', 0)], format_func=lambda x: x[0])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', options=[('Typical Angina', 0), ('Atypical Angina', 1), ('Non-anginal Pain', 2), ('Asymptomatic', 3)], format_func=lambda x: x[0])
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 120)
    chol = st.sidebar.slider('Cholesterol (chol)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[('False', 0), ('True', 1)], format_func=lambda x: x[0])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', options=[('Normal', 0), ('ST-T wave abnormality', 1), ('Left ventricular hypertrophy', 2)], format_func=lambda x: x[0])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', options=[('No', 0), ('Yes', 1)], format_func=lambda x: x[0])
    oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest (oldpeak)', 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', options=[('Upsloping', 0), ('Flat', 1), ('Downsloping', 2)], format_func=lambda x: x[0])
    ca = st.sidebar.selectbox('Number of major vessels (0-3) colored by fluoroscopy (ca)', options=[('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4)], format_func=lambda x: x[0]) # Assuming '4' is also a valid category after one-hot, or might represent missing/other
    thal = st.sidebar.selectbox('Thalassemia (thal)', options=[('Unknown', 0), ('Normal', 1), ('Fixed Defect', 2), ('Reversible Defect', 3)], format_func=lambda x: x[0])

    data = {
        'age': age,
        'sex': sex[1],
        'cp': cp[1],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs[1],
        'restecg': restecg[1],
        'thalach': thalach,
        'exang': exang[1],
        'oldpeak': oldpeak,
        'slope': slope[1],
        'ca': ca[1],
        'thal': thal[1]
    }

    # Convert categorical selections to their integer values
    for key in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        if isinstance(data[key], tuple): # If still a tuple, extract the integer
            data[key] = data[key][1]

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# One-hot encode the categorical features from user input
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
encoded_cols = []
for col in categorical_cols:
    for val in sorted(input_df[col].unique()):
        # Create columns like 'sex_1', 'cp_1', 'cp_2', etc.
        if val != 0: # Assuming 0 is the base category to drop (like drop_first=True)
            encoded_cols.append(f'{col}_{val}')

# Create a DataFrame for the encoded input, ensuring all expected columns are present
input_encoded = pd.DataFrame(0, index=[0], columns=feature_names)

# Fill in numerical features
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    if col in input_df.columns:
        input_encoded[col] = input_df[col]

# Fill in one-hot encoded features
for col_name in input_df.columns:
    if col_name in categorical_cols:
        val = input_df[col_name].iloc[0]
        if f'{col_name}_{val}' in feature_names: # Check if the specific encoded column exists
            input_encoded[f'{col_name}_{val}'] = 1

# Scale numerical features
numerical_cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
input_encoded[numerical_cols_to_scale] = scaler.transform(input_encoded[numerical_cols_to_scale])

# Make prediction
prediction_proba = model.predict(input_encoded)
prediction = (prediction_proba > 0.5).astype(int)

st.subheader('Prediction Result')

if prediction[0][0] == 0:
    st.success(f"The model predicts: **No Heart Disease** (Probability: {prediction_proba[0][0]:.2f})")
else:
    st.error(f"The model predicts: **Heart Disease** (Probability: {prediction_proba[0][0]:.2f})")

st.subheader('Prediction Probability')
st.write(f"Probability of having Heart Disease: {prediction_proba[0][0]:.4f}")