import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ======= Configuration Halaman =======
st.title("❤️ Prediksi Penyakit Jantung")
st.markdown(
    "Aplikasi ini memprediksi kemungkinan penyakit jantung berdasarkan berbagai parameter pasien "
    "menggunakan model Decision Tree, Random Forest, dan Artificial Neural Network (ANN)."
)

# ======= Load Models and Data =======
@st.cache_resource
def load_all_resources():
    ann_model = tf.keras.models.load_model('models/heart_disease_ann_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    dt_model = joblib.load('models/decision_tree_model.joblib')
    rf_model = joblib.load('models/random_forest_tuned_model.joblib')

    return ann_model, scaler, feature_names, dt_model, rf_model

ann_model, scaler, feature_names, dt_model, rf_model = load_all_resources()

# ======= Form Input Tengah =======
left_space, center_col, right_space = st.columns([1, 3, 1])

with center_col:

    st.subheader("🩺 Input Data Pasien")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Usia', 29, 77, 50)

        sex = st.selectbox(
            'Jenis Kelamin',
            options=[('Laki-laki', 1), ('Perempuan', 0)],
            format_func=lambda x: x[0]
        )

        cp = st.selectbox(
            'Tipe Nyeri Dada',
            options=[
                ('Angina Tipikal', 0),
                ('Angina Atipikal', 1),
                ('Nyeri Non-anginal', 2),
                ('Asimtomatik', 3)
            ],
            format_func=lambda x: x[0]
        )

        trestbps = st.slider(
            'Tekanan Darah Istirahat (mmHg)',
            94, 200, 120
        )

        chol = st.slider(
            'Kolesterol (mg/dL)',
            126, 564, 240
        )

        fbs = st.selectbox(
            'Gula Darah Puasa > 120 mg/dL',
            options=[('False', 0), ('True', 1)],
            format_func=lambda x: x[0]
        )

    with col2:

        restecg = st.selectbox(
            'Hasil Elektrokardiografi',
            options=[
                ('Normal', 0),
                ('Abnormalitas gelombang ST-T', 1),
                ('Hipertrofi ventrikel kiri', 2)
            ],
            format_func=lambda x: x[0]
        )

        thalach = st.slider(
            'Detak Jantung Maksimum',
            71, 202, 150
        )

        exang = st.selectbox(
            'Angina Akibat Olahraga',
            options=[('Tidak', 0), ('Ya', 1)],
            format_func=lambda x: x[0]
        )

        oldpeak = st.slider(
            'Oldpeak',
            0.0, 6.2, 1.0, 0.1
        )

        slope = st.selectbox(
            'Kemiringan Segmen ST',
            options=[
                ('Meningkat', 0),
                ('Datar', 1),
                ('Menurun', 2)
            ],
            format_func=lambda x: x[0]
        )

        ca = st.selectbox(
            'Jumlah Pembuluh Darah Utama',
            options=[
                ('0', 0),
                ('1', 1),
                ('2', 2),
                ('3', 3),
                ('4', 4)
            ],
            format_func=lambda x: x[0]
        )

        thal = st.selectbox(
            'Thalasemia',
            options=[
                ('Tidak Diketahui', 0),
                ('Normal', 1),
                ('Defek Tetap', 2),
                ('Defek Reversibel', 3)
            ],
            format_func=lambda x: x[0]
        )

    predict_btn = st.button(
        "🔍 Prediksi Penyakit Jantung",
        use_container_width=True
    )

# ======= Prediksi =======
if predict_btn:

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

    input_df = pd.DataFrame(data, index=[0])

    st.subheader("📋 Parameter Input")
    st.dataframe(input_df, use_container_width=True)

    categorical_cols = [
        'sex', 'cp', 'fbs', 'restecg',
        'exang', 'slope', 'ca', 'thal'
    ]

    input_encoded = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    numerical_cols = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak'
    ]

    for col in numerical_cols:
        input_encoded[col] = input_df[col]

    for col in categorical_cols:
        value = input_df[col].iloc[0]
        column_name = f"{col}_{value}"

        if column_name in feature_names:
            input_encoded[column_name] = 1

    input_encoded[numerical_cols] = scaler.transform(
        input_encoded[numerical_cols]
    )

    st.divider()
    st.subheader("🩺 Hasil Prediksi")

    # Decision Tree
    dt_pred = dt_model.predict(input_encoded)[0]

    # Random Forest
    rf_pred = rf_model.predict(input_encoded)[0]

    # ANN
    ann_prob = ann_model.predict(
        input_encoded,
        verbose=0
    )[0][0]

    ann_pred = 1 if ann_prob > 0.5 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        if dt_pred == 0:
            st.success("🌳 Decision Tree\n\nTidak Ada Penyakit Jantung")
        else:
            st.error("🌳 Decision Tree\n\nAda Penyakit Jantung")

    with col2:
        if rf_pred == 0:
            st.success("🌲 Random Forest\n\nTidak Ada Penyakit Jantung")
        else:
            st.error("🌲 Random Forest\n\nAda Penyakit Jantung")

    with col3:
        if ann_pred == 0:
            st.success(
                f"🧠 ANN\n\nTidak Ada Penyakit Jantung\n\nProbabilitas: {(1-ann_prob)*100:.1f}%"
            )
        else:
            st.error(
                f"🧠 ANN\n\nAda Penyakit Jantung\n\nProbabilitas: {ann_prob*100:.1f}%"
            )