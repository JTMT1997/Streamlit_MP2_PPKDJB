# %%writefile pages/1_visualization.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration Halaman
st.set_page_config(page_title="Visualisasi Data Penyakit Jantung", layout="wide")

st.title("📊 Visualisasi Data Penyakit Jantung")
st.markdown("Bagian ini menampilkan distribusi fitur numerik dan kategorikal dari dataset, dengan kemampuan filter.")

# ======= Load Data =======
@st.cache_data
def load_data():
    if os.path.exists('data/heart_data.csv'):
        data_df = pd.read_csv('data/heart_data.csv')
    else:
        st.error("File 'data/heart_data.csv' tidak ditemukan. Pastikan Anda telah menyimpannya di direktori root Colab.")
        st.stop()
    return data_df

data_df = load_data()

# Define categorical and numerical columns
categorical_cols_raw = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
target_col = 'target'

# Mappings for categorical features (from prediction.py)
sex_map = {0: 'Perempuan', 1: 'Laki-laki'}
cp_map = {0: 'Angina Tipikal', 1: 'Angina Atipikal', 2: 'Nyeri Non-anginal', 3: 'Asimtomatik'}
fbs_map = {0: 'False', 1: 'True'}
restecg_map = {0: 'Normal', 1: 'Abnormalitas gelombang ST-T', 2: 'Hipertrofi ventrikel kiri'}
exang_map = {0: 'Tidak', 1: 'Ya'}
slope_map = {0: 'Meningkat', 1: 'Datar', 2: 'Menurun'}
ca_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'} # Count of major vessels
thal_map = {0: 'Tidak Diketahui', 1: 'Normal', 2: 'Defek Tetap', 3: 'Defek Reversibel'}
target_map = {0: 'Tidak Ada Penyakit', 1: 'Ada Penyakit'}

# Apply mappings for display purposes in filters and plots
display_df = data_df.copy()
display_df['sex'] = display_df['sex'].map(sex_map)
display_df['cp'] = display_df['cp'].map(cp_map)
display_df['fbs'] = display_df['fbs'].map(fbs_map)
display_df['restecg'] = display_df['restecg'].map(restecg_map)
display_df['exang'] = display_df['exang'].map(exang_map)
display_df['slope'] = display_df['slope'].map(slope_map)
display_df['ca'] = display_df['ca'].map(ca_map)
display_df['thal'] = display_df['thal'].map(thal_map)
display_df['target'] = display_df['target'].map(target_map)

# Sidebar for Filters
st.sidebar.header("Filter Data")

filtered_display_df = display_df.copy()

# Numerical filters (range sliders)
st.sidebar.subheader("Filter Numerik")
for col in numerical_cols:
    min_val, max_val = float(data_df[col].min()), float(data_df[col].max()) # Use raw data for range
    selected_min, selected_max = st.sidebar.slider(
        f'{col.replace("_", " ").title()} Range',
        min_val, max_val, (min_val, max_val), key=f'slider_{col}'
    )
    # Filter based on original numerical values
    filtered_display_df = filtered_display_df[
        (data_df[col] >= selected_min) & (data_df[col] <= selected_max)
    ]

# Categorical filters (multiselect)
st.sidebar.subheader("Filter Kategorikal")
all_categorical_cols = categorical_cols_raw + [target_col] # Combine all categorical columns for filtering
for col in all_categorical_cols:
    if col in display_df.columns:
        unique_values = display_df[col].unique().tolist()
        selected_values = st.sidebar.multiselect(
            f'Pilih {col.replace("_", " ").title()}',
            unique_values,
            default=unique_values, # Select all by default
            key=f'multiselect_{col}'
        )
        filtered_display_df = filtered_display_df[filtered_display_df[col].isin(selected_values)]

st.subheader(f"Data Filtered ({len(filtered_display_df)} rows)")
if not filtered_display_df.empty:
    st.dataframe(filtered_display_df.head())
else:
    st.warning("Tidak ada data yang cocok dengan kriteria filter.")
    st.stop() # Stop the script if no data matches

# ======= Visualizations =======
st.subheader('Distribusi Fitur Numerik (Data yang Difilter)')
cols = st.columns(2)
for i, col in enumerate(numerical_cols):
    with cols[i % 2]:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(filtered_display_df[col], kde=True, ax=ax)
        ax.set_title(f'Distribusi {col.replace("_", " ").title()}')
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)
        plt.close(fig)

st.subheader('Distribusi Fitur Kategorikal (Data yang Difilter)')
cols = st.columns(2)
# Re-iterate through raw categorical columns for plotting, and target
for i, col in enumerate(all_categorical_cols):
    if col in filtered_display_df.columns:
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.countplot(x=col, data=filtered_display_df, ax=ax)
            ax.set_title(f'Distribusi {col.replace("_", " ").title()}')
            ax.set_xlabel(col.replace("_", " ").title())
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)