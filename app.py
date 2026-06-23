import streamlit as st

st.set_page_config(
    page_title=" Deteksi Data Penyakit Jantung",
    page_icon="🏥",
    layout="wide"
)

# Inisialisasi session state
if "username" not in st.session_state:
    st.session_state.username = None

# Halaman login / input nama
if st.session_state.username is None:
    st.title("🏥 Form Login")
    st.subheader("Silakan masukkan nama Anda terlebih dahulu baru bisa masuk")

    with st.form("form_nama"):
        nama = st.text_input("Nama")
        submit = st.form_submit_button("Masuk")

        if submit:
            if nama.strip():
                st.session_state.username = nama
                st.rerun()
            else:
                st.warning("Nama tidak boleh kosong")

# Setelah nama diisi, tampilkan aplikasi
else:
    st.sidebar.success(f"Halo, {st.session_state.username} 👋")

    if st.sidebar.button("Logout"):
        st.session_state.username = None
        st.rerun()

    pg = st.navigation([
        st.Page(
            "packages/1_visualization.py",
            title="Visualization",
            icon="📊"
        ),
        
        st.Page(
            "packages/2_prediction.py",
            title="Prediction",
            icon="💡"
        ),
            st.Page(
            "https://github.com/JTMT1997",
            title="Github",
            icon="🐙"
        ),
    ])

    pg.run()