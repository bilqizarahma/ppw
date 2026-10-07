import re
import joblib
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup


# ============================================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Berita Detik",
    page_icon="📰",
    layout="centered"
)

# Custom Styling Tampilan UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .card-sport {
        background-color: #EFF6FF;
        border-left: 6px solid #2563EB;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .card-finance {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model_w2v = joblib.load("model/w2v_model.pkl")
model_nb = joblib.load("model/nb_model.pkl")


# ============================================================
# PREPROCESSING
# ============================================================

def preprocessing(teks):

    teks = str(teks).lower()

    # Menghapus URL
    teks = re.sub(r"http\S+|www\S+", "", teks)

    # Menghapus angka
    teks = re.sub(r"\d+", "", teks)

    # Menghapus tanda baca
    teks = re.sub(r"[^a-z\s]", " ", teks)

    # Menghapus spasi berlebih
    teks = re.sub(r"\s+", " ", teks).strip()

    return teks


# ============================================================
# MENGUBAH TEKS MENJADI VEKTOR
# ============================================================

def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


# ============================================================
# MENGAMBIL ISI BERITA DARI URL
# ============================================================

def ambil_berita_dari_url(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Selector utama Detik
    isi = soup.select_one(
        "div.detail__body-text"
    )

    # Selector alternatif
    if isi is None:
        isi = soup.select_one(
            "div.detail__body"
        )

    if isi is None:
        isi = soup.select_one(
            "div.itp_bodycontent"
        )

    if isi is None:
        isi = soup.select_one(
            "article"
        )

    if isi is None:
        return None

    return isi.get_text(
        separator=" ",
        strip=True
    )


# ============================================================
# KLASIFIKASI BERITA
# ============================================================

def klasifikasi_berita(teks):

    teks_bersih = preprocessing(teks)

    tokens = teks_bersih.split()

    vektor = document_vector(
        tokens,
        model_w2v
    )

    # Mengambil probabilitas setiap kategori
    probabilitas = model_nb.predict_proba(
        [vektor]
    )[0]

    # Mengambil kategori dengan probabilitas tertinggi
    indeks = np.argmax(probabilitas)

    hasil = model_nb.classes_[indeks]

    confidence = probabilitas[indeks]

    return hasil, confidence


# ============================================================
# VALIDASI KATEGORI BERDASARKAN URL
# ============================================================

def cek_kategori_url(url):

    url = url.lower()

    if "sport.detik.com" in url:
        return "sport"

    elif "finance.detik.com" in url:
        return "finance"

    else:
        return None


# ============================================================
# TAMPILAN APLIKASI
# ============================================================

st.markdown('<p class="main-title">📰 Klasifikasi Berita Detik</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Aplikasi klasifikasi berita menggunakan <b>Skip-gram Word2Vec dan Naive Bayes</b>.</p>', unsafe_allow_html=True)


# ============================================================
# INPUT URL
# ============================================================

url = st.text_input(
    "🔗 URL Berita Detik",
    placeholder="https://sport.detik.com/... atau https://finance.detik.com/..."
)


# ============================================================
# TOMBOL KLASIFIKASI
# ============================================================

if st.button(
    "🔍 Klasifikasi Berita",
    use_container_width=True,
    type="primary"
):

    if url.strip() == "":

        st.warning(
            "Silakan masukkan URL berita terlebih dahulu."
        )

    else:

        # Mengecek kategori URL
        kategori_url = cek_kategori_url(url)

        if kategori_url is None:

            st.warning(
                "⚠️ Kategori berita tidak didukung."
            )

            st.write(
                "Aplikasi hanya dapat mengklasifikasikan "
                "berita dengan kategori SPORT dan FINANCE."
            )

        else:

            try:

                with st.spinner(
                    "Mengambil dan memproses berita..."
                ):

                    berita = ambil_berita_dari_url(
                        url
                    )

                if berita is None:

                    st.error(
                        "Isi berita tidak ditemukan."
                    )

                else:

                    hasil, confidence = klasifikasi_berita(
                        berita
                    )

                    st.success(
                        "Berita berhasil diklasifikasikan!"
                    )

                    st.subheader(
                        "Hasil Klasifikasi"
                    )

                    # Tampilan Kartu Hasil Klasifikasi
                    if hasil.lower() == "sport":

                        st.markdown(
                            f"""
                            <div class="card-sport">
                                <h3 style="margin:0; color:#1E40AF;">🏆 Kategori: SPORT</h3>
                                <p style="margin:5px 0 0 0; color:#2563EB; font-weight:600;">Tingkat Keyakinan: {confidence * 100:.2f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    elif hasil.lower() == "finance":

                        st.markdown(
                            f"""
                            <div class="card-finance">
                                <h3 style="margin:0; color:#065F46;">💰 Kategori: FINANCE</h3>
                                <p style="margin:5px 0 0 0; color:#10B981; font-weight:600;">Tingkat Keyakinan: {confidence * 100:.2f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with st.expander(
                        "📄 Lihat isi berita"
                    ):

                        st.write(berita)

            except requests.exceptions.RequestException:

                st.error(
                    "URL tidak dapat diakses. "
                    "Pastikan URL berita Detik benar."
                )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan: {e}"
                )