
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. KONFIGURASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
GLOVE_PATH = BASE_DIR / "datasets" / "glove_50dim_wiki.id.case.text.txt"
CSV_PATH = BASE_DIR / "datasets" / "indonesian_news_clean.csv"
VECTOR_SIZE = 50

st.set_page_config(
    page_title="Semantic News Search - Optimized",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Semantic News Search Engine")
st.caption(
    "Versi eksperimen: GloVe 50d + TF-IDF + "
    "Judul, Konten, dan Tag Berita"
)


# Stopword umum. Kata penting seperti "tidak", "tanpa",
# "naik", dan "turun" sengaja tidak dihapus.
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "untuk", "pada", "dengan", "atau", "sebagai", "oleh",
    "dalam", "adalah", "juga", "akan", "telah", "sudah",
    "dapat", "bisa", "para", "agar", "karena", "bahwa",
    "saat", "secara", "menjadi", "tersebut", "antara",
    "hingga", "sehingga", "serta", "pun", "ia", "dia",
}


def clean_text(text):
    """Normalisasi teks dasar tanpa mengubah makna kata."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.lower()
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return " ".join(tokens)


# ==================================================
# 2. MEMUAT MODEL DAN MEMBANGUN INDEKS
# ==================================================

@st.cache_resource(show_spinner="Memuat model GloVe...")
def load_glove_model():
    if not GLOVE_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {GLOVE_PATH}")

    embeddings = {}

    with open(GLOVE_PATH, "r", encoding="utf-8") as file:
        for line in file:
            values = line.rstrip().split()

            if len(values) != VECTOR_SIZE + 1:
                continue

            word = values[0]

            try:
                vector = np.asarray(values[1:], dtype=np.float32)
            except ValueError:
                continue

            embeddings[word] = vector

    if not embeddings:
        raise ValueError("Model GloVe kosong atau formatnya tidak valid.")

    return embeddings


def average_vector(text, model, max_tokens=None):
    """Rata-rata vektor kata yang tersedia dalam model."""
    tokens = text.split()

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    vectors = []

    for word in tokens:
        if word in STOPWORDS:
            continue

        vector = model.get(word)

        # Coba variasi kapitalisasi jika bentuk kecil tidak tersedia.
        if vector is None:
            vector = model.get(word.capitalize())

        if vector is not None:
            vectors.append(vector)

    if not vectors:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


@st.cache_resource(show_spinner="Membangun indeks pencarian berita...")
def build_search_index():
    model = load_glove_model()

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = ["Judul", "Content", "Link"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Kolom dataset tidak ditemukan: {missing}")

    # Tangani nilai kosong dan abaikan berita tanpa judul.
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["Judul"] != ""].copy()

    # Pengaman: jangan tampilkan URL yang sama berulang kali.
    df = df.drop_duplicates(subset=["Link"], keep="first")
    df = df.reset_index(drop=True)

    tag_columns = [
        col for col in df.columns
        if re.fullmatch(r"tag\d+", col, flags=re.IGNORECASE)
    ]

    df["_title_clean"] = df["Judul"].map(clean_text)
    df["_content_clean"] = df["Content"].map(clean_text)

    if tag_columns:
        df["_tags_clean"] = (
            df[tag_columns].agg(" ".join, axis=1).map(clean_text)
        )
    else:
        df["_tags_clean"] = ""

    # TF-IDF melihat seluruh informasi berita.
    # Judul sengaja diulang agar istilah pada judul lebih menonjol.
    df["_search_text"] = (
        df["_title_clean"] + " "
        + df["_title_clean"] + " "
        + df["_tags_clean"] + " "
        + df["_content_clean"]
    ).str.strip()

    tfidf = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )

    tfidf_matrix = tfidf.fit_transform(df["_search_text"])

    # Bangun vektor GloVe untuk setiap berita.
    # Judul paling penting, lalu konten, lalu tag.
    doc_vectors = np.zeros(
        (len(df), VECTOR_SIZE),
        dtype=np.float32,
    )

    for i, row in enumerate(
        df[["_title_clean", "_content_clean", "_tags_clean"]]
        .itertuples(index=False, name=None)
    ):
        title, content, tags = row

        title_vec = average_vector(title, model)
        content_vec = average_vector(content, model, max_tokens=250)
        tags_vec = average_vector(tags, model, max_tokens=50)

        parts = [
            (title_vec, 0.65),
            (content_vec, 0.25),
            (tags_vec, 0.10),
        ]

        valid_parts = [
            (vec, weight)
            for vec, weight in parts
            if np.linalg.norm(vec) > 0
        ]

        if valid_parts:
            total_weight = sum(weight for _, weight in valid_parts)

            combined = sum(
                vec * weight for vec, weight in valid_parts
            ) / total_weight

            doc_vectors[i] = combined

    return model, df, doc_vectors, tfidf, tfidf_matrix


def minmax_normalize(scores):
    """Normalisasi skor dalam satu kumpulan hasil query."""
    scores = np.asarray(scores, dtype=np.float32)

    low = float(scores.min())
    high = float(scores.max())

    if high - low < 1e-8:
        return np.zeros_like(scores)

    return (scores - low) / (high - low)


# ==================================================
# 3. MENYIAPKAN MODEL
# ==================================================

try:
    (
        glove_model,
        df_news,
        doc_vectors,
        tfidf_vectorizer,
        tfidf_matrix,
    ) = build_search_index()
except Exception as error:
    st.error(f"Gagal mempersiapkan indeks: {error}")
    st.stop()


col1, col2, col3 = st.columns(3)
col1.metric("Jumlah berita", f"{len(df_news):,}")
col2.metric("Dimensi GloVe", VECTOR_SIZE)
col3.metric("Metode pencarian", "Hybrid")

top_k = st.sidebar.slider(
    "Jumlah hasil pencarian",
    min_value=1,
    max_value=20,
    value=5,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Skor akhir menggabungkan GloVe dan TF-IDF. "
    "Skor bukan probabilitas relevansi."
)


# ==================================================
# 4. FORM PENCARIAN
# ==================================================

with st.form("search_form"):
    query = st.text_input(
        "Masukkan kata kunci atau kalimat",
        placeholder="Contoh: harga bahan pokok naik",
    )

    submitted = st.form_submit_button("🔍 Cari Berita")

if submitted:
    query_clean = clean_text(query)

    if not query_clean:
        st.warning("Masukkan kata kunci terlebih dahulu.")
        st.stop()

    tokens = [
        word for word in query_clean.split()
        if word not in STOPWORDS
    ]

    known_tokens = [
        word for word in tokens
        if word in glove_model or word.capitalize() in glove_model
    ]

    query_vector = average_vector(query_clean, glove_model)

    # Skor semantik GloVe.
    if np.linalg.norm(query_vector) > 0:
        glove_scores = cosine_similarity(
            query_vector.reshape(1, -1),
            doc_vectors,
        )[0]
    else:
        glove_scores = np.zeros(len(df_news), dtype=np.float32)

    # Skor pencocokan kata/frasa menggunakan TF-IDF.
    query_tfidf = tfidf_vectorizer.transform([query_clean])

    if query_tfidf.nnz > 0:
        tfidf_scores = cosine_similarity(
            query_tfidf,
            tfidf_matrix,
        )[0]
    else:
        tfidf_scores = np.zeros(len(df_news), dtype=np.float32)

    # Jika query tidak memiliki token GloVe, gunakan TF-IDF saja.
    if not known_tokens and query_tfidf.nnz == 0:
        st.warning(
            "Tidak ada kata query yang cocok dengan kosakata model "
            "atau indeks TF-IDF. Coba kata lain."
        )
        st.stop()

    glove_norm = minmax_normalize(glove_scores)
    tfidf_norm = minmax_normalize(tfidf_scores)

    if not known_tokens:
        combined_scores = tfidf_norm
    elif query_tfidf.nnz == 0:
        combined_scores = glove_norm
    else:
        combined_scores = (
            0.65 * glove_norm
            + 0.35 * tfidf_norm
        )

    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    st.subheader(f"Hasil pencarian: {query}")

    st.caption(
        "Urutan menggunakan skor gabungan GloVe dan TF-IDF. "
        "Skor gabungan telah dinormalisasi untuk query ini."
    )

    for rank, idx in enumerate(top_indices, start=1):
        row = df_news.iloc[idx]

        title = row["Judul"]
        link = row["Link"]

        with st.container(border=True):
            st.markdown(f"### #{rank} — {title}")

            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric(
                "Skor gabungan",
                f"{combined_scores[idx]:.4f}",
            )
            score_col2.metric(
                "Cosine GloVe",
                f"{glove_scores[idx]:.4f}",
            )
            score_col3.metric(
                "Cosine TF-IDF",
                f"{tfidf_scores[idx]:.4f}",
            )

            metadata = []

            if row.get("Waktu", ""):
                metadata.append(f"Tanggal: {row['Waktu']}")

            if row.get("source", ""):
                metadata.append(f"Sumber: {row['source']}")

            if metadata:
                st.caption(" | ".join(metadata))

            if link.startswith(("http://", "https://")):
                st.link_button("Buka berita asli", link)

            content = row.get("Content", "")

            if content:
                with st.expander("Lihat cuplikan konten"):
                    st.write(content[:800] + ("..." if len(content) > 800 else ""))

    # ==================================================
    # 5. DETAIL TEKNIS
    # ==================================================

    with st.expander("📊 Detail teknis GloVe dan TF-IDF"):
        st.write("Kata query yang ditemukan dalam model GloVe:")
        st.write(known_tokens or "Tidak ada")

        missing = [
            word for word in query_clean.split()
            if word not in glove_model
            and word.capitalize() not in glove_model
        ]

        st.write("Kata yang tidak ditemukan dalam GloVe:")
        st.write(missing or "Tidak ada")

        if known_tokens:
            st.write("Vektor kata (50 dimensi):")

            vectors = {
                word: glove_model.get(word, glove_model.get(word.capitalize()))
                for word in known_tokens
            }

            st.dataframe(
                pd.DataFrame(vectors).T,
                use_container_width=True,
            )

        st.write(
            "GloVe mengukur kedekatan semantik berdasarkan vektor. "
            "TF-IDF memberikan bobot pada kata dan frasa dalam dokumen. "
            "Keduanya digabung untuk menyusun peringkat berita."
        )
