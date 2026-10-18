import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. SETUP HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="Semantic Search Engine", layout="wide")
st.title("🔍 Mesin Pencari Berita Konseptual (GloVe NLP)")
st.write("Cari berita berdasarkan **makna/konsep**, bukan sekadar persamaan kata kunci!")

# ---------------------------------------------------------
# 2. LOAD MODEL GLOVE & DATASET
# ---------------------------------------------------------
@st.cache_resource
def load_glove_model(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

@st.cache_data
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df

GLOVE_PATH = r'datasets/glove_50dim_wiki.id.case.text.txt'
NEWS_CSV_PATH = r'datasets/indonesian_news_clean.csv'

glove_model = load_glove_model(GLOVE_PATH)
df_news = load_dataset(NEWS_CSV_PATH)

COLUMN_TITLE = 'Judul'  # Sesuaikan nama kolom judul berita
COLUMN_URL = 'Link'      # Sesuaikan nama kolom link berita

titles = df_news[COLUMN_TITLE].astype(str).tolist()
urls = df_news[COLUMN_URL].tolist() if COLUMN_URL in df_news.columns else ["#"] * len(titles)

# ---------------------------------------------------------
# 3. FUNGSI HITUNG VEKTOR & SEMANTIC SEARCH
# ---------------------------------------------------------
def get_document_vector(text):
    words = str(text).lower().split()
    word_vecs = [glove_model[w] for w in words if w in glove_model]
    if len(word_vecs) == 0:
        return np.zeros(50)
    return np.mean(word_vecs, axis=0)

@st.cache_data
def compute_all_doc_vectors(titles_list):
    return np.array([get_document_vector(t) for t in titles_list])

doc_vectors = compute_all_doc_vectors(titles)

# ---------------------------------------------------------
# 4. TAMPILAN INTERFACE PENCARIAN
# ---------------------------------------------------------
query = st.text_input("Masukkan kalimat pencarian:", placeholder="Contoh: Koperasi Merah Putih")
top_k = st.slider("Jumlah hasil yang ditampilkan:", min_value=1, max_value=20, value=4)

if query:
    query_vector = get_document_vector(query)
    similarities = cosine_similarity([query_vector], doc_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    st.subheader(f"Top {top_k} Hasil Pencarian Relevan:")
    
    results = []
    for idx in top_indices:
        results.append({
            "Judul Berita": titles[idx],
            "Link Berita": urls[idx]
        })
    
    df_display = pd.DataFrame(results)
    
    st.data_editor(
        df_display,
        column_config={
            "Link Berita": st.column_config.LinkColumn("Link Berita", display_text="Buka Berita 🔗")
        },
        hide_index=True,
        use_container_width=True,
        disabled=True
    )
    
    # ---------------------------------------------------------
    # 5. DETAIL TEKNIS & JABARAN DATA VEKTOR GLOVE (PRESENTASI)
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 Detail Teknis GloVe"):
        st.markdown("### 1. Nilai Similarity Score Hasil Pencarian")
        
        detail_results = []
        for idx in top_indices:
            detail_results.append({
                "Judul Berita": titles[idx],
                "Cosine Similarity Score": f"{similarities[idx]:.4f}"
            })
        st.dataframe(pd.DataFrame(detail_results), use_container_width=True)
        
        st.markdown("### 2. Alur Pembentukan Vektor GloVe pada Query Ini")
        
        words_in_query = [w for w in query.lower().split() if w in glove_model]
        words_missing = [w for w in query.lower().split() if w not in glove_model]
        
        st.write(f"**a. Kata yang Terdeteksi di Kamus GloVe:** `{words_in_query}`")
        if words_missing:
            st.write(f"**Kata Out-of-Vocabulary (OOV):** `{words_missing}`")
            
        if words_in_query:
            st.write("**b. Data Vektor 50-Dimensi Masing-Masing Kata:**")
            vec_dict = {word: glove_model[word] for word in words_in_query}
            df_vectors = pd.DataFrame(vec_dict).T
            # Tampilkan matriks vektor kata
            st.dataframe(df_vectors, use_container_width=True)
            
            st.write("**c. Hasil Rata-Rata Vektor Query (Vector Average) yang Digunakan untuk Mencari Berita:**")
            df_query_vec = pd.DataFrame([query_vector], index=["Vektor Query Rata-rata"])
            st.dataframe(df_query_vec, use_container_width=True)
        
        st.info("""
        **💡 Penjelasan:**
        1. Setiap kata dalam query diekstrak vektor 50-dimensinya dari pretrained GloVe.
        2. Seluruh vektor kata dijumlahkan dan dirata-ratakan (*Vector Averaging*) untuk membentuk 1 vektor representasi kalimat.
        3. Vektor kalimat ini dibandingkan dengan seluruh vektor judul berita menggunakan **Cosine Similarity** untuk mendapatkan skor kemiripan (0.0 - 1.0).
        """)