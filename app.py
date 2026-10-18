import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. SETUP HALAMAN & SIDEBAR
# ---------------------------------------------------------
st.set_page_config(
    page_title="Semantic News Search Engine",
    page_icon="📰",
    layout="wide"
)

# Sidebar Konfigurasi
st.sidebar.title("⚙️ Pengaturan")
top_k = st.sidebar.slider("Jumlah Hasil Pencarian:", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
st.sidebar.info("""
**Metode:** Word Embedding (GloVe 50d)  
**Dataset:** Indonesia News Dataset (2025) 
**Matriks Jarak:** Cosine Similarity
""")

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

COLUMN_TITLE = 'Judul'
COLUMN_URL = 'Link'

titles = df_news[COLUMN_TITLE].astype(str).tolist()
urls = df_news[COLUMN_URL].tolist() if COLUMN_URL in df_news.columns else ["#"] * len(titles)

# ---------------------------------------------------------
# 3. FUNGSI VECTOR AVERAGE & SEARCH
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
# 4. HEADER UTAMA & SEARCH BAR
# ---------------------------------------------------------
st.title("📰 Semantic News Search Engine")
st.caption("Mesin pencari berita cerdas berbasis kecerdasan bahasa **GloVe (Global Vectors)**")

# Layout Form Pencarian
with st.form(key='search_form'):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("", placeholder="Ketik kata kunci atau kalimat (contoh: peperangan timur tengah)...", label_visibility="collapsed")
    with col2:
        submit_button = st.form_submit_button(label="🔍", use_container_width=True)

# ---------------------------------------------------------
# 5. TAMPILAN HASIL PENCARIAN (NEWS LIST)
# ---------------------------------------------------------
if query or submit_button:
    query_vector = get_document_vector(query)
    similarities = cosine_similarity([query_vector], doc_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    st.markdown(f"### Menampilkan {top_k} Berita Paling Relevan untuk: *\"{query}\"*")
    st.markdown("---")
    
    # Loop Menampilkan Kartu Berita
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        title_text = titles[idx]
        link_url = urls[idx]
        
        # Container per artikel
        with st.container():
            col_rank, col_content = st.columns([0.5, 9.5])
            with col_rank:
                st.subheader(f"#{rank}")
            with col_content:
                # Menggunakan HTML tag <a> agar URL saat hover dan klik 100% mengarah ke URL berita asli
                st.markdown(
                    f'<h4 style="margin-bottom: 0px;"><a href="{link_url}" target="_blank" style="text-decoration: none; color: #1f77b4;">{title_text}</a></h4>', 
                    unsafe_allow_html=True
                )
                st.caption(f"🔗 Tautan: [{link_url}]({link_url}) | Relevansi Semantik: **{score:.4f}**")
            st.write("")
            
    # ---------------------------------------------------------
    # 6. EXPANDER BUKA-TUTUP DETAIL TEKNIS (PRESENTASI)
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 Detail Teknis GloVe & Visualisasi Vektor (Klik untuk Presentasi)"):
        st.markdown("#### 1. Skor Cosine Similarity")
        detail_results = []
        for idx in top_indices:
            detail_results.append({
                "Judul Berita": titles[idx],
                "Similarity Score": f"{similarities[idx]:.4f}"
            })
        st.dataframe(pd.DataFrame(detail_results), use_container_width=True)
        
        st.markdown("#### 2. Ekstraksi Vektor GloVe 50-Dimensi")
        words_in_query = [w for w in query.lower().split() if w in glove_model]
        words_missing = [w for w in query.lower().split() if w not in glove_model]
        
        st.write(f"- **Kata terdeteksi di kamus GloVe:** `{words_in_query}`")
        if words_missing:
            st.write(f"- **Kata Out-of-Vocabulary (OOV):** `{words_missing}`")
            
        if words_in_query:
            st.write("**Matriks Vektor Masing-Masing Kata:**")
            vec_dict = {word: glove_model[word] for word in words_in_query}
            st.dataframe(pd.DataFrame(vec_dict).T, use_container_width=True)
            
            st.write("**Hasil Vector Average Kalimat Query:**")
            st.dataframe(pd.DataFrame([query_vector], index=["Vektor Query Rata-rata"]), use_container_width=True)