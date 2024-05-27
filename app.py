import pandas as pd
import streamlit as st
import joblib
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import mysql.connector
from datetime import datetime

# Lakukan unduhan NLTK di awal skrip
nltk.download('stopwords')
nltk.download('punkt')

# Membaca model yang sudah dilatih
logreg_model = joblib.load("model100.pkl")

# Inisialisasi objek TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Membaca data dari file CSV
df = pd.read_csv("data.csv")

# Memisahkan fitur (X) dan label (y)
X = df['Text']
y = df['Human']

# Memisahkan data menjadi data pelatihan (training) dan data pengujian (testing) dengan rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi objek TF-IDF Vectorizer dan melakukan fit_transform pada data pelatihan
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Fungsi untuk membersihkan teks
def clean_text(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = text.lower()
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in cleaned_words]
    return " ".join(stemmed_words)

# Fungsi untuk melakukan klasifikasi teks
def classify_text(input_text):
    cleaned_text = clean_text(input_text)
    input_vector = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = logreg_model.predict(input_vector)[0]
    return predicted_label

# Fungsi untuk menyimpan hasil analisis ke dalam database
def save_to_database(input_text, result):
    try:
        # Konfigurasi koneksi ke database
        connection = mysql.connector.connect(
            host='localhost',       # Ganti dengan host MySQL Anda
            user='root',            # Ganti dengan username MySQL Anda
            password='password',    # Ganti dengan password MySQL Anda
            database='scentplus',   # Nama database
            port=3306               # Pastikan port sudah benar
        )
        cursor = connection.cursor()
        
        # Mendapatkan waktu saat ini
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Membuat query untuk menyimpan data
        query = "INSERT INTO riwayat (text, hasil, date) VALUES (%s, %s, %s)"
        values = (input_text, result, current_time)
        
        # Menjalankan query
        cursor.execute(query, values)
        
        # Commit perubahan
        connection.commit()
        
        # Menutup koneksi
        cursor.close()
        connection.close()
        
        return True
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return False

# Streamlit UI
st.title("Aplikasi Analisis Sentimen Scentplus")
input_text = st.text_input("Masukkan kalimat untuk analisis sentimen:")
if st.button("Analisis"):
    if input_text.strip() == "":
        st.error("Tolong masukkan sentimen terlebih dahulu.")
    else:
        result = classify_text(input_text)
        st.write("Hasil Analisis Sentimen:", result)
        
        # Simpan hasil analisis ke dalam database
        if save_to_database(input_text, result):
            st.success("Hasil analisis berhasil disimpan ke dalam database.")
        else:
            st.error("Gagal menyimpan hasil analisis ke dalam database.")