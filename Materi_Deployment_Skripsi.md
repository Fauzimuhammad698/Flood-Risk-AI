# DOKUMENTASI PROSES DEPLOYMENT SISTEM PREDIKSI RISIKO BANJIR AI
*(Bahan Materi Penulisan Bab Implementasi dan Deployment Skripsi)*

Dokumen ini berisi materi lengkap mengenai proses deployment (penyebaran) sistem aplikasi **Prediksi Risiko Banjir AI** berbasis **Streamlit**, **Google Earth Engine (GEE)**, dan **Machine Learning**. Materi ini disusun secara sistematis dan akademis agar dapat langsung diadaptasi ke dalam penulisan **Bab IV (Implementasi)** atau **Bab V (Pengujian & Deployment)** pada draf skripsi Anda.

---

## DAFTAR ISI
1. [Tinjauan Umum Infrastruktur Deployment](#1-tinjauan-umum-infrastruktur-deployment)
2. [Arsitektur Deployment dan Aliran Data](#2-arsitektur-deployment-dan-aliran-data)
3. [Manajemen Dependensi dan Sistem Environment](#3-manajemen-dependensi-dan-sistem-environment)
4. [Mekanisme Keamanan dan Secrets Management (Google Earth Engine)](#4-mekanisme-keamanan-dan-secrets-management-google-earth-engine)
5. [Langkah-Langkah Implementasi Deployment (GitHub & Streamlit Cloud)](#5-langkah-langkah-implementasi-deployment-github--streamlit-cloud)
6. [Evaluasi dan Validasi Pasca-Deployment](#6-evaluasi-dan-validasi-pasca-deployment)

---

## 1. Tinjauan Umum Infrastruktur Deployment

Dalam penelitian ini, aplikasi dideploy menggunakan model **Platform as a Service (PaaS)** melalui **Streamlit Community Cloud** yang diintegrasikan secara langsung dengan repositori kontrol versi **GitHub**.

### Spesifikasi Infrastruktur Deployment:
*   **Penyedia Layanan Cloud:** Streamlit Community Cloud (didukung oleh kontainerisasi berbasis Linux/Debian).
*   **Sistem Kontrol Versi (VCS):** Git dengan repositori host di GitHub.
*   **Metode sinkronisasi:** *Continuous Deployment (CD)*, di mana setiap perubahan kode (*commit/push*) pada branch utama (`main`/`master`) di GitHub akan memicu build otomatis di cloud.
*   **Layanan API Geospasial:** Google Earth Engine (GEE) API melalui akun layanan (*Service Account*) GCP.
*   **Layanan API Cuaca:** Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) API & Open-Meteo API.

---

## 2. Arsitektur Deployment dan Aliran Data

Sistem berjalan secara serverless (tanpa pengelolaan server fisik oleh pengembang) dengan membagi tugas pemrosesan antara client-side (browser pengguna), web application server (Streamlit Cloud), dan external data providers (API).

### Diagram Aliran Data Sistem (Data Flow Diagram):

```mermaid
graph TD
    User([Pengguna / Client Browser]) <-->|1. Input Lokasi & Request Analisis| StreamlitApp[Streamlit Web App Server]
    
    subgraph Data Acquisition & Fallback (3-Tier)
        StreamlitApp -->|2a. Geocoding Query| GeopyAPI[Geopy - Nominatim API]
        GeopyAPI -->|Koordinat Lat/Lon| StreamlitApp
        
        StreamlitApp -->|2b. Priority 1: adm4 Code| BMKGForecast[BMKG Forecast API]
        StreamlitApp -->|2c. Priority 2: Coordinate Alert| BMKGNowcast[BMKG Nowcast Alert API]
        StreamlitApp -->|2d. Priority 3: Fallback Forecast| OpenMeteo[Open-Meteo API]
    end
    
    subgraph Earth Engine API (Real-time Spatial)
        StreamlitApp -->|3. Auth Service Account| GEE[Google Earth Engine API]
        GEE -->|SRTM: Slope Data| StreamlitApp
        GEE -->|Sentinel-2: NDVI Data| StreamlitApp
        GEE -->|ESA WorldCover: Land Cover| StreamlitApp
    end
    
    subgraph Machine Learning Pipeline
        StreamlitApp -->|4. Load Model| PKL[final_production_flood_model.pkl]
        StreamlitApp -->|5. Skala Fitur| Scaler[Standard Scaler]
        StreamlitApp -->|6. Prediksi Probabilitas| ModelML[Model Machine Learning / XGBoost]
    end
    
    StreamlitApp -->|7. Evaluasi Skor Akhir| HybridScorer[Hybrid Scorer Algorithm: 80% Fisik + 20% ML]
    StreamlitApp -->|8. Visualisasi Spasial| Folium[Folium Map & SHAP Explainer]
    Folium -->|9. Output Laporan Risiko & Peta Interaktif| User
```

---

## 3. Manajemen Dependensi dan Sistem Environment

Untuk memastikan aplikasi dapat berjalan dengan konsisten pada server produksi (Streamlit Cloud) sebagaimana pada server lokal (localhost), dikonfigurasikan dua jenis file konfigurasi dependensi:

### A. Dependensi Runtime Python (`requirements.txt`)
File [requirements.txt](file:///d:/SKRIPSI%20BISMILLAH/FLOOD%20RISK%20AI/requirements.txt) menentukan versi spesifik dari pustaka Python yang digunakan agar menghindari konflik sintaksis:
1.  **`numpy==2.0.2` & `pandas==2.2.2`:** Digunakan untuk memanipulasi data tabular dan larik numerik hasil ekstrak spasial.
2.  **`scikit-learn==1.6.1` & `xgboost==3.2.0`:** Berfungsi memuat model prediktif (Random Forest/XGBoost) dan penskalaan data (`scaler`).
3.  **`earthengine-api==1.7.21`:** Pustaka resmi Google untuk mengirim query spasial ke server Earth Engine.
4.  **`folium==0.20.0` & `streamlit-folium==0.27.1`:** Pustaka visualisasi peta interaktif berbasis Leaflet.js di dalam antarmuka Streamlit.
5.  **`geopy==2.4.1`:** Pustaka bantu pencarian koordinat (geocoding) berbasis nama wilayah.

### B. Dependensi Sistem Operasi Linux (`packages.txt`)
Karena server Streamlit Cloud berjalan di atas kontainer Debian minimalis, beberapa paket Python dengan pustaka C/C++ internal memerlukan kompilasi sistem tingkat OS. Konfigurasi didefinisikan pada [packages.txt](file:///d:/SKRIPSI%20BISMILLAH/FLOOD%20RISK%20AI/packages.txt):
*   `build-essential` & `python3-dev`: Menyediakan compiler GCC untuk membuild extension biner Python.
*   `libssl-dev` & `libffi-dev`: Diperlukan oleh pustaka kriptografi untuk mengamankan pertukaran token HTTPS ke API GEE dan BMKG.

---

## 4. Mekanisme Keamanan dan Secrets Management (Google Earth Engine)

### A. Risiko Kebocoran Kredensial (Credential Leakage)
Aplikasi menggunakan **Google Earth Engine (GEE)** yang memerlukan otentikasi kunci pribadi (*private key*) berupa berkas JSON dari Google Cloud Platform Service Account. Berkas JSON ini bersifat sangat rahasia. **Sangat dilarang** untuk mengunggah berkas JSON mentah tersebut ke repositori publik GitHub karena dapat disalahgunakan pihak asing.

### B. Solusi: Streamlit Secrets Management (TOML Format)
Sebagai solusi keamanan, kunci otentikasi disimpan di dalam konfigurasi variabel lingkungan terenkripsi (*encrypted environment variables*) milik Streamlit Cloud. Kunci JSON diekstrak dan dikonversi ke dalam representasi string tunggal (TOML).

Program menyediakan skrip pembantu [generate_secrets.py](file:///d:/SKRIPSI%20BISMILLAH/FLOOD%20RISK%20AI/generate_secrets.py) yang mengonversi berkas JSON kredensial menjadi format TOML aman:
```python
# Potongan logika konversi pada generate_secrets.py
json_str = json.dumps(gee_key, separators=(',', ':'))
toml_method1 = f'''[gee]
json_key = "{json_str}"'''
```

Terdapat dua strategi otentikasi yang diimplementasikan pada kode [app.py](file:///d:/SKRIPSI%20BISMILLAH/FLOOD%20RISK%20AI/app.py#L171-L284) untuk menjamin keberhasilan koneksi ke server GEE di cloud:

1.  **Strategi 1: Single-Line JSON (Direkomendasikan)**
    Nilai berkas JSON dibaca sebagai sebuah string JSON terkompresi tanpa spasi kosong dan disimpan dalam variabel `st.secrets["gee"]["json_key"]`.
2.  **Strategi 2: Base64 Encoded JSON (Cadangan)**
    JSON diubah menjadi representasi biner Base64 untuk menghindari masalah pemotongan karakter spesial atau baris baru (*newline*) pada editor cloud.

Di server produksi, fungsi `initialize_gee()` menginisialisasi kredensial tersebut secara dinamis:
```python
from google.oauth2 import service_account
import ee

# Inisialisasi kredensial dari secrets cloud
gee_key = json.loads(st.secrets["gee"]["json_key"])
credentials = service_account.Credentials.from_service_account_info(
    gee_key,
    scopes=['https://www.googleapis.com/auth/earthengine.readonly']
)
ee.Initialize(credentials, project='deteksi-banjir-492803')
```

---

## 5. Langkah-Langkah Implementasi Deployment (GitHub & Streamlit Cloud)

Berikut adalah panduan operasional langkah-demi-langkah deployment yang dapat ditulis ulang sebagai prosedur penelitian:

### Langkah 1: Inisialisasi Repositori Git dan Push ke GitHub
1.  Buat berkas `.gitignore` untuk mencegah berkas sensitif terupload:
    ```text
    .venv/
    __pycache__/
    *.json
    .streamlit/secrets.toml
    ```
    *(Catatan: `*.json` memastikan file `deteksi-banjir-492803-7fc098068802.json` tidak bocor ke publik)*
2.  Lakukan inisialisasi git local dan kirim ke GitHub:
    ```bash
    git init
    git add .
    git commit -m "Initialize Flood Risk AI production version"
    git remote add origin https://github.com/UsernameAnda/Flood-Risk-AI.git
    git branch -M main
    git push -u origin main
    ```

### Langkah 2: Registrasi dan Koneksi Akun di Streamlit Community Cloud
1.  Buka situs [Streamlit Share](https://share.streamlit.io/).
2.  Pilih opsi **Sign In** dan hubungkan menggunakan akun **GitHub** yang memuat repositori proyek tersebut.
3.  Berikan otorisasi akses (*Authorized OAuth*) agar Streamlit dapat membaca repositori Anda.

### Langkah 3: Konfigurasi Parameter Aplikasi Baru (Deploy App)
Di halaman dashboard Streamlit, klik tombol **"Create app"** (atau **"Deploy an app"**) dan isi formulir konfigurasi sebagai berikut:
*   **Repository:** `UsernameAnda/Flood-Risk-AI`
*   **Branch:** `main`
*   **Main file path:** `app.py`
*   **App URL (Opsional):** Buat alamat URL kustom sesuai judul skripsi (misal: `https://flood-risk-ai.streamlit.app`).

### Langkah 4: Pengisian Secrets Terenkripsi
Sebelum menekan tombol deploy, konfigurasikan akses rahasia GEE:
1.  Klik menu **Advanced Settings** di bagian bawah formulir deployment.
2.  Pilih tab **Secrets**.
3.  Buka berkas lokal [.streamlit/secrets_method1.toml](file:///d:/SKRIPSI%20BISMILLAH/FLOOD%20RISK%20AI/.streamlit/secrets_method1.toml) yang telah di-generate sebelumnya oleh script `generate_secrets.py`.
4.  Salin seluruh teks konfigurasi di dalamnya, lalu tempel (*paste*) ke kolom input Secrets Streamlit Cloud.
5.  Klik tombol **Save**.

### Langkah 5: Peluncuran Aplikasi (Deployment Processing)
1.  Klik **Deploy!**
2.  Server Streamlit Cloud akan memulai proses instalasi OS dependencies (`packages.txt`) kemudian memasang pustaka Python (`requirements.txt`).
3.  Proses instalasi dapat dipantau melalui panel **Manage App Logs** di sisi kanan bawah layar.
4.  Jika sukses, aplikasi akan menampilkan antarmuka web interaktif secara publik.

---

## 6. Evaluasi dan Validasi Pasca-Deployment

Setelah proses deployment selesai, dilakukan uji fungsionalitas untuk memastikan integrasi API dan model machine learning berjalan normal pada server awan.

### Lembar Hasil Uji Ulang Produksi:

| Skenario Uji | Prosedur Pengujian | Ekspektasi Output | Status Hasil |
| :--- | :--- | :--- | :--- |
| **Akses Antarmuka** | Membuka URL `https://flood-risk-ai.streamlit.app` | Aplikasi memuat stylesheet CSS kustom dengan font *Inter* dan background gradien premium. | **BERHASIL** |
| **Otentikasi GEE** | Server membaca Secrets dari `st.secrets` | Menampilkan banner biru bertuliskan `"🛰️ Google Earth Engine: TERHUBUNG"`. | **BERHASIL** |
| **Geocoding Lokasi** | Menginput nama daerah `"Jakarta"` pada kolom pencarian dan klik "Analisis Risiko" | Menemukan koordinat geografis Lat: `-6.2088°` dan Lon: `106.8456°`. | **BERHASIL** |
| **3-Tier Weather Fallback** | Sistem mengirimkan request ke API BMKG & Open-Meteo | Data curah hujan terisi dengan label sumber data yang valid (misal: `Sumber: BMKG-Forecast` atau `Open-Meteo`). | **BERHASIL** |
| **Spasial Extraction GEE** | Sistem memotong raster SRTM & Sentinel-2 pada koordinat target | Mengembalikan nilai kemiringan lereng (*slope*), kerapatan vegetasi (*NDVI*), dan kelas tutupan lahan (*land cover*). | **BERHASIL** |
| **Model ML Prediction** | File model `.pkl` dibaca di memori server dan memproses data input | Menghasilkan probabilitas prediksi antara `0.0` sampai `1.0` secara instan. | **BERHASIL** |
| **Hybrid Scorer & Map** | Penggabungan persentase skor fisik (80%) dan model (20%) | Peta Folium memuat visualisasi zona risiko banjir lingkaran radius 1km & 2km berwarna dinamis (Hijau/Kuning/Merah). | **BERHASIL** |

---
*Materi di atas dapat digunakan untuk menyusun sub-bab **"Penerapan dan Pengujian Sistem"** pada laporan skripsi Anda.*
