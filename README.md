# Laporan Proyek Machine Learning - Christofel A Simbolon

## Project Overview

Industri game merupakan salah satu sektor hiburan dengan pertumbuhan tercepat secara global. Berdasarkan laporan [Newzoo](https://newzoo.com/resources/blog/last-looks-the-global-games-market-in-2023), pasar game dunia diperkirakan mencapai lebih dari $207.0 miliar pada tahun 2026. Dengan meningkatnya jumlah game yang tersedia di berbagai platform, pengguna sering mengalami kesulitan dalam menemukan game yang sesuai dengan preferensi mereka.Dalam konteks ini, sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan konten yang relevan, meningkatkan kepuasan pengguna, dan mendorong keterlibatan lebih lanjut.

Berbagai studi telah membuktikan efektivitas sistem rekomendasi dalam industri hiburan digital. Dalam industri game, pendekatan serupa juga semakin lazim, seperti pada platform Steam yang menggunakan sistem rekomendasi berdasarkan histori bermain dan review pengguna. Selain itu, Pérez-Marcos et al [[1]](https://link.springer.com/article/10.1007/s12652-020-01681-0) mengembangkan sistem rekomendasi hibrida yang menggabungkan collaborative filtering dan content-based filtering, serta memanfaatkan durasi bermain sebagai data implisit untuk memperkirakan preferensi pengguna secara lebih akurat
. Sementara itu, studi oleh Pragusma et al.[[2]](https://www.researchgate.net/publication/376539641_Game_Recommendation_Using_Content-based_Algorithm_Title) menunjukkan bahwa metode content-based filtering mampu menghasilkan rekomendasi game dengan tingkat akurasi hingga 82%, dengan memanfaatkan atribut seperti genre, kategori, dan pengembang sebagai dasar pembentukan profil pengguna.

Masalah ini penting untuk diselesaikan karena pengguna kerap mengalami kesulitan dalam menavigasi ribuan game yang tersedia di berbagai platform. Tanpa bantuan sistem rekomendasi, pengguna mungkin melewatkan game yang sesuai dengan minat mereka, yang pada akhirnya dapat menurunkan kepuasan dan keterlibatan. Dengan menghadirkan rekomendasi yang akurat dan personal, platform distribusi game dapat meningkatkan retensi pengguna, memperpanjang durasi interaksi, dan menciptakan pengalaman bermain yang lebih menyenangkan. Selain itu, perusahaan game juga dapat memperoleh manfaat bisnis berupa peningkatan konversi penjualan dan loyalitas pelanggan melalui saran game yang disesuaikan dengan preferensi masing-masing pengguna.

## Business Understanding
Seiring dengan meningkatnya kompleksitas dan jumlah game yang tersedia, dibutuhkan solusi cerdas untuk membantu pengguna dalam menemukan game yang sesuai dengan preferensi mereka. Sistem rekomendasi menjadi kunci dalam menjawab tantangan ini, baik dari sisi pengalaman pengguna maupun dari sisi bisnis penyedia layanan distribusi game.

Proyek ini bertujuan untuk mengembangkan dan membandingkan dua pendekatan sistem rekomendasi utama, yaitu Content-Based Filtering dan Collaborative Filtering, dalam konteks industri game. Kedua pendekatan ini dipilih karena masing-masing memiliki keunggulan dalam menangani berbagai jenis permasalahan yang umum dihadapi oleh sistem rekomendasi, seperti cold start dan relevansi konten.

### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana membangun sistem rekomendasi berbasis konten (Content-Based Filtering) yang dapat menyarankan game serupa berdasarkan genre?
- Bagaimana membangun sistem rekomendasi kolaboratif (Collaborative Filtering) untuk memprediksi rating suatu game oleh pengguna yang belum pernah mencoba game tersebut?
- Bagaimana mengevaluasi dan membandingkan performa kedua pendekatan rekomendasi tersebut?

### Goals
Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi Content-Based Filtering menggunakan fitur genre .
- Membangun model Collaborative Filtering berbasis deep learning menggunakan data interaksi (rating) antara pengguna dan game.
- Menilai performa masing-masing model dilakukan dengan menggunakan berbagai metrik evaluasi seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan akurasi prediksi rating untuk model Collaborative Filtering, serta Precision@k untuk model Content-Based Filtering, guna menentukan metode yang lebih efektif dalam memberikan rekomendasi yang relevan kepada pengguna

### Solution Approach
Untuk menyelesaikan masalah dan mencapai goals di atas, dua pendekatan utama akan digunakan:
- Solution Statement 1: Content-Based Filtering
Pendekatan ini merekomendasikan game berdasarkan kemiripan genre dengan game yang telah disukai atau dimainkan oleh pengguna sebelumnya. Setiap game direpresentasikan berdasarkan informasi genre-nya, sehingga sistem dapat menemukan game lain dengan genre serupa.Pendekatan ini cocok untuk pengguna baru (cold start) karena tidak memerlukan data interaksi dari pengguna lain, hanya membutuhkan histori preferensi genre dari satu pengguna.Rekomendasi yang dihasilkan bersifat personal, karena didasarkan pada kesamaan konten yang relevan dengan minat pengguna.

Teknologi yang digunakan:
TF-IDF Vectorizer: Mengubah teks genre menjadi representasi numerik.Dan Cosine Similarity: Mengukur kemiripan antara representasi genre antar game.

- Solution Statement 2: Collaborative Filtering
Pendekatan ini memberikan rekomendasi berdasarkan pola interaksi pengguna lain yang memiliki preferensi serupa. Sistem menggunakan data rating yang diberikan oleh pengguna terhadap game tertentu untuk mempelajari pola hubungan antara pengguna dan item.Metode ini mampu menangkap preferensi laten yang tidak terlihat hanya dari konten game saja.Pendekatan ini sangat efektif untuk pengguna aktif yang memiliki riwayat rating yang cukup.

Teknologi yang digunakan:
Neural Collaborative Filtering (NCF) berbasis deep learning.Dan Menggunakan embedding, lapisan dense, dan teknik regularisasi untuk memodelkan interaksi non-linear antara pengguna dan game berdasarkan rating.

## Data Understanding
**Informasi Dari Dataset**
| Tipe | Keterangan |
| ------ | ------ |
| Title | Games Metadata & Ratings (5K+ Dataset) |
| Source | [Kaggle](https://www.kaggle.com/datasets/bhanuprakashchegondi/games-metadata-and-ratings-5k-dataset) |
| Maintainer | [Bhanu prakash](https://www.kaggle.com/bhanuprakashchegondi) |
| License | MIT |
| Visibility | Publik |
| Tags | Games,NLP,Data Analytics,Video Games,knn|
| Usability | 5.88 |

Pada Dataset Ini terdapat 2 buah csv yaitu  games_ratings.csv dan  games_metadata_5k.csv
Dengan Struktur sebagai berikut:
1. games_ratings.csv
    - Jumlah baris : 273669 rating
    - Jumlah kolom : 3(game_id,user_id,rating)
    

2. games_metadata_5k.csv 
    - Jumlah baris : 5000 game
    - Jumlah kolom: 10 (game_id,name,description,genres,platforms,rating,released,cover_image,game_link,metacritic_url)

Variabel-variabel pada game_ratings.csv dataset adalah sebagai berikut:
- game_id : Merupakan Id unik untuk masing-masing game.
- user_id : Merupakan Id unik untuk masing-masing pengguna.
- rating : Merupakan nilai yang diberikan masing-masing user untuk game.

Variabel-variabel pada games_metadata_5k.csv dataset adalah sebagai berikut:
- game_id : Merupakan Id unik untuk masing-masing game.
- name : Merupakan nama atau judul dari game
- description : Merupakan deskripsi mengenai isi game,fitur atau cerita singkat.
- genres : Merupakan genre atau kategori dari game tersebut.
- platforms : Merupakan tempat dimana game tersebut tersedia.
- rating : Merupakan nilai yang diberikan oleh pengguna
- released : Merupakan tanggal rilis dari game
- cover_image :Merupakan URL/link ke gambar sampul/cover dari game tersebut.
- game_link:Merupakan URL/link ke halaman game di situs sumber.
- metacritic_url:Merupakan URL ke halaman game di Metacritic

Selanjutnya games_metadata_5k.csv akan dipanggil sebagai df_game dan game_ratings.csv sebagai df_rating.

Dari Kolom-kolom diatas Untuk Content-Based Filtering kita akan memakai dataset games_metadata_5k.csv dan berfokus pada kolom genres yang akan dilakukan TF-IDF.
Untuk Collaborative Filtering kita akan menggunakan dataset game_ratings.csv.

### EDA-Kondisi Data
**1.Cek Missing Value**

Pada df_game ditemukan missing value pada kolom description,genres,platform,released,cover_image dan metacritic_url.Sebagai berikut:

| Kolom           | Missing Values | 
|-----------------|----------------|
| game_id         | 0              | 
| name            | 0              | 
| description     | 30             | 
| genres          | 24             | 
| platforms       | 1              | 
| rating          | 0              | 
| released        | 36             | 
| cover_image     | 4              | 
| game_link       | 0              | 
| metacritic_url  | 30             | 

Pada Model CBF nanti kita hanya akan memakai game_id,name,genre dan rating sehingga kolom lainnya akan di drop pada Data Preparation.Sehingga pada data preparation nanti hanya missing value kolom genre yang akan di imputasi dengan mengisi "unknown".

Pada df_rating tidak ditemukan missing value
| Kolom           | Missing Values | 
|-----------------|----------------|
| game_id         | 0              | 
| user_id         | 0              | 
| rating          | 0              | 

**2.Cek Duplikasi Data**
Baik pada df_game dan df_rating tidak ditemukan duplikasi.

### EDA Univariate dan Multivariate df_game
![Distribusi Rating Game](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/distribusi_rating_game.png?raw=true)
Distribusi dari rating game cenderung normal namun sedikit miring ke kiri dengan pusat distribusi datanya berada di nilai diatas 3,dengan puncak frekuensi sekitar 3.5 sampai 4

![10 Genre Paling Populer](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/10_genre.png?raw=true)
Berdasarkan diagram batang diatas kita bisa melihat bahwa 10 genre terpopuler adalah Action sebanyak kurang dari 3000 diikuti dengan genre indie dan yang terakhir adalah Arcade.

![Distribusi Platform Game](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/platform.png?raw=true)
Diagram diatas memberikan insight bahwa kebanyakan game dapat di platform PC diikuti oleh macos lalu Playstation 4 dan yang paling sedikit adalah android.

![Wordcloud_Description](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/wordcloud.png?raw=true)
Wordcloud menunjukkan menunjukkan bahwa deskripsi game umumnya menekankan pada pengalaman pemain, dengan kata-kata seperti "player", "play", dan "experience" yang muncul secara dominan. Selain itu, aspek eksplorasi juga menjadi tema penting, terlihat dari seringnya kata "world", "explore", dan "level" digunakan.

![Rating VS Tahun rilis](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/rating_tahun.png?raw=true)
Dari scatterplot diatas bisa melihat bahwa jumlah game yang dirilis meningkat secara signifikan dari waktu ke waktu,terutama pada tahun 2010-an.Untuk distribusi rating game dengan rating 4 dan 5 konsisten terus sepanjang waktu namun untuk tidak ditemukan pola antara tahun rilis dengan rating.

![10 Game Rating Tertinggi](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/10_game.png?raw=true)
Barchart diatas menunjukkan 10 Game dengan Rating tertinggi.

### EDA Univariate dan Multivariate df_rating
![Distribusi rating user](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/rating1-5.png?raw=true)

Barchart diatas menujukkan bahwa Rating 1-5 memiliki distribusi yang seimbang.

![Distribusi jumlah Rating per user](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/distribusi_jumlah_rating_user.png?raw=true)
Berdasarkan grafik diatas dapat dilihat bahwa Distribusi jumlah rating per user seimbang.Mayoritas user memberi 25-30 rating.

![20 Game Paling Banyak di rating](https://github.com/Christofel2/sistem-rekomendasi-game/blob/main/images/20_game_rating.png?raw=true)
Berdasarkan diagram kita bisa melihat 20  game_id yang memiliki jumlah rating terbanyak.Yang mana jumlah ratingnya masing-masing sama.

## Data Preparation
### Data Preparation df_game untuk Content Based Filtering
1. Drop Kolom yang Tidak dipakai
```python
#1.Drop kolom yang tidak dipakai
columns_to_drop = ['platforms', 'released','cover_image', 'game_link', 'metacritic_url','year','description']
games_cbf = df_game.drop(columns=columns_to_drop)
```
Kami menghapus kolom-kolom seperti platforms, released, cover_image, game_link, metacritic_url, year, dan description karena informasi di dalamnya tidak cukup relevan dan terlalu banyak noise untuk model Content-Based Filtering yang kami terapkan. Langkah ini juga bertujuan untuk menyederhanakan dimensi dataset agar proses analisis lebih efisien dan fokus hanya pada fitur yang diperlukan.
2. Imputas Missing Values(genres)
```python
#2.Imputasi Missing Values
games_cbf['genres'] = games_cbf['genres'].fillna('Unknown')

# Mengecek kembali Missing Values
print("Missing Values in games_cbf:")
print(games_cbf.isnull().sum())
```
Langkah ini merupakan bagian dari proses data preparation untuk menangani nilai yang hilang (missing values) pada kolom genres. Nilai kosong pada kolom tersebut diisi dengan string 'Unknown' menggunakan metode imputasi sederhana.
Tahapan ini penting dilakukan karena kolom genres merupakan fitur utama dalam pendekatan Content-Based Filtering (CBF). Fitur ini digunakan untuk menghitung kemiripan antar item (dalam hal ini, game) berdasarkan konten atau atribut yang dimiliki. Jika terdapat nilai kosong, proses representasi konten—misalnya menggunakan metode TF-IDF—dapat terganggu dan menghasilkan output yang kurang akurat. Oleh karena itu, memastikan semua data pada kolom genres terisi adalah langkah krusial dalam menyiapkan dataset untuk CBF.
3. Mengubah Genre menjadi List 
```python
#3.Ubah Genre Jadi List
games_cbf['genre_list'] = games_cbf['genres'].apply(lambda x: [g.strip() for g in x.split(',')]) 
```
Langkah ini merupakan bagian dari proses data preparation yang bertujuan untuk mengubah kolom genres dari format string menjadi format list (daftar). Setiap entri genre yang awalnya dipisahkan oleh koma dipecah menjadi elemen-elemen individual dalam list, dan dibersihkan dari spasi yang tidak perlu.

Tahapan ini diperlukan karena dalam metode Content-Based Filtering (CBF), fitur seperti genre akan digunakan untuk membangun representasi konten item (game), misalnya menggunakan metode TF-IDF atau teknik pemrosesan teks lainnya. Dengan format list, proses ekstraksi fitur dan perhitungan kemiripan antar item menjadi lebih terstruktur dan akurat.

4. Ekstraksi Fitur TF-IDF dari Kolom 'genres'
```python
tfidf = TfidfVectorizer(stop_words='english')
# Melakukan perhitungan idf pada games_cbf 'genres'
tfidf.fit(games_cbf['genres'])
tfidf.get_feature_names_out() 
```
Pada tahap ini dilakukan proses data preparation menggunakan teknik TF-IDF Vectorization dengan menghapus stop words bahasa Inggris. Langkah ini bertujuan untuk mengubah data teks pada kolom genres menjadi representasi numerik yang dapat digunakan dalam analisis kesamaan antar game. Pertama, objek TfidfVectorizer diinisialisasi dengan parameter stop_words='english' untuk menghilangkan kata-kata umum yang tidak relevan. Selanjutnya, fungsi fit() diterapkan pada kolom genres untuk menghitung nilai IDF dari setiap genre yang muncul. Setelah proses ini, fitur-fitur unik dari genre dapat diperoleh menggunakan get_feature_names_out(). Tahapan ini penting karena memungkinkan sistem rekomendasi berbasis content-based filtering menghitung kemiripan antar game berdasarkan genre secara lebih akurat.

5. Transformasi Teks 'genres' Menjadi Matriks TF-IDF
```python
tfidf_matrix = tfidf.fit_transform(games_cbf['genres'])
tfidf_matrix.shape
```
Setelah melakukan proses fitting, tahap selanjutnya adalah mengubah data genre menjadi bentuk matriks menggunakan fungsi fit_transform(). Pada baris ini, objek tfidf diterapkan pada kolom genres untuk menghasilkan TF-IDF matrix, yaitu representasi numerik dari setiap game berdasarkan genre-nya. Setiap baris pada matriks merepresentasikan satu game, dan setiap kolom mewakili satu genre unik yang telah diproses sebelumnya. Hasil dari tfidf_matrix.shape menunjukkan dimensi matriks, yang memberikan informasi jumlah game (baris) dan jumlah fitur genre (kolom). Proses ini merupakan bagian penting dari data preparation karena memungkinkan sistem melakukan perhitungan matematis, seperti mengukur kemiripan antar game dalam sistem rekomendasi berbasis konten.

6. Konversi Matriks TF-IDF ke Bentuk Dense (Matriks Penuh)
```python
tfidf_matrix = tfidf.fit_transform(games_cbf['genres'])
tfidf_matrix.shape
```
Setelah mendapatkan TF-IDF matrix, langkah berikutnya adalah mengubah format sparse matrix menjadi bentuk matriks penuh (dense) menggunakan fungsi todense(). Hal ini dilakukan agar isi dari matriks dapat lebih mudah dibaca, ditelusuri, atau divisualisasikan, terutama saat proses eksplorasi data atau debugging. Dalam format dense, setiap nilai menunjukkan bobot TF-IDF suatu genre terhadap sebuah game: semakin tinggi nilainya, semakin penting genre tersebut bagi game itu.

7. Visualisasi Sampel Matriks TF-IDF dalam Bentuk DataFrame
```python
# Membuat DataFrame untuk melihat TF-IDF matrix
# Kolom diisi dengan genre
# Baris diisi dengan nama game

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=games_cbf.name
).sample(22, axis=1).sample(10, axis=0)
```
Langkah ini bertujuan untuk menampilkan hasil TF-IDF pada sebuah DataFrame. Nilai-nilai TF-IDF dari setiap game diubah ke format dense dan kemudian dikonversi menjadi tabel dengan baris berupa nama game dan kolom berupa genre unik yang telah dihasilkan sebelumnya. Dengan menggunakan sample(), ditampilkan sebagian kecil dari data secara acak, yaitu 22 genre (kolom) dan 10 game (baris), agar lebih mudah diamati. Tahapan ini termasuk dalam proses visualisasi dan eksplorasi data setelah data preparation, yang berguna untuk memastikan bahwa hasil transformasi TF-IDF sudah sesuai dan representatif sebelum digunakan dalam sistem rekomendasi.

### Data Preparation df_rating untuk Content Based Filtering
1. Mengubah Nama kolom
```python
#1. Mengubah Nama kolom
df_rating.rename(columns={'user_id': 'userID', 'game_id': 'gameID'}, inplace=True)
```
Langkah ini bertujuan untuk mengubah nama dari kolom user_id menjadi userID dan game_id menjadi gameID,proses ini bertujuan untuk membuat nama kolom lebih konsisten dan memudahkan untuk proses selanjutnya dan mengindari error akibat kesalahan pemanggilan kolom.

2. Encoded UserID
```python
# Mengubah userID menjadi list tanpa nilai yang sama
user_ids = df_rating['userID'].unique().tolist()
print('list userID: ', user_ids)

# Melakukan encoding userID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)

# Melakukan proses encoding angka ke ke userID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)
```
Langkah ini bertujuan untuk melakukan encoding terhadap userID agar data bisa diproses dalam bentuk numerik oleh model. Setiap userID diubah menjadi indeks unik, dan disiapkan pula dictionary kebalikannya untuk proses decoding hasil prediksi kembali ke ID pengguna asli. Encoding ini penting karena algoritma pembelajaran mesin tidak dapat bekerja langsung dengan data string/non-numerik.

3. Encoded gameID
```python
# Mengubah gameID menjadi list tanpa nilai yang sama
game_ids = df_rating['gameID'].unique().tolist()
print('list gameID: ', game_ids)

# Melakukan proses encoding gameID
game_to_game_encoded = {x: i for i, x in enumerate(game_ids)}

# Melakukan proses encoding angka ke gameID
game_encoded_to_game = {i: x for i, x in enumerate(game_ids)} 
```
Langkah ini sama seperti sebelumnya yaitu melakukan encoding terhadap kolom gameID

4. Mapping 
```python
# Mapping userID ke dataframe user
df_rating['user'] = df_rating['userID'].map(user_to_user_encoded)

# Mapping placeID ke dataframe game
df_rating['game'] = df_rating['gameID'].map(game_to_game_encoded)
```
 Melakukan mapping ID(useID,gameID) ke indeks numerik menggunkan hasil encoding sebelumnya,lalu dimasukkan sebagai dataframe kolom baru(user dan game).Tahapan ini penting karena kolom user dan game berisi indeks yang akan di proses deep learning nantinya.
 
5. Mendapatkan jumlah user dan game
```python
# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah game
num_game = len(game_encoded_to_game)
print(num_game)

print('Number of User: {}, Number of Game:'.format(
    num_users, num_game
))
```
Di Tahapan ini,kita menghitung jumlah user dan game berdasarkan hasil encoding sebelumnya,Tahapan ini diperlukan untuk mendefinisikan dimensi input model dan membantu efisiensi memori.

6. Konversi Tipe Data 
```python
# Mengubah rating menjadi nilai float
df_rating['rating'] = df_rating['rating'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(df_rating['rating'])

# Nilai maksimal rating
max_rating = max(df_rating['rating'])
```
Melakukan Perubahan tipe data kolom rating menjadi float untuk keperluan modelling dan menampilkan nilai maksimun dan minimun dari kolom rating.Tahapan ini dilakukan untuk keperluan saat modeling karena membutuhkan input bertipe float dan max dan min rating digunakan untuk normalisasi rating

7. Mengacak dataset
```python
df = df_rating.sample(frac=1, random_state=42)
df 
```
Data pada df_rating diacak menggunakan fungsi .sample,tetapi dalam pengacakannya diterapkan random_state = 42 untuk memastikan tiap pengacakan konsisten dan reproducible. Ini penting dilakukan agar model tidak belajar dari pola urutan data dan mengindari bias.

8. Feature Construction & Normalization (Scaling)
```python
# Membuat variabel x untuk mencocokkan data user dan resto menjadi satu value
x = df[['user', 'game']].values
# Membuat variabel y untuk membuat rating dari hasil
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```
Tahapan ini merupakan pembuatan fitur dan normalisasi target, Hal ini dilakukan untuk memastikan data memiliki format dan skala yang sesuai untuk dimasukkan ke dalam model deep learning

9. Data Splitting 
```python
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
print(x, y)
```
Tahapan ini merupakan pembagian dataset dengan porsi 80:20,tahapan ini sangat penting untuk melatih model berdasarkan 80% data yang ada dan 20% lagi sebagai validasi untuk mencegah overfitting

## Modeling
### 1. Content-Based Filtering
Content-Based Filtering adalah salah satu metode dalam sistem rekomendasi yang merekomendasikan item kepada pengguna berdasarkan kemiripan antara item yang disukai sebelumnya dan item lainnya, berdasarkan fitur atau konten dari item tersebut.
Model Content-Based Filtering ini diimplementasikan dengan menggunakan Cosine Similarity lewat hasil data preparation TF-IDF

**Cara Kerja**:
1. Informasi dari game(yaitu genre game) diolah menjadi vektor dengan menggunakan TF-IDF(Term Frequency-Inverse Document Frequency).Hasilnya akan didapat matriks berukuran 5000 x 22 yang akan menunjukkan bobot masing-masing genre terhadap game.
2. Kemudian dihitung dengan cosine similarity sebagai metrik untuk mengukur kemiripan antar game.Hasil matriksnya adalah 5000 x 5000 yang akan menunjukkan seberapa mirip kedua game berdasarkan genre.
3. Game terdekat/memiliki nilai kemiripan tertinggi akan direkomendasikan ke user.

```python
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
cosine_sim_df = pd.DataFrame(cosine_sim, index=games_cbf['name'], columns=games_cbf['name'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```

**Kelebihan:**
- Tidak memerlukan data dari pengguna lain (cold start)
- Mudah Diinterpretasikan,rekomendasi dapat diberikan lewat kemiripan konten
- Personalisasi yang Baik,mampu merekomendasikan item yang mirip lewat preferensi sebelumnya berdasarkan konten item

**Kekurangan:**
- Keterbatasan Fitur Konten,jika fitur konten tidak lengkap atau kurang representatif (misal, deskripsi kurang jelas, fitur terbatas), performa rekomendasi menurun.
- Tidak Memanfaatkan Data Interaksi Pengguna,tidak mempertimbangkan pola interaksi antar pengguna,sehingga kurang beragam
- Over-Specialization,sistem cenderung merekomendasikan item yang sangat mirip dengan yang sudah pernah user lihat, sehingga kurang memberikan variasi atau eksplorasi item baru (filter bubble).

**Contoh Hasil Rekomendasi:**

Contoh Hasil Rekomendasi (Top-5) untuk Game *"Grand Theft Auto: San Andreas"*
| No | Nama Game                     | Genre  |
|----|------------------------------|--------|
| 1  | Grand Theft Auto V            | Action |
| 2  | Ghostbusters: The Video Game  | Action |
| 3  | Silent Hill 4: The Room       | Action |
| 4  | The Legend of Korra           | Action |
| 5  | A.V.A. Alliance of Valiant Arms | Action |

### 2. Collaborative Filtering
Collaborative Filtering (CF) adalah teknik dalam sistem rekomendasi yang memberikan rekomendasi kepada pengguna berdasarkan preferensi dan perilaku pengguna lain yang mirip. CF tidak memerlukan informasi tentang konten item (seperti deskripsi produk), melainkan hanya menggunakan data interaksi pengguna terhadap item, seperti rating.

**Cara Kerja**:
1. Representasi Vektor:
    - Setiap pengguna dan game diubah menjadi vektor berdimensi tertentu (embedding_size) melalui layer embedding (user_embedding, game_embedding).
    - Vektor ini mewakili fitur laten dari pengguna dan game yang dipelajari selama pelatihan.
2. Penambahan Bias:
    - Model juga mempelajari bias untuk setiap pengguna dan game (user_bias, game_bias), yang menangkap kecenderungan umum seperti pengguna yang sering memberi rating tinggi atau game yang populer.

3. Kalkulasi Interaksi:
    - Interaksi utama antara pengguna dan game dihitung menggunakan perkalian titik (dot product) antara vektor embedding pengguna dan game. Ini menunjukkan seberapa cocok keduanya.

4. Kombinasi Informasi:
    - Vektor pengguna, vektor game, hasil dot product, dan bias digabung menjadi satu tensor besar (concat), yang kemudian diberi dropout untuk mencegah overfitting.

5. Pemrosesan Non-Linear:
    - Tensor gabungan dilewatkan ke beberapa lapisan Dense dengan aktivasi ReLU, Batch Normalization, dan Dropout untuk menangkap hubungan non-linear yang kompleks antara pengguna dan game. 

6. Prediksi Rating:
    - Lapisan output menggunakan fungsi aktivasi sigmoid, menghasilkan nilai dalam rentang 0 hingga 1 yang merepresentasikan rating pengguna terhadap game setelah normalisasi.

7. Pelatihan Model:
    - Model dilatih menggunakan fungsi loss MAE (Mean Absolute Error), yang bertujuan meminimalkan selisih absolut antara prediksi model dan nilai rating sebenarnya dari pengguna terhadap game. 

```python
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class RecommenderNet(Model):
    def __init__(self, num_users, num_games, embedding_size, dropout_rate, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)

        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-4)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.game_embedding = layers.Embedding(
            num_games, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=regularizers.l2(1e-4)
        )
        self.game_bias = layers.Embedding(num_games, 1)

        self.concat_dropout = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)

        self.dense2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)

        self.output_layer = layers.Dense(1, activation='sigmoid') 

    def call(self, inputs, training=False):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        game_vector = self.game_embedding(inputs[:, 1])
        game_bias = self.game_bias(inputs[:, 1])

        dot_product = tf.reduce_sum(user_vector * game_vector, axis=1, keepdims=True)
        concat = tf.concat([user_vector, game_vector, dot_product, user_bias, game_bias], axis=1)
        x = self.concat_dropout(concat, training=training)

        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        return self.output_layer(x)
```

**Kelebihan:**
- Memanfaatkan Data Interaksi Pengguna, mampu menemukan pola preferensi berdasarkan perilaku pengguna lain yang serupa.
- Rekomendasi Lebih Beragam, tidak terbatas pada kesamaan konten, sehingga bisa merekomendasikan item yang berbeda namun relevan berdasarkan kesamaan selera antar pengguna.
- Tidak Memerlukan Fitur Konten, cocok digunakan meskipun informasi konten item terbatas atau tidak tersedia (seperti deskripsi, genre, dll).

**Kekurangan:**
- Masalah Cold Start, sulit memberikan rekomendasi untuk pengguna baru (belum ada riwayat interaksi) atau item baru (belum pernah dinilai oleh pengguna lain).
- Skalabilitas, memerlukan perhitungan antar banyak pengguna atau item, yang bisa menjadi beban komputasi besar seiring bertambahnya data.
- Sparsity, performa bisa menurun jika data interaksi pengguna sangat jarang atau tidak merata, karena sulit menemukan kemiripan yang cukup signifikan.


**Contoh Hasil Rekomendasi:**

### Untuk pengguna dengan ID `user_5077`, sistem menampilkan:
1. Game dengan rating tertinggi yang telah diberikan oleh pengguna  
2. Top 10 rekomendasi game berdasarkan prediksi model

---

### **Game dengan Rating Tertinggi dari Pengguna**
| Judul Game                      | Genre                                                    |
|--------------------------------|----------------------------------------------------------|
| South Park: The Fractured But Whole | Adventure &#124; RPG                                    |
| Drones, The Human Condition    | Action &#124; Shooter &#124; Indie                       |
| Yakuza 4                       | Action                                                   |
| RIDGE RACER Unbounded          | Action &#124; Racing                                     |
| Agony UNRATED                  | Action &#124; Adventure &#124; Indie                     |

---

### **Top-10 Game Rekomendasi untuk User**
| No | Judul Game                                     | Genre                                                   |
|----|------------------------------------------------|----------------------------------------------------------|
| 1  | Borderlands 3                                  | Action &#124; Shooter &#124; Adventure &#124; RPG       |
| 2  | Guns of Icarus Online                          | Action &#124; Simulation &#124; Indie                   |
| 3  | Capsized                                       | Action &#124; Adventure &#124; Indie &#124; Platformer  |
| 4  | Hard Reset Redux                               | Action &#124; Adventure                                 |
| 5  | Caster                                         | Action &#124; Adventure &#124; RPG &#124; Strategy &#124; Casual &#124; Indie |
| 6  | Swag and Sorcery                               | Action &#124; Strategy &#124; Simulation &#124; Indie   |
| 7  | Legends of Eisenwald                           | Adventure &#124; RPG &#124; Strategy &#124; Indie       |
| 8  | Slingshot people                               | Action &#124; Simulation &#124; Casual &#124; Indie     |
| 9  | Dungeons & Dragons: Chronicles of Mystara      | Action &#124; Adventure &#124; RPG                      |
| 10 | Propagation VR                                 | Action                                                  |


## Evaluation
### Evaluasi Content-Based Filtering

Metrik yang digunakan untuk mengevaluasi sistem **Content-Based Filtering** adalah **Precision**.

---

#### Apa itu Precision?

**Precision** mengukur seberapa relevan item yang direkomendasikan oleh sistem kepada pengguna. Precision dihitung dengan rumus:

$$
\text{Precision@k} = \frac{\text{Jumlah item relevan dalam top-}k}{k}
$$

**Keterangan:**
- *Jumlah item relevan dalam top-k*: Jumlah item pada hasil rekomendasi yang juga terdapat pada daftar ground truth.
- *k*: Jumlah item rekomendasi yang dievaluasi (top-k).

---

**Langkah Evaluasi Precision@k:**

1. Sistem memberikan hasil rekomendasi untuk pengguna.
2. Ambil *top-k* dari hasil rekomendasi tersebut.
3. Bandingkan dengan daftar **ground truth** (item yang dianggap relevan).
4. Hitung jumlah item yang relevan dalam top-k.
5. Hitung nilai Precision menggunakan rumus di atas.

---

**Daftar Game Relevan (Ground Truth)**

| No. | Judul Game                     |
|-----|--------------------------------|
| 1   | Grand Theft Auto V             |
| 2   | Grand Theft Auto IV            |
| 3   | Red Dead Redemption            |
| 4   | Ghostbusters: The Video Game   |
| 5   | Silent Hill 4: The Room        |

---

**Daftar Game Rekomendasi**
*Berdasarkan Rekomendasi Game: Grand Theft Auto: San Andreas*
| No. | Judul Game                        |
|-----|-----------------------------------|
| 1   | Grand Theft Auto V                |
| 2   | Ghostbusters: The Video Game      |
| 3   | Silent Hill 4: The Room           |
| 4   | The Legend of Korra               |
| 5   | A.V.A. Alliance of Valiant Arms   |

---
**Perhitungan Precision@5**

- **Top-5 Rekomendasi**: 5 item
- **Item relevan yang muncul dalam top-5**:
  - Grand Theft Auto V 
  - Ghostbusters: The Video Game 
  - Silent Hill 4: The Room 
- **Jumlah item relevan dalam top-5**: 3

**Rumus:**
$$
\text{Precision@5} = \frac{3}{5} = 0.60
$$

---

**Hasil Evaluasi**
Nilai **Precision@5 = 0.60**, yang berarti **60%** dari game yang direkomendasikan oleh sistem termasuk dalam daftar game yang relevan bagi pengguna.

**Kesimpulan:**  
Sistem content-based filtering menunjukkan tingkat akurasi yang cukup baik, dengan lebih dari separuh rekomendasi terbukti relevan. Ini menunjukkan bahwa sistem berhasil memahami preferensi pengguna berdasarkan konten game yang diberikan.

### Evaluasi Collaborative Filtering
**Metrik Evaluasi Collaborative Filtering: MSE & RMSE**

#### Mean Squared Error (MSE)
MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual.

**Rumus MSE:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Dimana:

- $n$ = jumlah sampel (data prediksi)

- $y_i$ = rating aktual ke-i

- $\hat{y}_i$ = rating hasil prediksi ke-i


#### Root Mean Squared Error (RMSE)
RMSE mengukur akar dari rata-rata kuadrat selisih antara nilai prediksi dan aktual.

**Rumus RMSE:**

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}
$$

Dimana:
- $n$ = jumlah sampel (data prediksi)

- $y_i$ = rating aktual ke-i

- $\hat{y}_i$ = rating hasil prediksi ke-i


**Implementasi MSE,RMSE dan Evaluasi Model**
```python
results = model.evaluate(x_val, y_val, verbose=1)
print(f"\n[Hasil Evaluasi terhadap Data Validasi]")
print(f"Loss (MSE): {results[0]:.4f}")
print(f"RMSE      : {results[1]:.4f}")
```

```
1711/1711 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - loss: 0.1365 - root_mean_squared_error: 0.3563
[Hasil Evaluasi terhadap Data Validasi]
Loss (MSE): 0.1369
RMSE      : 0.3569
```
Hasil Evaluasi Collaborative Filtering menunjukkan performa pada data validasi dengan nilai Loss(MSE) sebesar 0.1369 dan RMSE sebesar 0.3569.Nilai MSE tersebut menunjukkan rata-rata kuadrat selisih antara rating aktual dan prediksi cukup kecil, yang berarti model memiliki tingkat kesalahan yang rendah dalam memprediksi rating user terhadap game.Dan Nilai RMSE,artinya secara rata-rata, prediksi rating model hanya meleset sekitar 0.35 poin dari rating asli, yang termasuk cukup baik untuk skala normalized rating (0–1).


## Kesimpulan
Proyek ini berhasil mengembangkan sistem rekomendasi game dengan mengimplementasikan dua algoritma yaitu Content-Based Filtering (CBF) dan Collaborative Filtering (CF).Model Content-Based Filtering mendapatkan performa yang cukup baik dengan precision sebesar 60% namun kedepannya dapat ditingkatkan lagi,sedangkan model Collaborative Filtering
memberikan hasil evaluasi berupa nilai MSE sebesar 0.1369 dan RMSE 0.3569 hasil ini cukup baik namun masih bisa ditingkatkan kedepannya.Kedua Model juga mampu memberikan Top-N Rekomendasi game kepada user.

Untuk saran peningkatan mungkin dapat dicoba dengan menggabungkan kedua algoritma ini menjadi pendekatan hybrid diharapkan dapat menyelesaikan kelemahan masing-masing algoritma sehingga semakin dapat membantu user menemukan game yang cocok untuk mereka.

