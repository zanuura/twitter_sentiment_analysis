![NLP-A Geffry Hinton](https://github.com/zanuura/twitter_sentiment_analys_training/assets/73764446/13bfb94c-a5bb-4bd5-b608-b92e47bcf76d)

# Twitter Sentiment Analys

This is Twitter Sentiment Analys for Pilpres 2019 in Indonesia.
<br>
<br>
**Background**:
Along with the increasing openness of information and freedom of opinion on social media, and
With the increase in political campaign activity on Twitter social media, the General Election Commission (KPU) feels it is necessary to monitor and evaluate campaign activities, especially regarding the sentiments of Twitter users regarding the 2019 election.
<br>
<br>
**Objective**:
The model aims to classify positive, neutral and negative sentiment in tweets.

## Datasets

**Datset Info**

- Row : 1815
- Column : 2 [”Sentimen”, “Tweet”]
- label : Negatif, Netral. Positif

_Sentimen Count Values_

- Negatif : 596
- Netral : 607
- Positif: 612

## What i do

- [Texts Cleaning](#texts-cleaning)
- [Spell Correcting](#spell-correcting)
- [Stopword Removal](#stopword-removal)
- [Texts Augmentation](#texts-augmentation)
- [Vectorization](#vectorization)
- [Embedding](#embedding)
- [Modelling](#modelling)
- [Evaluation](#evaluation)
- [Predicting](#prediction)

## Texts Cleaning

- remove url http, .com .id dll
- remove hashtag #, @
- remove angka
- remove repeating char
- remove double space
- case folding lower

## Spell Correcting

**Example** :

- bgt = banget
- aq = aku
- dmn = dimana
- drpd = daripada
- etc

```python
# Before
aku memang setuju macro ekonominya jokowi jago bgt meski mungkin itu memang rancangan yg sudah ditulis sm ahli yg merancang pembangunan indonesia ya tapi jokowi memang memahami dg baik

# After
aku memang setuju macro ekonominya jokowi jago banget meski mungkin itu memang rancangan yang sudah ditulis sama ahli yang merancang pembangunan indonesia ya tapi jokowi memang memahami dengan baik
```

## StopWord Removal

<!-- **Tools** NLTK word_tokenize.
'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh', 'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'ingin', 'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya', 'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'walau', 'tolong', 'tentu', 'amat', 'apalagi', 'bagaimanapun' . -->

```python
custom_stopwords = ['yah', 'ah', 'yang', 'ke', 'dan']

# Before
dipikirnya pengembangan esport itu main doang aplikasi yang ada di smartphone itu banyak yang mulainya dari main doang terus jadi aplikasi yang banyak manfaatnya kata siapa pengangguran digaji banyak baca buka wawasan jangan mpermalukan generasi z

# After
dipikirnya pengembangan esport itu main doang aplikasi ada di smartphone itu banyak mulainya dari main doang terus jadi aplikasi banyak manfaatnya kata siapa pengangguran digaji banyak baca buka wawasan jangan mpermalukan generasi z
```

<!-- ## Stemming
**Tools** Sastrawi
StemmerFactory -->

## Texts Augmentation

**Metods** NLTK, wordnet, random,

- Synonim Replacement
- Sentence Shuffling
- Word Insertion
- Style Transfer

```python
# Before
kata prabowo indonesia tidak dihargai bangsa asing berita ini pasti palsu buatan penguasa ya kan rockygerung

# After
bercakap prabowo Indonesia tidak dihargai bangsa asing berita ini jelas imitatif buatan pemerintah bahwasanya kan rockygerung
```

### Datasets Real + Augmented

**Datset Info**

- Row : 3630
- Column : 2 [”Sentimen”, “Tweet”]
- label : Negatif, Netral. Positif

**Sentimen Count Values**

- Negatif : 1192
- Netral : 1214
- Positif: 1224

## Vectorization

```python
max_vocab_length = 10000
max_length = 24

text_vectorizer = TextVectorization(
    max_tokens=max_vocab_length,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=max_length)


sample_setence = X_train[0]
text_vectorizer([sample_setence])

<tf.Tensor: shape=(1, 24), dtype=int64, numpy=
array([[ 231,    4,   10,    8,  389,   75,  851,  336,   11,  234,  398,
        3199, 5326,   53,   72, 1699,    0,    0,    0,    0,    0,    0,
           0,    0]], dtype=int64)>


# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:10] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-10:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}")
print(f"Bottom 5 least common words: {bottom_5_words}")

Number of words in vocab: 8184
Top 5 most common words: ['', '[UNK]', 'jokowi', 'ekonomi', 'prabowo', 'bapak', 'belum', 'gaji', 'tidak', 'itu']
Bottom 5 least common words: ['agamakristen', 'agamais', 'aduhhhh', 'adikuasa', 'adaa', 'activitas', 'acaranya', 'absolut', 'abdillah', 'abai']
```

## Embedding

```python
tf.random.set_seed(42)

embedding = layers.Embedding(
    input_dim=max_vocab_length,
    output_dim=64,
    embeddings_initializer="uniform",
    input_length=max_length,
    name="embedding_1")


Original text:
menterinya aja berprestasi apalagi pemimpinnya bisa dikatakan era jokowi lah dimana sejumlah menterinya banyak berlaku kelegaan

Embedded version:

<tf.Tensor: shape=(1, 24, 64), dtype=float32, numpy=
array([[[-0.04715041, -0.04502885,  0.0269253 , ..., -0.04094044,
          0.04728595,  0.02546448],
        [-0.03055869, -0.01405095, -0.04365781, ...,  0.01766321,
          0.00650858, -0.02361989],
        [-0.00188507, -0.00380278, -0.04707987, ..., -0.02362779,
         -0.02642873, -0.01059897],
        ...,
        [-0.03962184, -0.03749093, -0.02066393, ..., -0.03828555,
          0.00698172,  0.02110073],
        [-0.03962184, -0.03749093, -0.02066393, ..., -0.03828555,
          0.00698172,  0.02110073],
        [-0.03962184, -0.03749093, -0.02066393, ..., -0.03828555,
          0.00698172,  0.02110073]]], dtype=float32)>
```

## Modelling

![image](https://github.com/zanuura/twitter_sentiment_analys_training/assets/73764446/b1a46594-2ae2-4b42-865a-46a8e03bd324)

```python
def model4():

  inputs = layers.Input(shape=(1,), dtype="string")
  x = text_vectorizer(inputs)
  layer = embedding(x)
  layer = layers.LSTM(64)(layer)
  layer = layers.Dense(256, name='FC1')(layer)
  layer = layers.Dense(256,)(layer)
  layer = layers.Dense(256,)(layer)
  layer = layers.Dense(256,)(layer)
  layer = layers.Activation('relu')(layer)
  layer = layers.Dropout(0.5)(layer)
  layer = layers.Dense(3, name='out_layer')(layer)
  layer = layers.Activation('softmax')(layer)
  model = tf.keras.Model(inputs=inputs, outputs=layer)
  return model

lstm4 = model4()
lstm4.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
lstm4.summary()

lstm4_history = lstm4.fit(X_train, np.float32(Y_train),
                          epochs=20, # 20
                          batch_size=256, # 256
                          validation_data=(X_test, np.float32(Y_test)),)

{'accuracy': 88.42975206611571,
 'precision': 0.8844814077502243,
 'recall': 0.8842975206611571,
 'f1': 0.8842998589587284}
```

![image](https://github.com/zanuura/twitter_sentiment_analys_training/assets/73764446/77e1ba95-33a4-42a8-88a0-95b52eb1f05b)

![image](https://github.com/zanuura/twitter_sentiment_analys_training/assets/73764446/ca8bf6db-21c2-47cf-9aa3-e72942bf5404)

```
XT_vect = text_vectorizer(X_test)
rfc_score = rfc.score(XT_vect, np.float32(Y_test))
print(rfc_score)

y_pred = rfc.predict(XT_vect)
print(classification_report(Y_test, y_pred))

              precision    recall  f1-score   support

           0       0.88      0.87      0.87       242
           1       0.87      0.87      0.87       256
           2       0.84      0.85      0.84       228

    accuracy                           0.86       726
   macro avg       0.86      0.86      0.86       726
weighted avg       0.86      0.86      0.86       726
```

## Model Evaluation

```python
#1
- accuracy: 86.50137741046832,
- precision: 0.866572171963956,
- recall: 0.8650137741046832,
- f1: 0.8649952077269386
#2
{'accuracy': 88.42975206611571,
 'precision': 0.8844814077502243,
 'recall': 0.8842975206611571,
 'f1': 0.8842998589587284}
```
![image](https://github.com/zanuura/twitter_sentiment_analysis/assets/73764446/c833406f-53e4-42dd-b8da-116bab163c4d)


## Prediction

| index | sentimen | predicted_sentimen | tweet                                                                                                                                                                                                                         |
| ----- | -------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | 1        | 1                  | mau tanya impact buat negara kek gimana kalo gamers fasilitas buat negara mandang gamers indonesia hebat tuh diadain selenggara game internasional indonesia impact apa sponsor banyak kunjung wisatawan tingkat ekonomi maju |
| 1     | 1        | 1                  | jokowi raja salman sepakat tingkat kerja sama ekonomi indonesia maju                                                                                                                                                          |
| 2     | 0        | 0                  | kala tax amnesty kemarin prabowo sama sandiaga uno bayar bergadai tidak ya kok nama paradise papers                                                                                                                           |
| 3     | 1        | 1                  | gimana nih apa gaji tidak jokowi prabowo                                                                                                                                                                                      |
| 4     | 1        | 1                  | beranganangan gaji dokter binatang minimal juta camuk prabowosandi                                                                                                                                                            |
| 5     | 2        | 2                  | uang gaji prabowo sandiuno tidak ambilkami sumbang anak yatim kaum dhuafameleleh air mata dengar prabowo sandi pilih prabowo sandi                                                                                            |
| 6     | 1        | 1                  | capres jokowi nasabah usaha mikro pesantren hampir persen wanita yakin program beri mandiri ekonomi keluarga pilpres                                                                                                          |
| 7     | 1        | 1                  | demokrat minta prabowo harga kontribusi presiden belum                                                                                                                                                                        |
| 8     | 1        | 2                  | orang baik jelas islam baju putih kompak mula satu rasa saling harga hormat lengkap bangsa pikir maju depan jokowi khmaruf amin                                                                                               |
| 9     | 1        | 1                  | ps bolakbalik singgung soal deindustrialisasi meski kata tak salah jokowi atas kondisi itudia ujar salah arah pmbangunan ekonomi indonesia belum balik andil presiden sblum jokowi salah sby                                  |
| 10    | 1        | 1                  | ini cara tingkat tax ratio berbuat ekonomi guncang ditjen pajak republik indonesia prabowo sandiuno jga ninu donk                                                                                                             |

<div align="center" margin=auto>
<a href="https://www.python.org/"><img alt="Static Badge" src="https://img.shields.io/badge/-Python-green?style=flat&logo=python">
</a>
<a href="https://www.python.org/"><img alt="Static Badge" src="https://img.shields.io/badge/-Jupyter-green?style=flat&logo=jupyter">
</a>
<a href="https://www.python.org/"><img alt="Static Badge" src="https://img.shields.io/badge/-Colab-green?style=flat&logo=colab">
</a>
  <br>
  <a href="https://github.com/zanuura"><img alt="Static Badge" src="https://img.shields.io/badge/-Github-white?style=flat&logo=github&logoColor=black">
</a>
<a href="mailto:hammdmob@gmail.com"><img alt="Static Badge" src="https://img.shields.io/badge/-Gmail-white?style=flat&logo=gmail">
</a>
<a href="https://dev.to/zanuura"><img alt="Static Badge" src="https://img.shields.io/badge/-Dev-white?style=flat&logo=dev">
</a>
<a href="https://stackoverflow.com/users/zanuura"><img alt="Static Badge" src="https://img.shields.io/badge/-Stackoverflow-white?style=flat&logo=stackoverflow">
</a>
<a href="https://kaggle.com/hammamkusuma"><img alt="Static Badge" src="https://img.shields.io/badge/-Kaggle-white?style=flat&logo=kaggle">
</a>
<a href="https://instagram.com/hammdmob"><img alt="Static Badge" src="https://img.shields.io/badge/-Instagram-white?style=flat&logo=instagram">
</a>
<a href="https://medium.com/hammam"><img alt="Static Badge" src="https://img.shields.io/badge/-Medium-white?style=flat&logo=medium&logoColor=black">
</a>
</div>
