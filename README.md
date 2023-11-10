# Twitter Sentiment Analys
This is Twitter Sentiment Analys for Pilpres 2019 in Indonesia.
**Background**:
Along with the increasing openness of information and freedom of opinion on social media, and
With the increase in political campaign activity on Twitter social media, the General Election Commission (KPU) feels it is necessary to monitor and evaluate campaign activities, especially regarding the sentiments of Twitter users regarding the 2019 election.
**Objective**:
The model aims to classify positive, neutral and negative sentiment in tweets.

## Datasets
**Datset Info**
- Row : 1815
- Column : 2 [”Sentimen”, “Tweet”]
- label : Negatif, Netral. Positif

*Sentimen Count Values*
- Negatif : 596
- Netral : 607
- Positif: 612

## Text Preprocessing
**Text Cleaning**
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

## StopWord Removal
**Tools** NLTK word_tokenize.
'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'kepada', 'oleh', 'saat', 'harus', 'sementara', 'setelah', 'kami', 'sekitar', 'bagi', 'serta', 'di', 'dari', 'telah', 'sebagai', 'masih', 'hal', 'ketika', 'adalah', 'itu', 'dalam', 'bisa', 'bahwa', 'atau', 'hanya', 'kita', 'dengan', 'akan', 'juga', 'ada', 'mereka', 'sudah', 'saya', 'terhadap', 'secara', 'agar', 'lain', 'anda', 'begitu', 'mengapa', 'kenapa', 'yaitu', 'yakni', 'daripada', 'itulah', 'lagi', 'maka', 'tentang', 'demi', 'dimana', 'kemana', 'pula', 'sambil', 'sebelum', 'sesudah', 'supaya', 'guna', 'kah', 'pun', 'sampai', 'sedangkan', 'selagi', 'sementara', 'tetapi', 'apakah', 'kecuali', 'sebab', 'selain', 'seolah', 'seraya', 'seterusnya', 'tanpa', 'agak', 'boleh', 'dapat', 'dsb', 'dst', 'dll', 'dahulu', 'dulunya', 'anu', 'demikian', 'tapi', 'ingin', 'juga', 'nggak', 'mari', 'nanti', 'melainkan', 'oh', 'ok', 'seharusnya', 'sebetulnya', 'setiap', 'setidaknya', 'sesuatu', 'pasti', 'saja', 'toh', 'walau', 'tolong', 'tentu', 'amat', 'apalagi', 'bagaimanapun' .

## Stemming
**Tools** Sastrawi 
StemmerFactory

## Texts Augmentation
**Tools** NLTK, wordnet, random,
- Synonim Replacement
- Sentence Shuffling
- Word Insertion

### Datasets Real + Augmented
**Datset Info**
- Row : 3630
- Column : 2 [”Sentimen”, “Tweet”]
- label : Negatif, Netral. Positif

**Sentimen Count Values**
- Negatif : 1192
- Netral : 1214
- Positif: 1224

## Model Evaluation
- accuracy: 86.50137741046832,
- precision: 0.866572171963956,
- recall: 0.8650137741046832,
- f1: 0.8649952077269386

## Prediction

|index|sentimen|predicted_sentimen|tweet|
|---|---|---|---|
|0|1|1|mau tanya impact buat negara kek gimana kalo gamers fasilitas buat negara mandang gamers indonesia hebat tuh diadain selenggara game internasional indonesia impact apa sponsor banyak kunjung wisatawan tingkat ekonomi maju|
|1|1|1|jokowi raja salman sepakat tingkat kerja sama ekonomi indonesia maju|
|2|0|0|kala tax amnesty kemarin prabowo sama sandiaga uno bayar bergadai tidak ya kok nama paradise papers|
|3|1|1|gimana nih apa gaji tidak jokowi prabowo|
|4|1|1|beranganangan gaji dokter binatang minimal juta camuk prabowosandi|
|5|2|2|uang gaji prabowo sandiuno tidak ambilkami sumbang anak yatim kaum dhuafameleleh air mata dengar prabowo sandi pilih prabowo sandi|
|6|1|1|capres jokowi nasabah usaha mikro pesantren hampir persen wanita yakin program beri mandiri ekonomi keluarga pilpres|
|7|1|1|demokrat minta prabowo harga kontribusi presiden belum|
|8|1|2|orang baik jelas islam baju putih kompak mula satu rasa saling harga hormat lengkap bangsa pikir maju depan jokowi khmaruf amin|
|9|1|1|ps bolakbalik singgung soal deindustrialisasi meski kata tak salah jokowi atas kondisi itudia ujar salah arah pmbangunan ekonomi indonesia belum balik andil presiden sblum jokowi salah sby|
|10|1|1|ini cara tingkat tax ratio berbuat ekonomi guncang ditjen pajak republik indonesia prabowo sandiuno jga ninu donk|


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













