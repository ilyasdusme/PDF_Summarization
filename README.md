# PDF Özetleme Uygulaması

Bu proje, PDF dosyalarını yükleyip otomatik olarak özetleyen bir web uygulamasıdır. Türkçe metinleri işleyebilme özelliğine sahiptir ve kullanıcı dostu bir arayüz sunar.

## Özellikler

- PDF dosyası yükleme
- Otomatik metin çıkarma
- Türkçe metin işleme
- Özelleştirilebilir özet uzunluğu
- Admin paneli
- Ziyaret istatistikleri
- Toplu PDF silme
- PDF indirme

## Kullanılan Teknolojiler

- **Backend:**
  - Python 3.x
  - Flask (Web Framework)
  - PyPDF2 (PDF işleme)
  - NLTK (Doğal dil işleme)
  - SQLite (Veritabanı)

- **Frontend:**
  - HTML5
  - CSS3
  - Bootstrap 5
  - JavaScript

## Kurulum Adımları

1. Python'u yükleyin (3.x sürümü)
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install flask PyPDF2 nltk
   ```
3. NLTK veri setlerini indirin:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
4. Projeyi çalıştırın:
   ```bash
   python app.py
   ```

## Kullanım

1. Ana sayfada PDF dosyanızı seçin
2. Özet uzunluğunu ayarlayın (1-100 arası)
3. "Özet Oluştur" butonuna tıklayın
4. Oluşturulan özeti görüntüleyin

## Admin Paneli

Admin paneline erişmek için:
- Kullanıcı adı: admin
- Şifre: admin123

Admin panelinde yapabilecekleriniz:
- PDF'leri görüntüleme
- Toplu silme
- İndirme
- Ziyaret istatistiklerini görüntüleme

## Proje Yapısı

```
Pdf_Summarize/
├── app.py                 # Ana uygulama dosyası
├── database.db           # Veritabanı dosyası
├── static/              # Statik dosyalar
│   ├── css/
│   │   └── style.css    # Stil dosyası
│   └── js/
│       └── main.js      # JavaScript dosyası
├── templates/           # HTML şablonları
│   ├── index.html      # Ana sayfa
│   └── admin/          # Admin paneli şablonları
│       ├── dashboard.html
│       ├── login.html
│       ├── pdfs.html
│       └── visits.html
└── uploads/            # Yüklenen PDF'lerin saklandığı klasör
```

## Özelliklerin Detaylı Açıklaması

### PDF İşleme
- PDF'lerden metin çıkarma
- Türkçe karakter desteği
- Otomatik temizleme ve formatlama

### Özetleme Algoritması
- Cümle puanlama sistemi
- Bağlam analizi
- Geçiş kelimeleri desteği
- Türkçe metin optimizasyonu

### Güvenlik
- Dosya tipi kontrolü
- Güvenli dosya adı oluşturma
- Admin paneli koruması

### Veritabanı
- PDF bilgilerini saklama
- Ziyaret istatistikleri
- Admin kullanıcı yönetimi

## Geliştirme

Projeyi geliştirmek için:
1. Yeni özellikler ekleyebilirsiniz
2. Arayüzü özelleştirebilirsiniz
3. Özetleme algoritmasını geliştirebilirsiniz
4. Performans iyileştirmeleri yapabilirsiniz

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 