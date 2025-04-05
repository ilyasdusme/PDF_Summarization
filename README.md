# PDF Özetleme Sistemi

Bu uygulama, yüklenen PDF dosyalarını analiz ederek anlamlı özetler oluşturan bir web uygulamasıdır.

## Özellikler

- Modern ve kullanıcı dostu web arayüzü
- Sürükle-bırak dosya yükleme
- PDF dosyası yükleme ve metin çıkarma
- Farklı uzunluklarda özet oluşturma (5, 10 veya 15 sayfa)
- Özetleri metin dosyası olarak indirme
- Hata yönetimi ve kullanıcı geri bildirimi

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:
```bash
python app.py
```

3. Web tarayıcınızda şu adresi açın:
```
http://localhost:5000
```

## Kullanım

1. PDF dosyanızı sürükleyip bırakın veya "PDF Dosyanızı Seçin" alanına tıklayarak dosya seçin
2. İstediğiniz özet uzunluğunu seçin (5, 10 veya 15 sayfa)
3. "Özetle" butonuna tıklayın
4. İşlem tamamlandığında özeti görüntüleyin ve "Özeti İndir" butonu ile kaydedin

## Notlar

- Uygulama, Facebook'un BART-large-cnn modelini kullanarak özetleme yapar
- Büyük PDF dosyaları için işlem süresi uzayabilir
- PDF dosyasının metin içermesi gerekmektedir (taranmış belgeler için OCR gerekebilir)
- Maksimum dosya boyutu 16MB ile sınırlıdır 