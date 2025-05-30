from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash, session
import PyPDF2
import io
import os
import re
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import math
import sqlite3
from datetime import datetime, timedelta
import json
from functools import wraps
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import gc

# NLTK kütüphanesini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'ozetly_secret_key'  # Flash mesajları için gerekli

# Uploads klasörünü oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Veritabanı oluştur
def init_db():
    """Veritabanını başlat"""
    conn = None
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # PDF'ler tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS pdfs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT NOT NULL,
                      original_filename TEXT NOT NULL,
                      upload_date TEXT NOT NULL,
                      file_size INTEGER NOT NULL,
                      summary_length INTEGER NOT NULL,
                      file_path TEXT NOT NULL)''')
        
        # Ziyaret istatistikleri tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS visits
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      page_name TEXT NOT NULL,
                      ip_address TEXT NOT NULL,
                      visit_date TEXT NOT NULL)''')
        
        # Admin kullanıcıları tablosu
        c.execute('''CREATE TABLE IF NOT EXISTS admin_users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL)''')
        
        # Varsayılan admin kullanıcısı oluştur
        try:
            c.execute('INSERT OR IGNORE INTO admin_users (username, password) VALUES (?, ?)',
                     ('admin', 'admin123'))
        except sqlite3.IntegrityError:
            print("Admin kullanıcısı zaten mevcut")
        
        conn.commit()
        print("Veritabanı başarıyla başlatıldı")
    except sqlite3.Error as e:
        print(f"Veritabanı başlatma hatası: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# Veritabanını başlat
init_db()

# Admin girişi için decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session or not session['admin_logged_in']:
            flash('Bu sayfaya erişmek için admin girişi yapmalısınız.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Türkçe geçiş ifadeleri
TRANSITION_WORDS = [
    'önce', 'sonra', 'daha sonra', 'ardından', 'bununla birlikte',
    'ayrıca', 'bunun yanında', 'bununla beraber', 'fakat', 'ancak',
    'buna rağmen', 'bununla birlikte', 'dolayısıyla', 'bu nedenle',
    'sonuç olarak', 'özetle', 'kısacası', 'özellikle', 'örneğin',
    'ilk olarak', 'ikinci olarak', 'son olarak', 'birinci', 'ikinci',
    'üçüncü', 'dördüncü', 'beşinci', 'sonuçta', 'özetle', 'kısaca',
    'özetlemek gerekirse', 'sonuç olarak', 'dolayısıyla', 'bu nedenle',
    'bu yüzden', 'bu sebepten', 'bu itibarla', 'bu bakımdan'
]

# Türkçe anahtar kelimeler
KEYWORDS = [
    'önemli', 'öncelikli', 'kritik', 'temel', 'ana', 'başlıca',
    'öncelikle', 'özellikle', 'dikkat', 'dikkat edilmeli',
    'unutulmamalı', 'unutmayın', 'hatırlatma', 'özetle',
    'önemli', 'önemlidir', 'önemli bir', 'önemli bir şekilde',
    'önemli bir nokta', 'önemli bir husus', 'önemli bir konu',
    'önemli bir mesele', 'önemli bir sorun', 'önemli bir problem',
    'önemli bir durum', 'önemli bir olay', 'önemli bir gelişme',
    'önemli bir adım', 'önemli bir karar', 'önemli bir değişiklik',
    'önemli bir yenilik', 'önemli bir buluş', 'önemli bir keşif',
    'önemli bir icat', 'önemli bir gelişme', 'önemli bir ilerleme',
    'önemli bir başarı', 'önemli bir zafer', 'önemli bir kazanç',
    'önemli bir kayıp', 'önemli bir zarar', 'önemli bir tehlike',
    'önemli bir risk', 'önemli bir tehdit', 'önemli bir sorun',
    'önemli bir problem', 'önemli bir mesele', 'önemli bir konu',
    'önemli bir husus', 'önemli bir nokta', 'önemli bir şey',
    'önemli bir durum', 'önemli bir olay', 'önemli bir gelişme'
]

# Türkçe bağlaçlar
CONJUNCTIONS = [
    've', 'veya', 'ama', 'fakat', 'ancak', 'çünkü', 'zira', 'ki',
    'eğer', 'şayet', 'ise', 'ise de', 'ise bile', 'ise de', 'ise de',
    'ise de', 'ise de', 'ise de', 'ise de', 'ise de', 'ise de',
    'ise de', 'ise de', 'ise de', 'ise de', 'ise de', 'ise de',
    'ise de', 'ise de', 'ise de', 'ise de', 'ise de', 'ise de'
]

# Türkçe noktalama işaretleri
PUNCTUATION = string.punctuation + '…—'

# İngilizce kelimeleri tespit etmek için kullanılacak yaygın İngilizce kelimeler
ENGLISH_WORDS = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'it', 'for',
    'on', 'with', 'as', 'at', 'this', 'but', 'they', 'be', 'from',
    'have', 'has', 'had', 'what', 'when', 'where', 'who', 'which',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can',
    'will', 'just', 'should', 'now', 'then', 'there', 'here', 'where',
    'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
    'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
    'its', 'our', 'their', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'doing', 'done', 'can', 'could', 'will', 'would', 'shall',
    'should', 'may', 'might', 'must', 'ought', 'need', 'dare',
    'used', 'about', 'above', 'across', 'after', 'against', 'along',
    'amid', 'among', 'around', 'at', 'before', 'behind', 'below',
    'beneath', 'beside', 'between', 'beyond', 'by', 'down', 'during',
    'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near',
    'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past',
    'since', 'through', 'throughout', 'to', 'toward', 'under',
    'underneath', 'until', 'up', 'upon', 'with', 'within', 'without'
]

def is_english_sentence(sentence):
    """Cümlenin İngilizce olup olmadığını kontrol eder"""
    # Cümleyi küçük harfe çevir
    sentence_lower = sentence.lower()
    
    # İngilizce kelime sayısını hesapla
    english_word_count = sum(1 for word in ENGLISH_WORDS if f" {word} " in f" {sentence_lower} ")
    
    # Cümledeki toplam kelime sayısını hesapla
    total_words = len(sentence_lower.split())
    
    # Eğer cümlede hiç kelime yoksa, İngilizce değildir
    if total_words == 0:
        return False
    
    # Eğer cümledeki İngilizce kelime oranı %30'dan fazlaysa, İngilizce kabul et
    return (english_word_count / total_words) > 0.3

def extract_text_from_pdf(pdf_path):
    """PDF'den metin çıkarma ve temizleme - geliştirilmiş Türkçe karakter desteği"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Türkçe karakterleri koru ve temizle
                    page_text = page_text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
                    
                    # Türkçe karakterler için özel düzeltmeler
                    page_text = re.sub(r'(\s|^)[ıiİI]\s*([^\w\s]|\b)', r'\1ı\2', page_text)  # Tek 'ı' karakteri
                    page_text = re.sub(r'(\s|^)[şsŞS]\s*([^\w\s]|\b)', r'\1ş\2', page_text)  # Tek 'ş' karakteri
                    page_text = re.sub(r'(\s|^)[ğgĞG]\s*([^\w\s]|\b)', r'\1ğ\2', page_text)  # Tek 'ğ' karakteri
                    page_text = re.sub(r'(\s|^)[öoÖO]\s*([^\w\s]|\b)', r'\1ö\2', page_text)  # Tek 'ö' karakteri
                    page_text = re.sub(r'(\s|^)[üuÜU]\s*([^\w\s]|\b)', r'\1ü\2', page_text)  # Tek 'ü' karakteri
                    page_text = re.sub(r'(\s|^)[çcÇC]\s*([^\w\s]|\b)', r'\1ç\2', page_text)  # Tek 'ç' karakteri
                    
                    # Kelimelerin birleştirilmesi (parçalanmış kelimeler için)
                    page_text = re.sub(r'(\w)\s+([ıiİIğĞüÜşŞöÖçÇ])\s+(\w)', r'\1\2\3', page_text)
                    
                    # Fazla boşlukları temizle
                    page_text = re.sub(r'\s+', ' ', page_text)
                    
                    # Başlık ve alt başlıkları kaldır
                    page_text = re.sub(r'^[A-ZÇĞİÖŞÜ\s]+\n', '', page_text)  # Büyük harfli başlıklar
                    page_text = re.sub(r'\n[A-ZÇĞİÖŞÜ\s]+\n', '\n', page_text)  # Alt başlıklar
                    
                    # Yazar bilgilerini kaldır
                    page_text = re.sub(r'^[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?\n', '', page_text)
                    
                    # Sayfa numaralarını ve tarihleri kaldır
                    page_text = re.sub(r'\b\d+\s*$', '', page_text)  # Satır sonundaki sayılar
                    page_text = re.sub(r'^\d+\s*', '', page_text)    # Satır başındaki sayılar
                    page_text = re.sub(r'\d{1,2}\.\d{1,2}\.\d{2,4}', '', page_text)  # Tarihler
                    page_text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', '', page_text)      # Tarihler
                    
                    # Özel karakterleri ve gereksiz metinleri kaldır
                    page_text = re.sub(r'[*#\-_=+]+', '', page_text)  # Özel karakterler
                    page_text = re.sub(r'^\s*[A-ZÇĞİÖŞÜ\s]+\s*$', '', page_text)  # Tek satır başlıklar
                    page_text = re.sub(r'^\s*[a-zçğıöşü\s]+\s*$', '', page_text)  # Tek satır alt başlıklar
                    
                    # Parçalanmış Türkçe kelimeleri birleştir
                    page_text = re.sub(r'([a-zA-ZçğıöşüÇĞİÖŞÜ])\s+\'', r'\1\'', page_text)  # Kesme işaretleri
                    page_text = re.sub(r'"\s+([a-zA-ZçğıöşüÇĞİÖŞÜ])', r'"\1', page_text)  # Tırnak işaretleri
                    
                    # Fazla boşlukları tekrar temizle
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + " "
        
        # Son temizleme işlemleri
        # Noktalama işaretlerinin önünde boşluk olmamasını sağla
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        # Noktalama işaretlerinden sonra boşluk olmasını sağla
        text = re.sub(r'([.,;:!?])\s*([A-ZÇĞİÖŞÜa-zçğıöşü])', r'\1 \2', text)
        
        # Minimum metin uzunluğu kontrolü
        if len(text.strip()) < 50:
            raise ValueError("PDF'den yeterli metin çıkarılamadı")
            
        return text.strip()
    except Exception as e:
        print(f"PDF'den metin çıkarma hatası: {str(e)}")
        raise

def preprocess_text(text):
    """Metni ön işleme ve cümlelere ayırma - geliştirilmiş Türkçe metin desteği"""
    try:
        print(f"DEBUG: Metin ön işleme başladı. Metin uzunluğu: {len(text)}")
        
        # Metni temizle
        text = re.sub(r'\s+', ' ', text)  # Fazla boşlukları temizle
        text = text.strip()
        
        # Parçalanmış Türkçe metni düzelt
        # Tek harfli heceleri birleştir
        text = re.sub(r'(\w)\s+([ıiğüşöçĞÜŞİÖÇ])\s+', r'\1\2', text)
        
        # Anlamsız kısa parçaları düzelt
        common_fragments = ['ın ', 'un ', 'ün ', 'ğı ', 'ığı ', 'liğ', 'lik', 'ler', 'lar', 'dan', 'den', 'tan', 'ten']
        for fragment in common_fragments:
            text = re.sub(r'\s+' + fragment + r'\s+', fragment + ' ', text)
        
        # Tek tırnak işaretlerini düzelt
        text = re.sub(r'\s+\'\s+', '\'', text)
        
        # Noktalama işaretlerini düzelt
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
        
        # Türkçe cümle ayırıcı için özel işlemler
        text = re.sub(r'([.!?])\s+', r'\1\n', text)  # Noktalama işaretlerinden sonra satır sonu ekle
        text = re.sub(r'([.!?])\n+', r'\1\n', text)  # Fazla satır sonlarını temizle
        
        # Cümleleri ayır
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        print(f"DEBUG: İlk ayrıştırma sonrası cümle sayısı: {len(sentences)}")
        
        # Cümleleri filtrele ve temizle
        filtered_sentences = []
        for sentence in sentences:
            # Türkçe karakter kontrolü
            if re.search(r'[ğüşıöçĞÜŞİÖÇ]', sentence):
                # En az 3 kelime ve 10 karakter içeren cümleleri al (eşiği düşürdük)
                words = sentence.split()
                if len(words) >= 3 and len(sentence) >= 10:
                    # Çok fazla tek harfli kelime içeren cümleleri atla
                    single_char_words = sum(1 for word in words if len(word) == 1)
                    if single_char_words / len(words) < 0.3:  # %30'dan az tek harfli kelime
                        # Başlık benzeri cümleleri atla
                        if not re.match(r'^[A-ZÇĞİÖŞÜ\s]+$', sentence):
                            # Cümleyi son kez temizle
                            clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                            filtered_sentences.append(clean_sentence)
        
        print(f"DEBUG: Filtreleme sonrası cümle sayısı: {len(filtered_sentences)}")
        return filtered_sentences
        
    except Exception as e:
        print(f"ERROR: Metin ön işleme hatası: {str(e)}")
        traceback.print_exc()
        return []

def calculate_sentence_scores(sentences):
    """Cümle skorlarını hesapla"""
    if not sentences:
        return []
    
    try:
        # TF-IDF vektörizasyonu
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Kosinüs benzerliği hesapla
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Cümle skorlarını hesapla
        scores = []
        for i, sentence in enumerate(sentences):
            # Temel skor: Cümlenin diğer cümlelerle olan benzerliği
            base_score = np.mean(similarity_matrix[i])
            
            # Ek skorlar
            additional_score = 0
            
            # Geçiş ifadeleri kontrolü
            for word in TRANSITION_WORDS:
                if f" {word} " in f" {sentence.lower()} ":
                    additional_score += 0.2
            
            # Anahtar kelime kontrolü
            for word in KEYWORDS:
                if f" {word} " in f" {sentence.lower()} ":
                    additional_score += 0.3
            
            # Bağlaç kontrolü
            for word in CONJUNCTIONS:
                if f" {word} " in f" {sentence.lower()} ":
                    additional_score += 0.1
            
            # Cümle uzunluğu kontrolü (5-30 kelime arası ideal)
            words = sentence.split()
            if 5 <= len(words) <= 30:
                additional_score += 0.2
            
            # Cümle pozisyonu kontrolü
            if i < len(sentences) * 0.2:  # İlk %20
                additional_score += 0.2
            elif i > len(sentences) * 0.8:  # Son %20
                additional_score += 0.2
            
            # Toplam skor
            total_score = base_score + additional_score
            scores.append(total_score)
        
        return scores
    except Exception as e:
        print(f"Skor hesaplama hatası: {str(e)}")
        traceback.print_exc()
        return [1.0] * len(sentences)

def create_extractive_summary(text, max_length=1000):
    """Çıkarımsal özetleme ile metni özetle"""
    try:
        print("DEBUG: Çıkarımsal özetleme başladı")
        
        # Metni cümlelere ayır
        sentences = preprocess_text(text)
        if not sentences:
            print("ERROR: Cümle ayırma başarısız")
            return "Özet oluşturulamadı: Metin işlenemedi."
            
        print(f"DEBUG: Toplam {len(sentences)} cümle işlenecek")
        
        # Cümle skorlarını hesapla
        scores = calculate_sentence_scores(sentences)
        
        # Cümleleri skorlarına göre sırala, indeksleri koru
        ranked_sentences = [(i, sentences[i], scores[i]) for i in range(len(sentences))]
        ranked_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # İstenen uzunluğa göre seçilecek cümle sayısını belirle
        # Ortalama cümle uzunluğunu hesapla
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        target_sentences = max(1, int(max_length / avg_words_per_sentence))
        
        # En yüksek skorlu cümleleri seç, ancak orijinal sıralamayı koru
        selected_sentences = ranked_sentences[:target_sentences]
        selected_sentences.sort(key=lambda x: x[0])  # Orijinal sıraya göre sırala
        
        # Seçilen cümleleri birleştir
        summary = " ".join([s[1] for s in selected_sentences])
        
        # Özeti temizle
        summary = re.sub(r'\s+', ' ', summary)  # Fazla boşlukları temizle
        summary = re.sub(r'([.,!?;:])\s*([A-ZÇĞİÖŞÜ])', r'\1 \2', summary)  # Noktalama sonrası boşluk ekle
        
        print(f"DEBUG: Çıkarımsal özet oluşturuldu. Uzunluk: {len(summary.split())} kelime")
        return summary
        
    except Exception as e:
        print(f"ERROR: Çıkarımsal özetleme hatası: {str(e)}")
        traceback.print_exc()
        return "Özet oluşturulamadı: Bir hata oluştu."

# Model yükleme işlemini değiştir
print("BERTurk modeli yükleniyor...")
model_name = "dbmdz/bert-base-turkish-cased"  # Türkçe için özel BERT modeli
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Özetleme sırasında bellek temizliği
gc.collect()
torch.cuda.empty_cache()  # GPU kullanıyorsanız

print("BERTurk modeli başarıyla yüklendi.")

def create_summary_with_t5(text, max_length=1000):
    """T5 modeli kullanarak metni özetle"""
    try:
        print("DEBUG: T5 özetleme başladı")
        
        # Metni cümlelere ayır
        sentences = preprocess_text(text)
        if not sentences:
            print("ERROR: Cümle ayırma başarısız")
            return "Özet oluşturulamadı: Metin işlenemedi."
            
        print(f"DEBUG: Toplam {len(sentences)} cümle işlenecek")
        
        # Metni 1000 kelimelik parçalara böl
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > 1000:  # Parça boyutunu artırdık
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(words)
            else:
                current_chunk.append(sentence)
                current_length += len(words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        print(f"DEBUG: Metin {len(chunks)} parçaya bölündü")
        
        # Her parça için özet oluştur
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"DEBUG: Parça {i+1}/{len(chunks)} özetleniyor...")
            
            # Metni temizle - Geliştirilmiş temizleme
            chunk = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', chunk)  # Sadece anlamlı karakterleri tut
            chunk = re.sub(r'\s+', ' ', chunk)  # Fazla boşlukları temizle
            chunk = re.sub(r'([.,!?;:])\s*([A-ZÇĞİÖŞÜ])', r'\1 \2', chunk)  # Noktalama sonrası boşluk ekle
            chunk = re.sub(r'\(\s*\)', '', chunk)  # Boş parantezleri kaldır
            chunk = re.sub(r'["\']{2,}', '"', chunk)  # Tekrarlayan tırnakları temizle
            chunk = re.sub(r'\.{2,}', '.', chunk)  # Tekrarlayan noktaları temizle
            
            # Geliştirilmiş temizleme - Noktalama işaretlerini azalt
            chunk = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', chunk)  # Ardışık noktalama işaretlerini temizle
            chunk = re.sub(r'\(\s*([.,!?;:])\s*\)', r'\1', chunk)  # Parantez içindeki noktalama işaretlerini düzelt
            chunk = re.sub(r'["\']\s*([.,!?;:])\s*["\']', r'\1', chunk)  # Tırnak içindeki noktalama işaretlerini düzelt
            chunk = re.sub(r'([.,!?;:])\s*["\']', r'\1', chunk)  # Noktalama sonrası tırnakları temizle
            chunk = re.sub(r'["\']\s*([.,!?;:])', r'\1', chunk)  # Tırnak sonrası noktalamayı temizle
            
            # Metni tokenize et
            inputs = tokenizer.encode("özetle: " + chunk, return_tensors="pt", max_length=2048, truncation=True)
            
            # Özet oluştur
            summary_ids = model.generate(
                inputs,
                max_length=min(max_length // len(chunks), 2048),  # Her parça için eşit uzunluk
                min_length=300,  # Minimum uzunluğu artırdık
                length_penalty=0.8,  # Length penalty'yi düşürdük
                num_beams=4,  # Beam sayısını artırdık
                early_stopping=True,
                no_repeat_ngram_size=3,  # Tekrarları önle
                temperature=0.7  # Yaratıcılığı artır
            )
            
            # Özeti decode et
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Özeti temizle - Geliştirilmiş temizleme
            summary = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', summary)  # Sadece anlamlı karakterleri tut
            summary = re.sub(r'\s+', ' ', summary)  # Fazla boşlukları temizle
            summary = re.sub(r'([.,!?;:])\s*([A-ZÇĞİÖŞÜ])', r'\1 \2', summary)  # Noktalama sonrası boşluk ekle
            summary = re.sub(r'\(\s*\)', '', summary)  # Boş parantezleri kaldır
            summary = re.sub(r'["\']{2,}', '"', summary)  # Tekrarlayan tırnakları temizle
            summary = re.sub(r'\.{2,}', '.', summary)  # Tekrarlayan noktaları temizle
            
            # Geliştirilmiş temizleme - Noktalama işaretlerini azalt
            summary = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', summary)  # Ardışık noktalama işaretlerini temizle
            summary = re.sub(r'\(\s*([.,!?;:])\s*\)', r'\1', summary)  # Parantez içindeki noktalama işaretlerini düzelt
            summary = re.sub(r'["\']\s*([.,!?;:])\s*["\']', r'\1', summary)  # Tırnak içindeki noktalama işaretlerini düzelt
            summary = re.sub(r'([.,!?;:])\s*["\']', r'\1', summary)  # Noktalama sonrası tırnakları temizle
            summary = re.sub(r'["\']\s*([.,!?;:])', r'\1', summary)  # Tırnak sonrası noktalamayı temizle
            
            summaries.append(summary)
            print(f"DEBUG: Parça {i+1} özetlendi. Uzunluk: {len(summary)}")
        
        # Özetleri birleştir ve son özeti oluştur
        if len(summaries) > 1:
            print("DEBUG: Özetler birleştiriliyor ve son özet oluşturuluyor...")
            combined_summary = " ".join(summaries)
            
            # Son özeti oluştur
            inputs = tokenizer.encode("özetle: " + combined_summary, return_tensors="pt", max_length=2048, truncation=True)
            summary_ids = model.generate(
                inputs,
                max_length=max_length,
                min_length=max_length // 2,  # İstenen uzunluğun en az yarısı kadar
                length_penalty=0.8,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            final_summary = summaries[0]
        
        # Son özeti temizle ve düzenle - Geliştirilmiş temizleme
        final_summary = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', final_summary)  # Sadece anlamlı karakterleri tut
        final_summary = re.sub(r'\s+', ' ', final_summary)  # Fazla boşlukları temizle
        final_summary = re.sub(r'([.,!?;:])\s*([A-ZÇĞİÖŞÜ])', r'\1 \2', final_summary)  # Noktalama sonrası boşluk ekle
        final_summary = re.sub(r'\(\s*\)', '', final_summary)  # Boş parantezleri kaldır
        final_summary = re.sub(r'["\']{2,}', '"', final_summary)  # Tekrarlayan tırnakları temizle
        final_summary = re.sub(r'\.{2,}', '.', final_summary)  # Tekrarlayan noktaları temizle
        
        # Geliştirilmiş temizleme - Noktalama işaretlerini azalt
        final_summary = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', final_summary)  # Ardışık noktalama işaretlerini temizle
        final_summary = re.sub(r'\(\s*([.,!?;:])\s*\)', r'\1', final_summary)  # Parantez içindeki noktalama işaretlerini düzelt
        final_summary = re.sub(r'["\']\s*([.,!?;:])\s*["\']', r'\1', final_summary)  # Tırnak içindeki noktalama işaretlerini düzelt
        final_summary = re.sub(r'([.,!?;:])\s*["\']', r'\1', final_summary)  # Noktalama sonrası tırnakları temizle
        final_summary = re.sub(r'["\']\s*([.,!?;:])', r'\1', final_summary)  # Tırnak sonrası noktalamayı temizle
        
        # Ek temizleme - Tekrarlayan karakterleri azalt
        final_summary = re.sub(r'([a-zA-ZÇĞİÖŞÜçğıöşü])\1{2,}', r'\1', final_summary)  # Tekrarlayan harfleri azalt
        final_summary = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', final_summary)  # Ardışık noktalama işaretlerini temizle
        final_summary = re.sub(r'\(\s*\)', '', final_summary)  # Boş parantezleri kaldır
        final_summary = re.sub(r'\(\s*([.,!?;:])\s*\)', r'\1', final_summary)  # Parantez içindeki noktalama işaretlerini düzelt
        
        final_summary = final_summary.strip()
        
        print(f"DEBUG: T5 özeti oluşturuldu. Uzunluk: {len(final_summary)}")
        return final_summary
        
    except Exception as e:
        print(f"ERROR: T5 özetleme hatası: {str(e)}")
        traceback.print_exc()
        return "Özet oluşturulamadı: Bir hata oluştu."

def create_berturk_extractive_summary(text, max_length=1000):
    """BERTurk modelinin vektör temsillerini kullanarak çıkarımsal özetleme yap"""
    try:
        print("DEBUG: BERTurk çıkarımsal özetleme başladı")
        
        # GPU kontrolü
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Metni cümlelere ayır
        sentences = preprocess_text(text)
        if not sentences:
            print("ERROR: Cümle ayırma başarısız")
            return "Özet oluşturulamadı: Metin işlenemedi."
            
        print(f"DEBUG: Toplam {len(sentences)} cümle işlenecek")
        
        # Her cümle için BERTurk modelini kullanarak vektör temsilleri oluştur
        sentence_embeddings = []
        for sentence in sentences:
            # BERTurk tokenizer ile cümleyi kodla
            inputs = tokenizer(sentence, return_tensors="pt", max_length=512, 
                               truncation=True, padding="max_length").to(device)
            
            # Model ile cümlenin temsilini oluştur (son gizli durumu kullan)
            with torch.no_grad():
                outputs = model(**inputs)
                # Son katmanın hidden state'ini al ve ortalama al
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                sentence_embeddings.append(embedding)
        
        # Cümle vektörleri arasındaki benzerliği hesapla
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                # Kosinüs benzerliği hesapla
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                        np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j]) + 1e-8
                    )
        
        # Cümle skorlarını hesapla
        sentence_scores = []
        for i in range(len(sentences)):
            # Her cümlenin diğer tüm cümlelerle olan benzerliklerinin ortalaması
            score = np.mean(similarity_matrix[i])
            
            # Ek skorlama faktörleri
            # Cümlenin pozisyonu (başta ve sondaki cümlelere daha yüksek puan)
            position_score = 0
            if i < len(sentences) * 0.2:  # İlk %20
                position_score = 0.2
            elif i > len(sentences) * 0.8:  # Son %20
                position_score = 0.1
                
            # Cümle uzunluğu (çok kısa ve çok uzun cümlelere ceza)
            length_score = 0
            word_count = len(sentences[i].split())
            if 5 <= word_count <= 30:
                length_score = 0.1
            
            # Türkçe anahtar kelimeler içeriyorsa bonus
            keyword_score = 0
            for keyword in KEYWORDS:
                if keyword in sentences[i].lower():
                    keyword_score += 0.02
            keyword_score = min(0.2, keyword_score)  # En fazla 0.2 puan
            
            # Geçiş ifadeleri içeriyorsa bonus
            transition_score = 0
            for word in TRANSITION_WORDS:
                if word in sentences[i].lower():
                    transition_score += 0.01
            transition_score = min(0.1, transition_score)  # En fazla 0.1 puan
            
            # Toplam skor
            total_score = score + position_score + length_score + keyword_score + transition_score
            sentence_scores.append(total_score)
        
        # Cümleleri skorlarına göre sırala, indeksleri koru
        ranked_sentences = [(i, sentences[i], sentence_scores[i]) for i in range(len(sentences))]
        ranked_sentences.sort(key=lambda x: x[2], reverse=True)
        
        # İstenen uzunluğa göre seçilecek cümle sayısını belirle
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        target_sentences = max(1, int(max_length / avg_words_per_sentence))
        
        # En yüksek skorlu cümleleri seç
        selected_sentences = ranked_sentences[:target_sentences]
        
        # Cümleleri orijinal sıralamalarına göre yeniden sırala
        selected_sentences.sort(key=lambda x: x[0])
        
        # Seçilen cümleleri birleştir
        summary = " ".join([s[1] for s in selected_sentences])
        
        # Özeti temizle
        summary = re.sub(r'\s+', ' ', summary)  # Fazla boşlukları temizle
        summary = re.sub(r'([.,!?;:])\s*([A-ZÇĞİÖŞÜ])', r'\1 \2', summary)  # Noktalama sonrası boşluk ekle
        
        print(f"DEBUG: BERTurk çıkarımsal özet oluşturuldu. Uzunluk: {len(summary.split())} kelime")
        return summary
        
    except Exception as e:
        print(f"ERROR: BERTurk çıkarımsal özetleme hatası: {str(e)}")
        traceback.print_exc()
        return "Özet oluşturulamadı: Bir hata oluştu."

# Ziyaret istatistiklerini kaydet
def log_visit(page_name):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO visits (page_name, ip_address, visit_date) VALUES (?, ?, ?)",
             (page_name, request.remote_addr, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Ana sayfa"""
    # Session'dan özet ve dosya adını al
    summary = session.pop('summary', None)
    filename = session.pop('filename', None)
    
    return render_template('index.html', summary=summary, filename=filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("DEBUG: Dosya yükleme işlemi başladı")
        
        # Dosya kontrolü
        if 'file' not in request.files:
            print("ERROR: Dosya seçilmedi")
            flash('Dosya seçilmedi.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            print("ERROR: Dosya adı boş")
            flash('Dosya seçilmedi.', 'error')
            return redirect(url_for('index'))
        
        if not file.filename.endswith('.pdf'):
            print("ERROR: Geçersiz dosya formatı")
            flash('Sadece PDF dosyaları kabul edilir.', 'error')
            return redirect(url_for('index'))
        
        # Dosyayı kaydet
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"DEBUG: Dosya kaydedildi: {file_path}")
        
        # PDF'den metin çıkar
        try:
            text = extract_text_from_pdf(file_path)
            print(f"DEBUG: Metin çıkarıldı, uzunluk: {len(text)}")
            if not text:
                raise ValueError("PDF'den metin çıkarılamadı")
        except Exception as e:
            print(f"ERROR: Metin çıkarma hatası: {str(e)}")
            flash('PDF dosyası okunamadı veya boş.', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
        
        # Özet uzunluğunu al
        try:
            summary_length = int(request.form.get('summary_length', 1000))
            if summary_length < 100 or summary_length > 5000:
                summary_length = 1000
            print(f"DEBUG: Özet uzunluğu: {summary_length}")
        except (ValueError, TypeError):
            summary_length = 1000
            print(f"DEBUG: Özet uzunluğu değeri geçersiz, varsayılan değer kullanılıyor: {summary_length}")
        
        # Özet oluştur
        try:
            # BERTurk modelini kullanan çıkarımsal özet oluştur
            summary = create_berturk_extractive_summary(text, max_length=summary_length)
            print(f"DEBUG: BERTurk çıkarımsal özet oluşturuldu, uzunluk: {len(summary) if summary else 0}")
            
            # Özet kontrolü
            if not summary:
                summary = "Özet oluşturulamadı: Metin işlenemedi."
            elif summary == "Özet oluşturulamadı: Metin işlenemedi." or summary == "Özet oluşturulamadı: Bir hata oluştu.":
                raise ValueError("Özet oluşturulamadı")
                
            # Özeti düzenle
            summary = summary.strip()
            summary = re.sub(r'\s+', ' ', summary)
            summary = re.sub(r'\.+', '.', summary)
            summary = re.sub(r'\.\s*([A-ZÇĞİÖŞÜ])', r'. \1', summary)
            
            # Özeti ve dosya adını session'a kaydet
            session['summary'] = summary
            session['filename'] = file.filename
            
            # Başarılı mesajı göster
            flash('PDF başarıyla yüklendi ve özetlendi!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"ERROR: Özet oluşturma hatası: {str(e)}")
            flash('Özet oluşturulurken bir hata oluştu.', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f"KRİTİK HATA: {str(e)}")
        traceback.print_exc()
        flash('Beklenmeyen bir hata oluştu.', 'error')
        return redirect(url_for('index'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin giriş sayfası"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM admin_users WHERE username = ? AND password = ?', (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['admin_logged_in'] = True
            flash('Başarıyla giriş yaptınız.', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Geçersiz kullanıcı adı veya şifre.', 'error')
    
    return render_template('admin/login.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard sayfası"""
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Toplam PDF sayısı
    c.execute('SELECT COUNT(*) FROM pdfs')
    total_pdfs = c.fetchone()[0]
    
    # Son 24 saatteki PDF sayısı
    c.execute('SELECT COUNT(*) FROM pdfs WHERE upload_date >= datetime("now", "-1 day")')
    recent_pdfs = c.fetchone()[0]
    
    # Toplam ziyaret sayısı
    c.execute('SELECT COUNT(*) FROM visits')
    total_visits = c.fetchone()[0]
    
    # Bugünkü ziyaret sayısı
    c.execute('SELECT COUNT(*) FROM visits WHERE visit_date >= date("now")')
    today_visits = c.fetchone()[0]
    
    # Son yüklenen PDF'ler
    c.execute('SELECT * FROM pdfs ORDER BY upload_date DESC LIMIT 5')
    recent_pdf_list = [dict(zip(['id', 'filename', 'original_filename', 'upload_date', 'file_size', 'summary_length'], row))
                      for row in c.fetchall()]
    
    # Sayfa ziyaret istatistikleri
    c.execute('''
        SELECT page_name, COUNT(*) as visit_count, MAX(visit_date) as last_visit 
        FROM visits 
        GROUP BY page_name 
        ORDER BY visit_count DESC
    ''')
    page_stats = [dict(zip(['page_name', 'visit_count', 'last_visit'], row))
                 for row in c.fetchall()]
    
    conn.close()
    
    return render_template('admin/dashboard.html',
                         total_pdfs=total_pdfs,
                         recent_pdfs=recent_pdfs,
                         total_visits=total_visits,
                         today_visits=today_visits,
                         recent_pdf_list=recent_pdf_list,
                         page_stats=page_stats)

@app.route('/admin/pdfs', methods=['GET', 'POST'])
@admin_required
def admin_pdfs():
    """PDF yönetim sayfası"""
    if request.method == 'POST':
        # Seçili PDF'leri sil
        selected_ids = request.form.getlist('selected_pdfs')
        if selected_ids:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            
            # PDF dosyalarını sil
            for pdf_id in selected_ids:
                c.execute('SELECT file_path FROM pdfs WHERE id = ?', (pdf_id,))
                result = c.fetchone()
                if result:
                    file_path = result[0]
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            # Veritabanından kayıtları sil
            c.execute('DELETE FROM pdfs WHERE id IN ({})'.format(','.join('?' * len(selected_ids))), selected_ids)
            conn.commit()
            conn.close()
            
            flash(f'{len(selected_ids)} PDF başarıyla silindi.', 'success')
    
    # PDF listesini getir
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM pdfs ORDER BY upload_date DESC')
    pdfs = [dict(zip(['id', 'filename', 'original_filename', 'upload_date', 'file_size', 'summary_length'], row))
            for row in c.fetchall()]
    conn.close()
    
    return render_template('admin/pdfs.html', pdfs=pdfs)

@app.route('/admin/visits')
@admin_required
def admin_visits():
    """Ziyaret istatistikleri sayfası"""
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Son 7 günün ziyaret istatistikleri
    c.execute('''
        SELECT date(visit_date) as date, COUNT(*) as count 
        FROM visits 
        WHERE visit_date >= date("now", "-7 days")
        GROUP BY date(visit_date)
        ORDER BY date
    ''')
    visit_stats = c.fetchall()
    
    dates = [stat[0] for stat in visit_stats]
    visit_counts = [stat[1] for stat in visit_stats]
    
    # Sayfa bazlı istatistikler
    c.execute('''
        SELECT page_name, COUNT(*) as visit_count, MAX(visit_date) as last_visit 
        FROM visits 
        GROUP BY page_name 
        ORDER BY visit_count DESC
    ''')
    page_stats = [dict(zip(['page_name', 'visit_count', 'last_visit'], row))
                 for row in c.fetchall()]
    
    # Son ziyaretler
    c.execute('''
        SELECT page_name, ip_address, visit_date
        FROM visits
        ORDER BY visit_date DESC 
        LIMIT 10
    ''')
    recent_visits = [dict(zip(['page_name', 'ip_address', 'visit_date'], row))
                    for row in c.fetchall()]
    
    conn.close()
    
    return render_template('admin/visits.html',
                         dates=dates,
                         visit_counts=visit_counts,
                         page_stats=page_stats,
                         recent_visits=recent_visits)

@app.route('/admin/logout')
def admin_logout():
    """Admin çıkış işlemi"""
    session.pop('admin_logged_in', None)
    flash('Başarıyla çıkış yaptınız.', 'success')
    return redirect(url_for('admin_login'))

if __name__ == '__main__':
    app.run(debug=True) 