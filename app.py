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
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline, T5ForConditionalGeneration, T5Tokenizer
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
    """PDF'den metin çıkarma ve temizleme"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Türkçe karakterleri koru ve temizle
                    page_text = page_text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
                    
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
                    
                    # İngilizce kelimeleri kaldır
                    page_text = re.sub(r'\b[a-zA-Z]+\b', '', page_text)
                    
                    # Fazla boşlukları temizle
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + " "
        
        # Minimum metin uzunluğu kontrolü
        if len(text.strip()) < 50:
            raise ValueError("PDF'den yeterli metin çıkarılamadı")
            
        return text.strip()
    except Exception as e:
        print(f"PDF'den metin çıkarma hatası: {str(e)}")
        raise

def preprocess_text(text):
    """Metni ön işleme ve cümlelere ayırma"""
    try:
        print(f"DEBUG: Metin ön işleme başladı. Metin uzunluğu: {len(text)}")
        
        # Metni temizle
        text = re.sub(r'\s+', ' ', text)  # Fazla boşlukları temizle
        text = text.strip()
        
        # Türkçe cümle ayırıcı için özel işlemler
        text = re.sub(r'([.!?])\s+', r'\1\n', text)  # Noktalama işaretlerinden sonra satır sonu ekle
        text = re.sub(r'([.!?])\n+', r'\1\n', text)  # Fazla satır sonlarını temizle
        
        # Cümleleri ayır
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        print(f"DEBUG: İlk ayrıştırma sonrası cümle sayısı: {len(sentences)}")
        
        # Cümleleri filtrele
        filtered_sentences = []
        for sentence in sentences:
            # Türkçe karakter kontrolü
            if re.search(r'[ğüşıöçĞÜŞİÖÇ]', sentence):
                # En az 3 kelime ve 15 karakter içeren cümleleri al
                words = sentence.split()
                if len(words) >= 3 and len(sentence) >= 15:
                    # İngilizce kelime içeren cümleleri atla
                    if not re.search(r'\b[a-zA-Z]+\b', sentence):
                        # Rakam içeren cümleleri atla
                        if not re.search(r'\d', sentence):
                            # Başlık benzeri cümleleri atla
                            if not re.match(r'^[A-ZÇĞİÖŞÜ\s]+$', sentence):
                                filtered_sentences.append(sentence)
        
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

# Uygulama başlatıldığında modeli yükle
print("T5 modeli yükleniyor...")
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Modeli CPU'ya taşı (GPU yoksa)
model = model.to('cpu')

# Özetleme sırasında bellek temizliği
import gc
gc.collect()
torch.cuda.empty_cache()  # GPU kullanıyorsanız

print("T5 modeli başarıyla yüklendi.")

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
            
            # Metni tokenize et
            inputs = tokenizer.encode("özetle: " + chunk, return_tensors="pt", max_length=2048, truncation=True)
            
            # Özet oluştur
            summary_ids = model.generate(
                inputs,
                max_length=min(max_length // len(chunks), 2048),  # Her parça için eşit uzunluk
                min_length=300,  # Minimum uzunluğu artırdık
                length_penalty=0.8,  # Length penalty'yi düşürdük
                num_beams=8,  # Beam sayısını artırdık
                early_stopping=True,
                no_repeat_ngram_size=3,  # Tekrarları önle
                temperature=0.7  # Yaratıcılığı artır
            )
            
            # Özeti decode et
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
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
                num_beams=8,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7
            )
            final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            final_summary = summaries[0]
        
        # Özeti temizle ve düzenle
        final_summary = final_summary.strip()
        final_summary = re.sub(r'\s+', ' ', final_summary)
        final_summary = re.sub(r'\.+', '.', final_summary)
        final_summary = re.sub(r'\.\s*([A-ZÇĞİÖŞÜ])', r'. \1', final_summary)
        
        print(f"DEBUG: T5 özeti oluşturuldu. Uzunluk: {len(final_summary)}")
        return final_summary
        
    except Exception as e:
        print(f"ERROR: T5 özetleme hatası: {str(e)}")
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
            if summary_length < 500 or summary_length > 5000:
                summary_length = 1000
            print(f"DEBUG: Özet uzunluğu: {summary_length}")
        except (ValueError, TypeError):
            summary_length = 1000
            print(f"DEBUG: Özet uzunluğu değeri geçersiz, varsayılan değer kullanılıyor: {summary_length}")
        
        # T5 ile özet oluştur
        try:
            # T5 modeli ile özet oluştur
            summary = create_summary_with_t5(text, max_length=summary_length)
            print(f"DEBUG: T5 özeti oluşturuldu, uzunluk: {len(summary) if summary else 0}")
            
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

if __name__ == '__main__':
    app.run(debug=True)