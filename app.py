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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
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

def create_summary(text, summary_length=5):
    """Metni özetleme"""
    try:
        print(f"DEBUG: Özet oluşturma başladı. Metin uzunluğu: {len(text)}")
        
        # Metni cümlelere ayır
        sentences = preprocess_text(text)
        print(f"DEBUG: Cümle sayısı: {len(sentences)}")
        
        if not sentences:
            print("ERROR: Özetlenecek cümle bulunamadı")
            return None
        
        # Cümleleri puanla ve grupla
        sentence_scores = {}
        sentence_groups = []
        current_group = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Kelime frekansı ve bağlam puanı
            words = sentence.lower().split()
            for word in words:
                if len(word) > 3:  # Kısa kelimeleri atla
                    # Türkçe kelime kontrolü
                    if re.search(r'[ğüşıöçĞÜŞİÖÇ]', word):
                        # Kelimenin diğer cümlelerde geçme sıklığı
                        frequency = sum(1 for s in sentences if word in s.lower())
                        # Bağlam puanı: Kelime diğer cümlelerde ne kadar yakın geçiyor
                        context_score = sum(1 for s in sentences if word in s.lower() and abs(sentences.index(s) - i) <= 2)
                        score += frequency + (context_score * 2)
            
            # Cümle uzunluğu
            score += len(words) * 0.5
            
            # Cümle pozisyonu
            position_score = 1 - (i / len(sentences))
            score += position_score * 2
            
            # Bağlaç ve geçiş kelimeleri kontrolü
            transition_words = ['ancak', 'fakat', 'ama', 'çünkü', 'dolayısıyla', 'bu nedenle', 
                              'sonuç olarak', 'özetle', 'kısacası', 'özellikle', 'örneğin',
                              'bununla birlikte', 'ayrıca', 'buna ek olarak', 'dahası',
                              'bunun yanında', 'bununla beraber', 'buna rağmen']
            for word in transition_words:
                if word in sentence.lower():
                    score += 2
            
            sentence_scores[sentence] = score
            
            # Cümleleri anlam gruplarına ayır
            if not current_group or (i > 0 and any(word in sentences[i-1].lower() for word in transition_words)):
                if current_group:
                    sentence_groups.append(current_group)
                current_group = [sentence]
            else:
                current_group.append(sentence)
        
        if current_group:
            sentence_groups.append(current_group)
        
        print(f"DEBUG: Cümle grupları oluşturuldu. Grup sayısı: {len(sentence_groups)}")
        
        # Her gruptan en önemli cümleleri seç
        selected_sentences = []
        for group in sentence_groups:
            if group:
                # Grubun en yüksek puanlı cümlesini seç
                best_sentence = max(group, key=lambda s: sentence_scores.get(s, 0))
                selected_sentences.append(best_sentence)
        
        # Özet uzunluğuna göre cümleleri seç
        selected_sentences = selected_sentences[:summary_length]
        print(f"DEBUG: Seçilen cümle sayısı: {len(selected_sentences)}")
        
        # Cümleleri orijinal sırasına göre sırala ve bağlamı güçlendir
        summary = []
        for i, sentence in enumerate(sentences):
            if sentence in selected_sentences:
                # Önceki cümle ile bağlantı kur
                if i > 0 and sentences[i-1] in selected_sentences:
                    # Bağlaç ekle
                    if not any(word in sentence.lower() for word in transition_words):
                        # Cümleler arasındaki mesafeye göre bağlaç seç
                        distance = abs(sentences.index(sentence) - sentences.index(sentences[i-1]))
                        if distance > 2:
                            sentence = "Ayrıca, " + sentence
                        else:
                            sentence = "Bununla birlikte, " + sentence
                
                # Noktalama işaretlerini düzenle
                sentence = sentence.strip()
                
                # Noktalama işaretlerinden sonra boşluk ekle
                sentence = re.sub(r'([.,!?])', r'\1 ', sentence)
                
                # Birden fazla boşluğu tek boşluğa indir
                sentence = re.sub(r'\s+', ' ', sentence)
                
                # Noktalama işaretlerinden önceki boşlukları kaldır
                sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)
                
                # Noktalama işaretlerinden sonraki fazla boşlukları kaldır
                sentence = re.sub(r'([.,!?])\s+', r'\1 ', sentence)
                
                # Cümle sonunda noktalama işareti yoksa nokta ekle
                if not re.search(r'[.!?]$', sentence):
                    sentence += '.'
                
                # Büyük harfle başla
                sentence = sentence[0].upper() + sentence[1:]
                
                summary.append(sentence)
        
        # Özeti birleştir ve temizle
        summary_text = ' '.join(summary)
        
        # Son düzenlemeler
        summary_text = re.sub(r'\s+', ' ', summary_text)  # Fazla boşlukları temizle
        summary_text = re.sub(r'\.+', '.', summary_text)  # Birden fazla noktayı tek noktaya indir
        summary_text = re.sub(r'\.\s*([A-Z])', r'. \1', summary_text)  # Noktadan sonra büyük harf başlat
        summary_text = summary_text.strip()
        
        print(f"DEBUG: Özet oluşturuldu. Özet uzunluğu: {len(summary_text)}")
        return summary_text
        
    except Exception as e:
        print(f"ERROR: Özet oluşturma hatası: {str(e)}")
        traceback.print_exc()
        return None

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
        if 'file' not in request.files:
            flash('Dosya seçilmedi.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi.', 'error')
            return redirect(url_for('index'))
        
        if not file.filename.endswith('.pdf'):
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
        except Exception as e:
            print(f"ERROR: Metin çıkarma hatası: {str(e)}")
            flash('PDF dosyası okunamadı veya boş.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        if not text:
            flash('PDF dosyası okunamadı veya boş.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        # Özet uzunluğunu al
        try:
            summary_length = int(request.form.get('summary_length', 50))
            print(f"DEBUG: Form'dan alınan özet uzunluğu: {summary_length}")
        except ValueError:
            summary_length = 50
            print(f"DEBUG: Özet uzunluğu değeri geçersiz, varsayılan değer kullanılıyor: {summary_length}")
        
        # Özet oluştur
        try:
            summary = create_summary(text, summary_length)
            print(f"DEBUG: Özet oluşturuldu, uzunluk: {len(summary)}")
        except Exception as e:
            print(f"ERROR: Özet oluşturma hatası: {str(e)}")
            flash('Özet oluşturulurken bir hata oluştu.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        if not summary:
            flash('PDF dosyasından özet oluşturulamadı.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        # PDF bilgilerini veritabanına kaydet
        conn = None
        try:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            
            # Dosya boyutunu al
            file_size = os.path.getsize(file_path)
            
            # Veritabanına kaydet
            c.execute("INSERT INTO pdfs (filename, original_filename, upload_date, file_size, summary_length, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                     (filename, file.filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file_size, summary_length, file_path))
            
            conn.commit()
            print(f"DEBUG: Veritabanına kaydedildi. Özet uzunluğu: {summary_length}")
        except sqlite3.Error as e:
            print(f"ERROR: Veritabanı hatası: {str(e)}")
            flash('Veritabanına kaydedilirken bir hata oluştu.', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
        finally:
            if conn:
                conn.close()
        
        # Özeti düzenle ve formatla
        summary = summary.strip()
        summary = re.sub(r'\s+', ' ', summary)
        summary = re.sub(r'\.+', '.', summary)
        summary = re.sub(r'\.\s*([A-Z])', r'. \1', summary)
        
        # Özeti ve dosya adını session'a kaydet
        session['summary'] = summary
        session['filename'] = file.filename
        
        # Başarılı mesajı göster
        flash('PDF başarıyla yüklendi ve özetlendi!', 'success')
        return redirect(url_for('index'))
    
    except Exception as e:
        print(f"KRİTİK HATA: {str(e)}")
        traceback.print_exc()
        flash('Beklenmeyen bir hata oluştu.', 'error')
        return redirect(url_for('index'))

@app.route('/download')
def download():
    log_visit('Özet İndirme')
    summary_path = os.path.join(app.config['UPLOAD_FOLDER'], 'summary.txt')
    return send_file(
        summary_path,
        as_attachment=True,
        download_name='ozet.txt',
        mimetype='text/plain'
    )

# Admin giriş sayfası
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM admin_users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            flash('Başarıyla giriş yaptınız!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Geçersiz kullanıcı adı veya şifre!', 'error')
    
    return render_template('admin/login.html')

# Admin çıkış
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    flash('Başarıyla çıkış yaptınız!', 'success')
    return redirect(url_for('index'))

# Admin paneli ana sayfası
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    log_visit('Admin Paneli')
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Toplam PDF sayısı
    c.execute("SELECT COUNT(*) FROM pdfs")
    total_pdfs = c.fetchone()[0]
    
    # Son 24 saatteki PDF sayısı
    c.execute("SELECT COUNT(*) FROM pdfs WHERE upload_date >= datetime('now', '-1 day')")
    recent_pdfs = c.fetchone()[0]
    
    # Toplam ziyaret sayısı
    c.execute("SELECT COUNT(*) FROM visits")
    total_visits = c.fetchone()[0]
    
    # Bugünkü ziyaret sayısı
    c.execute("SELECT COUNT(*) FROM visits WHERE date(visit_date) = date('now')")
    today_visits = c.fetchone()[0]
    
    # Son yüklenen PDF'ler
    c.execute("""
        SELECT * FROM pdfs 
        ORDER BY upload_date DESC 
        LIMIT 5
    """)
    recent_pdf_list = []
    for row in c.fetchall():
        # Tarihi formatla
        upload_date = row[3]
        if upload_date:
            try:
                upload_date = datetime.strptime(upload_date, '%Y-%m-%d %H:%M:%S')
                upload_date = upload_date.strftime('%d.%m.%Y %H:%M')
            except:
                upload_date = upload_date
        recent_pdf_list.append({
            'id': row[0],
            'filename': row[1],
            'original_filename': row[2],
            'upload_date': upload_date,
            'file_size': row[4],
            'summary_length': row[5]
        })
    
    # Sayfa ziyaret istatistikleri
    c.execute("""
        SELECT page_name, COUNT(*) as visit_count, MAX(visit_date) as last_visit 
        FROM visits 
        GROUP BY page_name 
        ORDER BY visit_count DESC
    """)
    page_stats = []
    for row in c.fetchall():
        # Tarihi formatla
        last_visit = row[2]
        if last_visit:
            try:
                last_visit = datetime.strptime(last_visit, '%Y-%m-%d %H:%M:%S')
                last_visit = last_visit.strftime('%d.%m.%Y %H:%M')
            except:
                last_visit = last_visit
        page_stats.append({
            'page_name': row[0],
            'visit_count': row[1],
            'last_visit': last_visit
        })
    
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
        # Toplu silme işlemi
        pdf_ids = request.form.getlist('pdf_ids')
        if pdf_ids:
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            
            # Seçilen PDF'leri sil
            for pdf_id in pdf_ids:
                # PDF bilgilerini al
                c.execute("SELECT file_path FROM pdfs WHERE id = ?", (pdf_id,))
                result = c.fetchone()
                if result:
                    file_path = result[0]
                    # Dosyayı sil
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    # Veritabanından sil
                    c.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
            
            conn.commit()
            conn.close()
            flash('Seçilen PDF\'ler başarıyla silindi!', 'success')
    
    # PDF listesini getir
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM pdfs ORDER BY upload_date DESC")
    pdfs = []
    for row in c.fetchall():
        pdf = dict(zip(['id', 'filename', 'original_filename', 'upload_date', 'file_size', 'summary_length', 'file_path'], row))
        print(f"DEBUG: PDF ID: {pdf['id']}, Özet Uzunluğu: {pdf['summary_length']}")
        pdfs.append(pdf)
    conn.close()
    
    return render_template('admin/pdfs.html', pdfs=pdfs)

# Ziyaret istatistikleri sayfası
@app.route('/admin/visits')
@admin_required
def admin_visits():
    log_visit('Ziyaret İstatistikleri')
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Son 30 günlük ziyaret istatistikleri
    c.execute("""
        SELECT date(visit_date) as date, COUNT(*) as count 
        FROM visits 
        WHERE visit_date >= datetime("now", "-30 days")
        GROUP BY date(visit_date)
        ORDER BY date
    """)
    daily_stats = c.fetchall()
    
    dates = [row[0] for row in daily_stats]
    visit_counts = [row[1] for row in daily_stats]
    
    # Sayfa ziyaret istatistikleri
    c.execute("""
        SELECT page_name, COUNT(*) as visit_count, MAX(visit_date) as last_visit 
        FROM visits 
        GROUP BY page_name 
        ORDER BY visit_count DESC
    """)
    page_stats = []
    for row in c.fetchall():
        # Tarihi formatla
        last_visit = row[2]
        if last_visit:
            try:
                last_visit = datetime.strptime(last_visit, '%Y-%m-%d %H:%M:%S')
                last_visit = last_visit.strftime('%d.%m.%Y %H:%M')
            except:
                last_visit = last_visit
        page_stats.append({
            'page_name': row[0],
            'visit_count': row[1],
            'last_visit': last_visit
        })
    
    # Son ziyaretler
    c.execute("""
        SELECT * FROM visits 
        ORDER BY visit_date DESC 
        LIMIT 10
    """)
    recent_visits = []
    for row in c.fetchall():
        # Tarihi formatla
        visit_date = row[3]
        if visit_date:
            try:
                visit_date = datetime.strptime(visit_date, '%Y-%m-%d %H:%M:%S')
                visit_date = visit_date.strftime('%d.%m.%Y %H:%M')
            except:
                visit_date = visit_date
        recent_visits.append({
            'id': row[0],
            'page_name': row[1],
            'ip_address': row[2],
            'visit_date': visit_date
        })
    
    conn.close()
    
    return render_template('admin/visits.html',
                         dates=dates,
                         visit_counts=visit_counts,
                         page_stats=page_stats,
                         recent_visits=recent_visits)

@app.route('/admin/download_pdf/<int:pdf_id>')
@admin_required
def download_pdf(pdf_id):
    """PDF indirme işlemi"""
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # PDF bilgilerini al
        c.execute("SELECT file_path, original_filename FROM pdfs WHERE id = ?", (pdf_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            file_path, original_filename = result
            if os.path.exists(file_path):
                return send_file(
                    file_path,
                    as_attachment=True,
                    download_name=original_filename,
                    mimetype='application/pdf'
                )
        
        flash('PDF dosyası bulunamadı!', 'error')
        return redirect(url_for('admin_pdfs'))
    except Exception as e:
        print(f"PDF indirme hatası: {str(e)}")
        flash('PDF indirilirken bir hata oluştu!', 'error')
        return redirect(url_for('admin_pdfs'))

if __name__ == '__main__':
    app.run(debug=True) 