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
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # PDF'ler tablosu
    c.execute('''CREATE TABLE IF NOT EXISTS pdfs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  original_filename TEXT NOT NULL,
                  upload_date TEXT NOT NULL,
                  file_size INTEGER NOT NULL,
                  summary_length INTEGER NOT NULL)''')
    
    # Ziyaret istatistikleri tablosu - Eski tabloyu sil ve yeniden oluştur
    c.execute('DROP TABLE IF EXISTS visits')
    c.execute('''CREATE TABLE visits
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
    c.execute('INSERT OR IGNORE INTO admin_users (username, password) VALUES (?, ?)',
             ('admin', 'admin123'))  # Gerçek uygulamada güvenli bir şifre kullanın
    
    conn.commit()
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
    """PDF'den metin çıkarma"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"PDF okuma hatası: {str(e)}")
        return None

def preprocess_text(text):
    """Metni ön işleme"""
    # Gereksiz boşlukları temizle
    text = ' '.join(text.split())
    # Noktalama işaretlerini düzenle
    text = text.replace(' .', '.').replace(' ,', ',').replace(' :', ':')
    return text

def calculate_sentence_scores(sentences):
    """Cümle skorlarını hesapla"""
    if not sentences:
        return []
    
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
        
        # İngilizce cümle kontrolü - İngilizce cümleleri daha düşük puan ver
        if is_english_sentence(sentence):
            additional_score -= 0.5  # İngilizce cümleleri önemli ölçüde düşük puanla
        
        # Geçiş ifadeleri kontrolü
        for word in TRANSITION_WORDS:
            if word in sentence.lower():
                additional_score += 0.3
        
        # Anahtar kelime kontrolü
        for word in KEYWORDS:
            if word in sentence.lower():
                additional_score += 0.4
        
        # Bağlaç kontrolü
        for word in CONJUNCTIONS:
            if word in sentence.lower():
                additional_score += 0.1
        
        # Cümle uzunluğu kontrolü (çok kısa veya çok uzun cümleleri cezalandır)
        words = sentence.split()
        if len(words) < 5:
            additional_score -= 0.3
        elif len(words) > 30:
            additional_score -= 0.2
        
        # Cümle pozisyonu kontrolü (ilk ve son cümleler genellikle önemlidir)
        if i < len(sentences) * 0.1:  # İlk %10
            additional_score += 0.3
        elif i > len(sentences) * 0.9:  # Son %10
            additional_score += 0.3
        
        # Noktalama işaretleri kontrolü
        punctuation_count = sum(1 for c in sentence if c in PUNCTUATION)
        if punctuation_count > 3:  # Çok fazla noktalama işareti varsa
            additional_score += 0.1
        
        # Sayısal değer kontrolü (sayılar genellikle önemlidir)
        if re.search(r'\d+', sentence):
            additional_score += 0.2
        
        # Soru işareti kontrolü (sorular genellikle önemlidir)
        if '?' in sentence:
            additional_score += 0.2
        
        # Ünlem işareti kontrolü (vurgular genellikle önemlidir)
        if '!' in sentence:
            additional_score += 0.2
        
        # Toplam skor
        total_score = base_score + additional_score
        scores.append(total_score)
    
    return scores

def create_summary(text, summary_length):
    """Metni özetle"""
    if not text:
        return "Metin çıkarılamadı."
    
    # Metni cümlelere ayır
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return "Cümle bulunamadı."
    
    # Cümle skorlarını hesapla
    scores = calculate_sentence_scores(sentences)
    
    # En yüksek skorlu cümleleri seç
    num_sentences = max(1, int(len(sentences) * summary_length / 100))
    selected_indices = np.argsort(scores)[-num_sentences:]
    selected_indices = sorted(selected_indices)  # Orijinal sırayı koru
    
    # Seçilen cümleleri birleştir
    summary = ' '.join([sentences[i] for i in selected_indices])
    
    # Özeti daha okunabilir hale getir
    summary = re.sub(r'\s+', ' ', summary)  # Fazla boşlukları temizle
    summary = re.sub(r'\.\s*\.', '.', summary)  # Fazla noktaları temizle
    summary = re.sub(r'\.\s*([A-Z])', r'. \1', summary)  # Nokta ve büyük harf arasına boşluk ekle
    
    return summary

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
    log_visit('Ana Sayfa')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    log_visit('Özet Oluşturma')
    # Hata kontrolü
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
    
    try:
        # Dosyayı kaydet
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # PDF'den metin çıkar
        text = extract_text_from_pdf(file_path)
        if not text:
            flash('PDF dosyası okunamadı.', 'error')
            return redirect(url_for('index'))
        
        # Metni ön işle
        text = preprocess_text(text)
        
        # Özet uzunluğunu al
        summary_length = int(request.form.get('summary_length', 50))
        
        # Özet oluştur
        summary = create_summary(text, summary_length)
        
        # PDF bilgilerini veritabanına kaydet
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO pdfs (filename, original_filename, upload_date, file_size, summary_length) VALUES (?, ?, ?, ?, ?)",
                 (filename, file.filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), os.path.getsize(file_path), summary_length))
        conn.commit()
        conn.close()
        
        # Geçici dosyayı sil
        os.remove(file_path)
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        traceback.print_exc()
        flash('Özet oluşturulurken bir hata oluştu.', 'error')
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
    recent_pdf_list = [dict(zip(['id', 'filename', 'original_filename', 'upload_date', 'file_size', 'summary_length'], row))
                      for row in c.fetchall()]
    
    # Sayfa ziyaret istatistikleri
    c.execute("""
        SELECT page_name, COUNT(*) as visit_count, MAX(visit_date) as last_visit 
        FROM visits 
        GROUP BY page_name 
        ORDER BY visit_count DESC
    """)
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

# PDF listesi sayfası
@app.route('/admin/pdfs')
@admin_required
def admin_pdfs():
    log_visit('PDF Listesi')
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM pdfs ORDER BY upload_date DESC")
    pdfs = [dict(zip(['id', 'filename', 'original_filename', 'upload_date', 'file_size', 'summary_length'], row))
            for row in c.fetchall()]
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
        page_stats.append({
            'page_name': row[0],
            'visit_count': row[1],
            'last_visit': row[2]  # String olarak bırak
        })
    
    # Son ziyaretler
    c.execute("""
        SELECT * FROM visits 
        ORDER BY visit_date DESC 
        LIMIT 10
    """)
    recent_visits = []
    for row in c.fetchall():
        recent_visits.append({
            'id': row[0],
            'page_name': row[1],
            'ip_address': row[2],
            'visit_date': row[3]  # String olarak bırak
        })
    
    conn.close()
    
    return render_template('admin/visits.html',
                         dates=dates,
                         visit_counts=visit_counts,
                         page_stats=page_stats,
                         recent_visits=recent_visits)

# PDF silme işlemi
@app.route('/admin/delete_pdf/<int:pdf_id>', methods=['POST'])
@admin_required
def delete_pdf(pdf_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # PDF bilgilerini al
    c.execute("SELECT filename FROM pdfs WHERE id = ?", (pdf_id,))
    pdf = c.fetchone()
    
    if pdf:
        # Dosyayı sil
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf[0])
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Veritabanından sil
        c.execute("DELETE FROM pdfs WHERE id = ?", (pdf_id,))
        conn.commit()
        flash('PDF başarıyla silindi!', 'success')
    else:
        flash('PDF bulunamadı!', 'error')
    
    conn.close()
    return redirect(url_for('admin_pdfs'))

if __name__ == '__main__':
    app.run(debug=True) 