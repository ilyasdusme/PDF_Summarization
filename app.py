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
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Metni temizle
                    page_text = page_text.strip()
                    # Unicode karakterleri düzelt
                    page_text = page_text.encode('ascii', 'ignore').decode('utf-8')
                    # Fazla boşlukları temizle
                    page_text = re.sub(r'\s+', ' ', page_text)
                    # Noktalama işaretlerini düzelt
                    page_text = re.sub(r'\.+', '.', page_text)
                    page_text = re.sub(r'\.\s*([A-Z])', r'. \1', page_text)
                    text.append(page_text)
            
            if not text:
                return None
            
            return ' '.join(text)
    except Exception as e:
        print(f"PDF okuma hatası: {str(e)}")
        traceback.print_exc()
        return None

def preprocess_text(text):
    """Metni ön işleme"""
    if not text:
        return []
    
    try:
        # Metni temizle
        text = text.strip()
        # Unicode karakterleri düzelt
        text = text.encode('ascii', 'ignore').decode('utf-8')
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        # Noktalama işaretlerini düzelt
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\.\s*([A-Z])', r'. \1', text)
        
        # Metni cümlelere ayır
        sentences = []
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # En az 3 kelime içeren cümleleri al
                sentences.append(sentence)
        
        return sentences
    except Exception as e:
        print(f"Metin ön işleme hatası: {str(e)}")
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
            
            # İngilizce cümle kontrolü
            if is_english_sentence(sentence):
                additional_score -= 0.3
            
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

def create_summary(text, summary_length):
    """Metni özetle"""
    if not text:
        return "Metin çıkarılamadı."
    
    try:
        # Metni cümlelere ayır
        sentences = preprocess_text(text)
        
        if not sentences:
            return "Cümle bulunamadı."
        
        # Cümle skorlarını hesapla
        scores = calculate_sentence_scores(sentences)
        
        # En yüksek skorlu cümleleri seç
        num_sentences = max(1, int(len(sentences) * summary_length / 100))
        selected_indices = np.argsort(scores)[-num_sentences:]
        selected_indices = sorted(selected_indices)  # Orijinal sırayı koru
        
        # Seçilen cümleleri birleştir
        summary = '. '.join([sentences[i] for i in selected_indices])
        
        # Özeti temizle ve formatla
        summary = summary.strip()
        # Unicode karakterleri düzelt
        summary = summary.encode('ascii', 'ignore').decode('utf-8')
        # Fazla boşlukları temizle
        summary = re.sub(r'\s+', ' ', summary)
        # Fazla noktaları temizle
        summary = re.sub(r'\.+', '.', summary)
        # Nokta ve büyük harf arasına boşluk ekle
        summary = re.sub(r'\.\s*([A-Z])', r'. \1', summary)
        
        if not summary:
            return "Özet oluşturulamadı."
        
        return summary
    except Exception as e:
        print(f"Özet oluşturma hatası: {str(e)}")
        traceback.print_exc()
        return "Özet oluşturulurken bir hata oluştu."

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
    try:
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
        
        # Dosyayı kaydet
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # PDF'den metin çıkar
        text = extract_text_from_pdf(file_path)
        if not text:
            flash('PDF dosyası okunamadı veya boş.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        # Özet uzunluğunu al
        summary_length = int(request.form.get('summary_length', 50))
        
        # Özet oluştur
        summary = create_summary(text, summary_length)
        if not summary or summary == "Metin çıkarılamadı." or summary == "Cümle bulunamadı." or summary == "Özet oluşturulurken bir hata oluştu.":
            flash('PDF dosyasından özet oluşturulamadı.', 'error')
            os.remove(file_path)
            return redirect(url_for('index'))
        
        # PDF bilgilerini veritabanına kaydet
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO pdfs (filename, original_filename, upload_date, file_size, summary_length) VALUES (?, ?, ?, ?, ?)",
                 (filename, file.filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), os.path.getsize(file_path), summary_length))
        conn.commit()
        conn.close()
        
        # Geçici dosyayı sil
        os.remove(file_path)
        
        # Özeti düzenle ve formatla
        summary = summary.strip()
        # Unicode karakterleri düzelt
        summary = summary.encode('ascii', 'ignore').decode('utf-8')
        # Fazla boşlukları temizle
        summary = re.sub(r'\s+', ' ', summary)
        # Fazla noktaları temizle
        summary = re.sub(r'\.+', '.', summary)
        # Nokta ve büyük harf arasına boşluk ekle
        summary = re.sub(r'\.\s*([A-Z])', r'. \1', summary)
        
        # Özeti cümlelere ayır ve düzenle
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        summary = '. '.join(sentences) + '.'
        
        return render_template('index.html', summary=summary, filename=file.filename)
    
    except Exception as e:
        print(f"Yükleme hatası: {str(e)}")
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