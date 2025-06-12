class ChatbotWidget {
    constructor(options = {}) {
        this.options = {
            position: options.position || 'bottom-right',
            title: options.title || 'Chatbot',
            welcomeMessage: options.welcomeMessage || 'Merhaba! Size nasıl yardımcı olabilirim?',
            theme: options.theme || 'light',
            maxMessages: options.maxMessages || 50,
            ...options
        };
        
        this.messageHistory = [];
        this.isOpen = false;
        this.init();
    }

    init() {
        this.createWidget();
        this.attachEventListeners();
        this.loadMessageHistory();
        this.showWelcomeMessage();
        this.setupCategoryListeners();
    }

    setupCategoryListeners() {
        // Kategori başlıklarına tıklama olaylarını ekle
        const categories = document.querySelectorAll('.dropdown-item');
        categories.forEach(category => {
            category.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const categoryName = category.textContent.trim();
                this.handleCategoryClick(categoryName);
            });
        });
    }

    handleCategoryClick(categoryName) {
        // Chatbot'u aç
        if (!this.isOpen) {
            this.toggleWidget();
        }
        
        // Kategori mesajını göster
        setTimeout(() => {
            this.addMessage(`${categoryName} kategorisi hakkında bilgi almak ister misiniz?`, 'bot');
        }, 300);
    }

    createWidget() {
        const widget = document.createElement('div');
        widget.className = 'chat-widget';
        widget.innerHTML = `
            <button class="chat-widget-button" aria-label="Sohbeti aç">
                <i class="fas fa-comments"></i>
            </button>
            <div class="chat-widget-container">
                <div class="chat-widget-header">
                    <h3>${this.options.title}</h3>
                    <button class="chat-widget-close" aria-label="Sohbeti kapat">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="chat-widget-messages"></div>
                <div class="chat-widget-input">
                    <input type="text" placeholder="Mesajınızı yazın..." aria-label="Mesaj girişi">
                    <button aria-label="Mesaj gönder">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(widget);
        
        this.widget = widget;
        this.button = widget.querySelector('.chat-widget-button');
        this.container = widget.querySelector('.chat-widget-container');
        this.messages = widget.querySelector('.chat-widget-messages');
        this.input = widget.querySelector('input');
        this.sendButton = widget.querySelector('.chat-widget-input button');
        this.closeButton = widget.querySelector('.chat-widget-close');
    }

    attachEventListeners() {
        this.button.addEventListener('click', () => this.toggleWidget());
        this.closeButton.addEventListener('click', () => this.toggleWidget());
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Dışarı tıklandığında kapatma
        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.widget.contains(e.target)) {
                this.toggleWidget();
            }
        });

        // ESC tuşu ile kapatma
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.toggleWidget();
            }
        });
    }

    toggleWidget() {
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            this.container.classList.add('visible');
            this.input.focus();
        } else {
            this.container.classList.remove('visible');
            // Chatbot kapatıldığında sıfırla
            this.resetChatbot();
        }
    }

    resetChatbot() {
        // Mesaj geçmişini temizle
        this.messageHistory = [];
        this.messages.innerHTML = '';
        // Karşılama mesajını göster
        this.showWelcomeMessage();
        // LocalStorage'dan mesaj geçmişini sil
        localStorage.removeItem('chatbot_history');
    }

    showWelcomeMessage() {
        if (this.messageHistory.length === 0) {
            this.addMessage(this.options.welcomeMessage, 'bot');
        }
    }

    addMessage(text, type) {
        const message = document.createElement('div');
        message.className = `chat-message ${type}-message`;
        message.textContent = text;
        this.messages.appendChild(message);
        this.scrollToBottom();

        // Mesaj geçmişini kaydet
        this.messageHistory.push({ text, type, timestamp: new Date().toISOString() });
        this.saveMessageHistory();
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        this.messages.appendChild(indicator);
        this.scrollToBottom();
        return indicator;
    }

    removeTypingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }

    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.input.value = '';

        const typingIndicator = this.showTypingIndicator();

        try {
            const response = await this.getBotResponse(message);
            this.removeTypingIndicator(typingIndicator);
            this.addMessage(response, 'bot');
        } catch (error) {
            this.removeTypingIndicator(typingIndicator);
            this.addMessage('Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.', 'bot');
        }
    }

    async getBotResponse(message) {
        return new Promise((resolve) => {
            setTimeout(() => {
                const lowerMessage = message.toLowerCase();
                
                // PDF özetleme ile ilgili sorular
                if (lowerMessage.includes('pdf') && lowerMessage.includes('yükle')) {
                    resolve('PDF dosyanızı yüklemek için "PDF Dosyası Seçin" butonuna tıklayın veya dosyayı sürükleyip bırakın. Desteklenen formatlar: PDF.');
                }
                else if (lowerMessage.includes('özet') && lowerMessage.includes('oluştur')) {
                    resolve('Özet oluşturmak için dosyayı yükledikten sonra "Özet Oluştur" butonuna tıklayın. Özet uzunluğunu kaydırıcı ile ayarlayabilirsiniz.');
                }
                else if (lowerMessage.includes('özet') && lowerMessage.includes('uzunluk')) {
                    resolve('Özet uzunluğunu kaydırıcı ile ayarlayabilirsiniz. Kısa özetler ana fikirleri içerirken, uzun özetler daha detaylı bilgi sağlar.');
                }
                else if (lowerMessage.includes('desteklenen') && lowerMessage.includes('format')) {
                    resolve('Şu anda sadece PDF formatındaki dosyaları destekliyoruz. Gelecekte diğer formatları da eklemeyi planlıyoruz.');
                }
                
                // Genel sorular
                else if (lowerMessage.includes('merhaba') || lowerMessage.includes('selam')) {
                    resolve('Merhaba! Ben ÖZETLY asistanıyım. PDF özetleme konusunda size nasıl yardımcı olabilirim?');
                }
                else if (lowerMessage.includes('nasılsın') || lowerMessage.includes('iyi misin')) {
                    resolve('İyiyim, teşekkür ederim! Siz nasılsınız?');
                }
                else if (lowerMessage.includes('yardım') || lowerMessage.includes('nasıl')) {
                    resolve('Size şu konularda yardımcı olabilirim:\n1. PDF dosyası yükleme\n2. Özet oluşturma\n3. Özet uzunluğunu ayarlama\n4. Desteklenen formatlar\n\nBaşka bir sorunuz var mı?');
                }
                else if (lowerMessage.includes('teşekkür') || lowerMessage.includes('sağol')) {
                    resolve('Rica ederim! Başka bir konuda yardıma ihtiyacınız olursa bana sorabilirsiniz.');
                }
                else if (lowerMessage.includes('görüşürüz') || lowerMessage.includes('hoşça kal')) {
                    resolve('Görüşmek üzere! İyi günler dilerim.');
                }
                else {
                    resolve('Üzgünüm, anlayamadım. PDF özetleme ile ilgili sorularınızı yanıtlamaktan mutluluk duyarım. Size nasıl yardımcı olabilirim?');
                }
            }, 1000);
        });
    }

    scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }

    saveMessageHistory() {
        try {
            const history = this.messageHistory.slice(-this.options.maxMessages);
            localStorage.setItem('chatbot_history', JSON.stringify(history));
        } catch (error) {
            console.error('Mesaj geçmişi kaydedilemedi:', error);
        }
    }

    loadMessageHistory() {
        try {
            const history = JSON.parse(localStorage.getItem('chatbot_history') || '[]');
            this.messageHistory = history;
            history.forEach(msg => {
                const message = document.createElement('div');
                message.className = `chat-message ${msg.type}-message`;
                message.textContent = msg.text;
                this.messages.appendChild(message);
            });
            this.scrollToBottom();
        } catch (error) {
            console.error('Mesaj geçmişi yüklenemedi:', error);
        }
    }
}

// Widget'ı başlat
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatbotWidget({
        title: 'ÖZETLY Asistanı',
        welcomeMessage: 'Merhaba! Ben ÖZETLY asistanıyım. PDF özetleme konusunda size yardımcı olabilirim. Nasıl yardımcı olabilirim?',
        maxMessages: 50
    });
}); 