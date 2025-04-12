document.addEventListener('DOMContentLoaded', function() {
    const chatbotButton = document.querySelector('.chatbot-button');
    const chatbotWindow = document.querySelector('.chatbot-window');
    const closeButton = document.querySelector('.close-button');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Load conversation history from localStorage
    let conversationHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];

    // Display conversation history
    function displayHistory() {
        chatMessages.innerHTML = '';
        conversationHistory.forEach(msg => {
            if (msg.type === 'user') {
                addUserMessage(msg.text);
            } else {
                addBotMessage(msg.text);
            }
        });
    }

    // Open chatbot window
    chatbotButton.addEventListener('click', function() {
        chatbotWindow.style.display = 'block';
        displayHistory();
        if (conversationHistory.length === 0) {
            setTimeout(() => {
                const welcomeMessage = 'Merhaba! Size nasıl yardımcı olabilirim?';
                addBotMessage(welcomeMessage);
                conversationHistory.push({ type: 'bot', text: welcomeMessage });
                saveHistory();
            }, 500);
        }
    });

    // Close chatbot window
    closeButton.addEventListener('click', function() {
        chatbotWindow.style.display = 'none';
    });

    // Send message
    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addUserMessage(message);
            conversationHistory.push({ type: 'user', text: message });
            userInput.value = '';
            
            // Simulate bot response
            setTimeout(() => {
                const response = getBotResponse(message);
                addBotMessage(response);
                conversationHistory.push({ type: 'bot', text: response });
                saveHistory();
            }, 500);
        }
    }

    // Save conversation history to localStorage
    function saveHistory() {
        localStorage.setItem('chatHistory', JSON.stringify(conversationHistory));
    }

    // Send message on button click
    sendButton.addEventListener('click', sendMessage);

    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Add user message to chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Add bot message to chat
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Get bot response based on user message
    function getBotResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('merhaba') || lowerMessage.includes('selam')) {
            return 'Merhaba! PDF özetleme konusunda size nasıl yardımcı olabilirim?';
        }
        else if (lowerMessage.includes('nasılsın') || lowerMessage.includes('iyi misin')) {
            return 'İyiyim, teşekkür ederim! Siz nasılsınız?';
        }
        else if (lowerMessage.includes('pdf') && lowerMessage.includes('yükle')) {
            return 'PDF dosyanızı yüklemek için "PDF Dosyası Seçin" butonuna tıklayın veya dosyayı sürükleyip bırakın.';
        }
        else if (lowerMessage.includes('özet') && lowerMessage.includes('oluştur')) {
            return 'Özet oluşturmak için dosyayı yükledikten sonra "Özet Oluştur" butonuna tıklayın. Özet uzunluğunu kaydırıcı ile ayarlayabilirsiniz.';
        }
        else if (lowerMessage.includes('yardım') || lowerMessage.includes('nasıl')) {
            return 'Size nasıl yardımcı olabilirim? PDF yükleme, özet oluşturma veya başka bir konuda sorunuz var mı?';
        }
        else if (lowerMessage.includes('teşekkür') || lowerMessage.includes('sağol')) {
            return 'Rica ederim! Başka bir konuda yardıma ihtiyacınız olursa bana sorabilirsiniz.';
        }
        else {
            return 'Üzgünüm, anlayamadım. Lütfen başka bir şekilde sorunuzu ifade edin.';
        }
    }

    // Slider değerini güncelle
    const slider = document.getElementById('summary_length');
    const sliderValue = document.querySelector('.slider-value');

    if (slider && sliderValue) {
        slider.addEventListener('input', function() {
            sliderValue.textContent = this.value + '%';
        });
    }
}); 