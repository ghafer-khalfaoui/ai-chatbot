document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn'); 
    
    // UI Elements
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    // --- 1. VOICE CONFIGURATION ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = SpeechRecognition ? new SpeechRecognition() : null;

    if (recognition) {
        recognition.continuous = false;
        recognition.lang = "en-US";
        recognition.interimResults = false;

        recognition.onstart = function() {
            micBtn.classList.add("mic-active");
            userInput.placeholder = "Listening...";
        };

        recognition.onend = function() {
            micBtn.classList.remove("mic-active");
            userInput.placeholder = "Ask me anything...";
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            // OPTIONAL: Uncomment the next line if you want it to send AUTOMATICALLY after speaking
            // sendMessage(); 
        };

        micBtn.addEventListener('click', () => {
            if (micBtn.classList.contains("mic-active")) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        micBtn.style.display = 'none';
    }

    // --- 2. TEXT-TO-SPEECH (Triggered by Button Only) ---
    window.speakText = function(text) {
        window.speechSynthesis.cancel(); // Stop any current speech
        
        let cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1'); // Remove bold markdown
        cleanText = cleanText.replace(/<br>/g, '. '); // Remove html breaks
        cleanText = cleanText.replace(/(<([^>]+)>)/gi, ""); // Remove other tags

        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = "en-US";
        utterance.rate = 1.0; 
        window.speechSynthesis.speak(utterance);
    }

    // --- 3. UI LOGIC (Sidebar & Theme) ---
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
        sidebar.classList.toggle('active');
    });

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        if (document.body.classList.contains('dark-mode')) {
            themeIcon.textContent = '☀️';
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.textContent = '🌙';
            localStorage.setItem('theme', 'light');
        }
    });

    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
        themeIcon.textContent = '☀️';
    }

    // --- 4. CHAT LOGIC ---
    function formatMessage(text) {
        let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\n/g, '<br>');
        return formatted;
    }

    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message');
        msgDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.textContent = sender === 'user' ? '👤' : '🤖';

        const bubble = document.createElement('div');
        bubble.classList.add('bubble');
        
        if (sender === 'bot') {
            bubble.innerHTML = formatMessage(text);
            
            // --- NEW: Add Speaker Button to Bot Messages ---
            // We use encodeURIComponent to ensure the text doesn't break the HTML
            const safeText = text.replace(/"/g, '&quot;');
            const speakBtn = document.createElement('button');
            speakBtn.className = 'speak-btn';
            speakBtn.innerHTML = '🔊';
            speakBtn.onclick = function() { window.speakText(safeText); };
            
            // Append bubble then button
            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
            msgDiv.appendChild(speakBtn); 
        } else {
            bubble.textContent = text;
            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
        }

        chatBox.appendChild(msgDiv);
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text }),
            });
            const data = await response.json();
            
            addMessage(data.response, 'bot');
            // REMOVED: speak(data.response); <<-- No more auto speaking!

        } catch (error) {
            addMessage("⚠️ Network error. Please check your connection.", 'bot');
            console.error(error);
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // Suggestion Chips
    document.querySelectorAll('.info-box li').forEach(li => {
        li.addEventListener('click', () => {
            userInput.value = li.textContent.replace(/"/g, '');
            sendMessage();
        });
    });
});