document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    
    // New UI Elements
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    // --- 1. Sidebar Toggle Logic ---
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
    });

    // --- 2. Dark Mode Logic ---
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        
        // Change Icon based on mode
        if (document.body.classList.contains('dark-mode')) {
            themeIcon.textContent = '☀️'; // Sun icon for dark mode
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.textContent = '🌙'; // Moon icon for light mode
            localStorage.setItem('theme', 'light');
        }
    });

    // Check saved preference on load
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
        themeIcon.textContent = '☀️';
    }

    // --- 3. Message Formatting ---
    function formatMessage(text) {
        // Basic Markdown support: Bold, Newlines, Bullet points
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/- (.*?)(<br>|$)/g, '<li>$1</li>');
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
        } else {
            bubble.textContent = text;
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);
        chatBox.appendChild(msgDiv);
        
        // Smooth scroll to bottom
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
        } catch (error) {
            addMessage("⚠️ Network error. Please check your connection.", 'bot');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});