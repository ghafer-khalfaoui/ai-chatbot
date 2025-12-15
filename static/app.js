// Wait for the DOM to be fully loaded before running the script
document.addEventListener("DOMContentLoaded", () => {
    
    // Get references to the HTML elements we need
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Function to add a message to the chat box
    function addMessage(message, sender) {
        const messageElement = document.createElement('div');
        
        // This adds a class 'user-message' or 'bot-message'
        messageElement.classList.add('message');
        if (sender === 'user') {
            messageElement.classList.add('user-message');
        } else {
            messageElement.classList.add('bot-message');
        }
        
        const p = document.createElement('p');
        p.textContent = message;
        messageElement.appendChild(p);
        
        chatBox.appendChild(messageElement);
        
        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to send a message to the bot
    async function sendMessage() {
        const message = userInput.value.trim();
        
        if (message === "") {
            return; // Don't send empty messages
        }

        // Add the user's message to the chat box
        addMessage(message, 'user');
        
        // Clear the input field
        userInput.value = '';

        try {
            // Send the message to the Flask server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Add the bot's response to the chat box
            addMessage(data.response, 'bot');

        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, something went wrong. Please try again.', 'bot');
        }
    }

    // --- Event Listeners ---
    
    // Send message when the "Send" button is clicked
    sendBtn.addEventListener('click', sendMessage);

    // Send message when the "Enter" key is pressed in the input field
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});