from flask import Flask, render_template, request, jsonify
import json
import os
from src.chatbot import AIChChatBot
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize the chatbot
chatbot = AIChChatBot()

@app.route('/')
def index():
    """Main page with chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        user_message = request.json.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from chatbot
        bot_response = chatbot.get_response(user_message)
        
        # Log the conversation
        log_conversation(user_message, bot_response)
        
        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().strftime('%H:%M')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'chatbot': 'ready'})

def log_conversation(user_msg, bot_msg):
    """Log conversations for analysis"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_msg,
        'bot': bot_msg
    }
    
    log_file = 'data/conversation_logs.jsonl'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

if __name__ == '__main__':
    print("🤖 Starting AI ChatBot...")
    print("🌐 Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
