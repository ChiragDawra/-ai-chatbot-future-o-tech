import os
import json
import re

# Optional imports - will work without them
try:
    import pandas as pd
    from datasets import load_dataset
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    HAS_ADVANCED_DEPS = True
except ImportError:
    HAS_ADVANCED_DEPS = False
    print("📦 Some optional dependencies not found. Using basic functionality.")

class DataPreparation:
    """
    Handles dataset downloading, preprocessing, and preparation
    Uses free datasets from Hugging Face
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def download_conversational_dataset(self):
        """Download a free conversational dataset"""
        print("📥 Downloading conversational dataset...")
        
        try:
            # Use the "daily_dialog" dataset - it's free and good for general conversation
            dataset = load_dataset("daily_dialog")
            
            # Extract training data
            train_data = dataset['train']
            
            conversations = []
            for example in train_data:
                dialog = example['dialog']
                
                # Create conversation pairs
                for i in range(len(dialog) - 1):
                    if dialog[i].strip() and dialog[i+1].strip():
                        conversations.append({
                            'input': dialog[i].strip(),
                            'response': dialog[i+1].strip()
                        })
            
            # Save to JSON
            output_file = os.path.join(self.data_dir, 'conversations.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversations[:5000], f, indent=2, ensure_ascii=False)  # Limit to 5000 for efficiency
            
            print(f"✅ Downloaded {len(conversations[:5000])} conversation pairs")
            print(f"💾 Saved to: {output_file}")
            
            return conversations[:5000]
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            # Create fallback sample data
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample conversation data as fallback"""
        print("📝 Creating sample conversation data...")
        
        sample_conversations = [
            {"input": "Hello", "response": "Hi there! How can I help you today?"},
            {"input": "How are you?", "response": "I'm doing great, thank you for asking! How are you?"},
            {"input": "What's your name?", "response": "I'm an AI chatbot. You can call me Assistant!"},
            {"input": "What can you do?", "response": "I can answer questions, have conversations, and help with various tasks."},
            {"input": "Tell me a joke", "response": "Why don't scientists trust atoms? Because they make up everything!"},
            {"input": "What's the weather like?", "response": "I don't have access to weather data, but I hope it's nice where you are!"},
            {"input": "Help me", "response": "Of course! What do you need help with?"},
            {"input": "Thank you", "response": "You're very welcome! I'm glad I could help."},
            {"input": "Goodbye", "response": "Goodbye! It was nice chatting with you. Have a great day!"},
            {"input": "What is AI?", "response": "AI stands for Artificial Intelligence. It's technology that enables machines to simulate human intelligence."},
        ]
        
        # Save sample data
        output_file = os.path.join(self.data_dir, 'conversations.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created {len(sample_conversations)} sample conversations")
        return sample_conversations
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text.strip()
    
    def create_training_data(self):
        """Create and preprocess training data"""
        print("🔄 Preparing training data...")
        
        # Load conversations
        conv_file = os.path.join(self.data_dir, 'conversations.json')
        if not os.path.exists(conv_file):
            conversations = self.download_conversational_dataset()
        else:
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        
        # Preprocess conversations
        processed_conversations = []
        for conv in conversations:
            processed_input = self.preprocess_text(conv['input'])
            processed_response = self.preprocess_text(conv['response'])
            
            if processed_input and processed_response:
                processed_conversations.append({
                    'input': processed_input,
                    'response': processed_response
                })
        
        # Save processed data
        processed_file = os.path.join(self.data_dir, 'processed_conversations.json')
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Processed {len(processed_conversations)} conversations")
        print(f"💾 Saved to: {processed_file}")
        
        return processed_conversations
    
    def create_faq_dataset(self):
        """Create FAQ dataset for quick responses"""
        print("📋 Creating FAQ dataset...")
        
        faq_data = [
            {"question": "what is this chatbot", "answer": "This is an AI-powered chatbot created for the Future-O-Tech event using natural language processing."},
            {"question": "who created you", "answer": "I was created by a team of developers for the Future-O-Tech event as part of an AI project."},
            {"question": "what technology do you use", "answer": "I use advanced natural language processing with transformer models and machine learning techniques."},
            {"question": "how do you work", "answer": "I process your messages using AI algorithms to understand context and generate appropriate responses."},
            {"question": "can you learn", "answer": "Yes! I can learn from conversations and improve my responses over time."},
            {"question": "what is future o tech", "answer": "Future-O-Tech is an innovation event where teams create AI and technology solutions to real-world problems."},
            {"question": "are you real", "answer": "I'm an AI assistant - not human, but I'm real software designed to help and chat with you!"},
            {"question": "do you have feelings", "answer": "I don't have feelings like humans do, but I'm designed to be helpful and engaging in our conversations."},
        ]
        
        # Save FAQ data
        faq_file = os.path.join(self.data_dir, 'faq_data.json')
        with open(faq_file, 'w', encoding='utf-8') as f:
            json.dump(faq_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created {len(faq_data)} FAQ entries")
        return faq_data

if __name__ == "__main__":
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Download and prepare datasets
    conversations = data_prep.create_training_data()
    faq_data = data_prep.create_faq_dataset()
    
    print("\n🎉 Data preparation completed!")
    print(f"📊 Total conversation pairs: {len(conversations)}")
    print(f"❓ FAQ entries: {len(faq_data)}")
