import re
import random
import os
from typing import List, Dict

# Optional imports - will gracefully degrade without them
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import nltk
    HAS_AI_DEPS = True
except ImportError:
    HAS_AI_DEPS = False
    print("📦 AI dependencies not found. Running in basic mode with rule-based responses.")

class AIChChatBot:
    """
    AI-powered chatbot using Hugging Face transformers
    Optimized for FAQ and conversational assistance
    """
    
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"  # Free conversational model
        self.device = "cpu"  # Default to CPU
        self.ai_mode = False
        
        print(f"🚀 Initializing chatbot...")
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Load FAQ responses
        self.faq_responses = self._load_faq_responses()
        
        # Try to initialize AI components if available
        if HAS_AI_DEPS:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"📱 Using device: {self.device}")
                
                # Initialize model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.ai_mode = True
                print("🤖 AI mode enabled with transformer model")
                
                # Download NLTK data
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except:
                    pass
                    
            except Exception as e:
                print(f"⚠️ Could not load AI model: {e}")
                print("🔄 Falling back to rule-based responses")
                self.ai_mode = False
        else:
            print("📋 Running in basic mode with rule-based responses")
        
        print("✅ Chatbot initialized successfully!")
    
    def _load_faq_responses(self) -> Dict[str, str]:
        """Load predefined FAQ responses"""
        return {
            "hello": "Hello! I'm your AI assistant. How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "help": "I'm here to help! You can ask me questions, have a conversation, or ask for assistance with various topics.",
            "what is your name": "I'm an AI chatbot created for the Future-O-Tech event. You can call me ChatBot!",
            "who are you": "I'm an AI-powered conversational assistant built using natural language processing.",
            "how are you": "I'm doing great, thank you for asking! How are you doing today?",
            "bye": "Goodbye! It was nice talking with you. Have a great day!",
            "thank you": "You're welcome! I'm glad I could help.",
            "thanks": "You're welcome! Feel free to ask if you need anything else.",
            "what can you do": "I can answer questions, have conversations, provide information, and assist with various tasks. What would you like to know?",
            "weather": "I don't have access to real-time weather data, but I'd be happy to help you with other questions!",
            "time": "I don't have access to real-time data, but you can check your device for the current time.",
        }
    
    def _preprocess_input(self, text: str) -> str:
        """Clean and preprocess user input"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def _check_faq(self, text: str) -> str:
        """Check if input matches FAQ patterns"""
        processed_text = self._preprocess_input(text)
        
        # Check for exact matches first
        if processed_text in self.faq_responses:
            return self.faq_responses[processed_text]
        
        # Check for partial matches
        for key, response in self.faq_responses.items():
            if key in processed_text or any(word in processed_text for word in key.split()):
                return response
        
        return None
    
    def _generate_ai_response(self, text: str) -> str:
        """Generate response using the AI model or rule-based fallback"""
        if not self.ai_mode:
            # Use rule-based responses when AI is not available
            return self._get_rule_based_response(text)
        
        try:
            # Encode the input text
            input_text = text + self.tokenizer.eos_token
            encoded_input = self.tokenizer.encode(input_text, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    encoded_input,
                    max_length=encoded_input.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(text):].strip()
            
            # If response is empty or too short, use fallback
            if len(response) < 5:
                return self._get_fallback_response()
            
            return response
        
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._get_fallback_response()
    
    def _get_rule_based_response(self, text: str) -> str:
        """Generate rule-based responses when AI is not available"""
        text_lower = text.lower()
        
        # Simple pattern matching for common conversation topics
        if any(word in text_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            return "AI is fascinating! It involves creating systems that can perform tasks that typically require human intelligence."
        
        elif any(word in text_lower for word in ['future', 'technology', 'tech']):
            return "Technology is rapidly evolving! The future holds exciting possibilities for AI and automation."
        
        elif any(word in text_lower for word in ['learn', 'study', 'education']):
            return "Learning is wonderful! I'm here to help answer questions and provide information on various topics."
        
        elif any(word in text_lower for word in ['project', 'work', 'build']):
            return "That sounds like an interesting project! What specific aspects are you working on?"
        
        elif '?' in text:
            return "That's a great question! While I might not have all the details, I'm happy to discuss it further."
        
        else:
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Provide fallback responses when AI generation fails"""
        fallbacks = [
            "That's interesting! Could you tell me more about that?",
            "I understand. What else would you like to discuss?",
            "Thanks for sharing that. How can I help you further?",
            "I see. Is there anything specific you'd like to know?",
            "That's a good point. What other questions do you have?",
        ]
        return random.choice(fallbacks)
    
    def get_response(self, user_input: str) -> str:
        """
        Generate a response to user input
        
        Args:
            user_input: The user's message
            
        Returns:
            Bot's response
        """
        if not user_input or not user_input.strip():
            return "I didn't receive any message. Could you please say something?"
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Check FAQ first for quick responses
        faq_response = self._check_faq(user_input)
        if faq_response:
            response = faq_response
        else:
            # Generate AI response
            response = self._generate_ai_response(user_input)
        
        # Add bot response to history
        self.conversation_history.append({"role": "bot", "content": response})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared!"
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "model": self.model_name,
            "device": self.device,
            "faq_entries": len(self.faq_responses)
        }
