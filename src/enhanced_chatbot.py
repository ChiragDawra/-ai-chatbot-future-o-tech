#!/usr/bin/env python3
"""
Enhanced AI Chatbot using Large Dataset
Uses 219k+ conversations from Cornell Movie Dialogs for better responses
"""

import json
import re
import random
import os
from typing import Dict, List, Optional
from pathlib import Path
import difflib

# Optional imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_AI_DEPS = True
except ImportError:
    HAS_AI_DEPS = False
    print("📦 AI dependencies not found. Using enhanced rule-based responses.")

class EnhancedChatBot:
    """
    Enhanced chatbot with large dataset integration
    """
    
    def __init__(self, use_large_dataset=True):
        self.conversation_history = []
        self.large_dataset = []
        self.faq_responses = self._load_faq_responses()
        
        # Load large dataset for enhanced responses
        if use_large_dataset:
            self._load_large_dataset()
        
        # Initialize AI components if available
        self.ai_mode = False
        if HAS_AI_DEPS:
            try:
                self._initialize_ai_model()
            except Exception as e:
                print(f"⚠️ Could not load AI model: {e}")
                print("🔄 Using enhanced rule-based mode")
        
        print(f"🤖 Enhanced ChatBot initialized with {len(self.large_dataset)} training examples")
    
    def _load_large_dataset(self):
        """Load the large conversational dataset"""
        dataset_path = Path("data/large/large_conversations.json")
        
        if dataset_path.exists():
            try:
                print("📊 Loading large conversational dataset...")
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.large_dataset = json.load(f)
                print(f"✅ Loaded {len(self.large_dataset):,} conversation examples")
            except Exception as e:
                print(f"⚠️ Could not load large dataset: {e}")
                self.large_dataset = []
        else:
            print("📋 Large dataset not found. Using basic responses.")
            self.large_dataset = []
    
    def _initialize_ai_model(self):
        """Initialize AI model if available"""
        try:
            print("🤖 Initializing AI model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_name = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.ai_mode = True
            print(f"✅ AI model loaded on {self.device}")
        except Exception as e:
            print(f"❌ AI model initialization failed: {e}")
            self.ai_mode = False
    
    def _load_faq_responses(self) -> Dict[str, str]:
        """Load enhanced FAQ responses"""
        return {
            # Basic greetings
            "hello": "Hello! I'm your enhanced AI assistant trained on over 200,000 conversations. How can I help you today?",
            "hi": "Hi there! I'm ready to chat with you using my large conversational knowledge base.",
            "hey": "Hey! What's up? I'm here to have a great conversation with you.",
            
            # About the chatbot
            "what is your name": "I'm an AI chatbot trained on a massive dataset of movie conversations for the Future-O-Tech event. You can call me ChatBot!",
            "who are you": "I'm an enhanced AI conversational assistant powered by over 219,000 real conversations from Cornell Movie Dialogs.",
            "what can you do": "I can have natural conversations, answer questions, discuss topics, and provide assistance using my training on hundreds of thousands of real conversations.",
            
            # About AI and technology
            "what is ai": "AI, or Artificial Intelligence, is technology that enables machines to simulate human intelligence and learn from large datasets like the one I was trained on.",
            "how do you work": "I process your messages using natural language understanding and generate responses based on patterns learned from over 200,000 real conversations.",
            "what technology do you use": "I use transformer neural networks, trained on Cornell Movie Dialogs dataset with 219,000+ conversation pairs, plus advanced NLP techniques.",
            
            # About the project
            "what is this chatbot": "This is an enhanced AI chatbot for the Future-O-Tech event, trained on massive conversational datasets to provide human-like interactions.",
            "who created you": "I was created by a team of developers for the Future-O-Tech event, using state-of-the-art AI and large-scale conversational datasets.",
            "what is future o tech": "Future-O-Tech is an innovation event where teams create cutting-edge AI solutions. This chatbot demonstrates advanced conversational AI.",
            
            # Conversation topics
            "tell me about yourself": "I'm an AI trained on hundreds of thousands of real conversations from movies. This gives me a rich understanding of natural dialogue patterns!",
            "what makes you special": "I'm powered by one of the largest conversational datasets - over 219,000 real movie conversations - making my responses more natural and diverse.",
            "how are you different": "Unlike basic chatbots, I'm trained on massive real conversation data, allowing me to understand context and generate more human-like responses.",
            
            # Goodbyes
            "bye": "Goodbye! It was great chatting with you. Thanks for trying my enhanced conversational abilities!",
            "goodbye": "Farewell! I hope you enjoyed experiencing AI powered by large-scale conversational data. Have a wonderful day!",
            "see you later": "See you later! Feel free to come back anytime to chat with my enhanced AI system.",
            
            # Thanks
            "thank you": "You're very welcome! I'm glad my enhanced conversational training could help you.",
            "thanks": "You're welcome! My large dataset training allows me to assist in many different ways."
        }
    
    def _preprocess_input(self, text: str) -> str:
        """Clean and preprocess user input"""
        return re.sub(r'[^\w\s.,!?]', '', text.lower().strip())
    
    def _find_similar_conversation(self, user_input: str, top_k: int = 5) -> Optional[str]:
        """Find similar conversations from the large dataset"""
        if not self.large_dataset:
            return None
        
        processed_input = self._preprocess_input(user_input)
        
        # Simple similarity matching
        similarities = []
        for conv in self.large_dataset:
            conv_input = self._preprocess_input(conv.get('input', ''))
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, processed_input, conv_input).ratio()
            
            if similarity > 0.3:  # Threshold for relevance
                similarities.append((similarity, conv['response']))
        
        if similarities:
            # Sort by similarity and return best response
            similarities.sort(reverse=True, key=lambda x: x[0])
            return similarities[0][1]
        
        return None
    
    def _get_random_movie_response(self) -> str:
        """Get a random response from the movie dataset"""
        if self.large_dataset:
            random_conv = random.choice(self.large_dataset)
            return random_conv['response']
        return None
    
    def _generate_contextual_response(self, user_input: str) -> str:
        """Generate contextual responses based on input content"""
        text_lower = user_input.lower()
        
        # Enhanced contextual responses using dataset knowledge
        if any(word in text_lower for word in ['movie', 'film', 'cinema']):
            responses = [
                "I love talking about movies! I was actually trained on conversations from thousands of movie scripts.",
                "Movies are fascinating! My knowledge comes from analyzing dialogue from countless films.",
                "That's interesting! Having been trained on movie conversations, I find film dialogue patterns really captivating."
            ]
            return random.choice(responses)
        
        elif any(word in text_lower for word in ['love', 'relationship', 'heart']):
            responses = [
                "Relationships are complex and beautiful. I've learned about them through thousands of movie conversations.",
                "Love is a universal theme I've encountered countless times in my training data from film dialogues.",
                "That touches my artificial heart! I've seen so many relationship dynamics in movie conversations."
            ]
            return random.choice(responses)
        
        elif any(word in text_lower for word in ['funny', 'joke', 'laugh', 'humor']):
            responses = [
                "I appreciate humor! Training on movie dialogues exposed me to everything from witty one-liners to slapstick comedy.",
                "Laughter is wonderful! I've learned comedic timing from thousands of movie conversations.",
                "Humor comes in so many forms - I've seen them all in the movie scripts I was trained on!"
            ]
            return random.choice(responses)
        
        elif any(word in text_lower for word in ['sad', 'upset', 'angry', 'frustrated']):
            responses = [
                "I understand those feelings. The movie conversations I learned from show the full spectrum of human emotions.",
                "Emotions are complex. My training on dramatic dialogues taught me to recognize and respond to different feelings.",
                "I can sense the emotion in your words. Movie characters have taught me about the depth of human experience."
            ]
            return random.choice(responses)
        
        elif any(word in text_lower for word in ['dream', 'future', 'hope', 'wish']):
            responses = [
                "Dreams and aspirations are powerful! I've seen characters pursue their dreams in countless movie stories.",
                "The future holds so much possibility. Movie narratives have taught me about hope and ambition.",
                "Following dreams is a common theme in the film conversations I learned from. What are you hoping for?"
            ]
            return random.choice(responses)
        
        # Check for questions
        elif '?' in user_input:
            return "That's a thoughtful question! While I may not have all the answers, my training on diverse movie conversations helps me engage meaningfully with many topics."
        
        return None
    
    def get_response(self, user_input: str) -> str:
        """Generate enhanced response using large dataset"""
        if not user_input or not user_input.strip():
            return "I didn't receive any message. Could you please say something?"
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # 1. Check FAQ first
        processed_input = self._preprocess_input(user_input)
        for key, response in self.faq_responses.items():
            if key in processed_input or any(word in processed_input for word in key.split()):
                self.conversation_history.append({"role": "bot", "content": response})
                return response
        
        # 2. Try to find similar conversation from large dataset
        similar_response = self._find_similar_conversation(user_input)
        if similar_response and len(similar_response.split()) > 2:
            self.conversation_history.append({"role": "bot", "content": similar_response})
            return similar_response
        
        # 3. Generate contextual response
        contextual_response = self._generate_contextual_response(user_input)
        if contextual_response:
            self.conversation_history.append({"role": "bot", "content": contextual_response})
            return contextual_response
        
        # 4. Use AI model if available
        if self.ai_mode:
            try:
                ai_response = self._generate_ai_response(user_input)
                if ai_response:
                    self.conversation_history.append({"role": "bot", "content": ai_response})
                    return ai_response
            except Exception as e:
                print(f"AI generation error: {e}")
        
        # 5. Fallback to enhanced responses
        fallbacks = [
            f"That's fascinating! My training on {len(self.large_dataset):,} conversations helps me appreciate diverse perspectives like yours.",
            "I find that intriguing! Having learned from thousands of movie dialogues, I love exploring different viewpoints.",
            "Thanks for sharing that! My extensive conversational training helps me engage with many topics.",
            "That's really interesting! The diversity in my training data from movie conversations helps me understand various subjects.",
            "I appreciate you telling me that! Learning from such a large dataset of real conversations makes every interaction unique."
        ]
        
        response = random.choice(fallbacks)
        self.conversation_history.append({"role": "bot", "content": response})
        return response
    
    def _generate_ai_response(self, text: str) -> Optional[str]:
        """Generate AI response if model is available"""
        if not self.ai_mode:
            return None
        
        try:
            input_text = text + self.tokenizer.eos_token
            encoded_input = self.tokenizer.encode(input_text, return_tensors='pt')
            
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
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(text):].strip()
            
            if len(response) > 5:
                return response
            
        except Exception as e:
            print(f"AI response generation failed: {e}")
        
        return None
    
    def get_stats(self) -> Dict:
        """Get enhanced chatbot statistics"""
        return {
            "total_exchanges": len(self.conversation_history) // 2,
            "training_examples": len(self.large_dataset),
            "faq_entries": len(self.faq_responses),
            "ai_mode": self.ai_mode,
            "dataset_source": "Cornell Movie Dialogs"
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared! Ready for a fresh chat with my enhanced capabilities."

# For backward compatibility
AIChChatBot = EnhancedChatBot
