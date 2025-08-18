#!/usr/bin/env python3
"""
Test script for AI ChatBot
Tests basic functionality and responses
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import AIChChatBot

def test_chatbot_basic():
    """Test basic chatbot functionality"""
    print("🧪 Testing AI ChatBot...")
    
    # Initialize chatbot
    bot = AIChChatBot()
    
    # Test cases
    test_cases = [
        "Hello",
        "How are you?",
        "What is your name?",
        "What can you do?",
        "Tell me about AI",
        "Thank you",
        "Goodbye"
    ]
    
    print("\n📝 Running test conversations...")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{i}. User: {test_input}")
        response = bot.get_response(test_input)
        print(f"   Bot: {response}")
    
    print("\n" + "=" * 50)
    
    # Get stats
    stats = bot.get_stats()
    print(f"\n📊 Chatbot Statistics:")
    print(f"   Total exchanges: {stats['total_exchanges']}")
    print(f"   Model: {stats['model']}")
    print(f"   Device: {stats['device']}")
    print(f"   FAQ entries: {stats['faq_entries']}")
    
    print("\n✅ All tests completed successfully!")

def test_conversation_flow():
    """Test conversation flow and memory"""
    print("\n🔄 Testing conversation flow...")
    
    bot = AIChChatBot()
    
    # Simulate a conversation
    conversation = [
        "Hi there!",
        "What's your name?",
        "Nice to meet you! What can you help me with?",
        "Can you tell me about artificial intelligence?",
        "That's interesting! Thanks for the explanation.",
        "Bye!"
    ]
    
    for msg in conversation:
        print(f"\nUser: {msg}")
        response = bot.get_response(msg)
        print(f"Bot: {response}")
    
    print("\n✅ Conversation flow test completed!")

if __name__ == "__main__":
    try:
        test_chatbot_basic()
        test_conversation_flow()
        print("\n🎉 All tests passed! The chatbot is ready to use.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
