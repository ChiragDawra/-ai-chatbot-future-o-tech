#!/usr/bin/env python3
"""
Simple Model Training for AI ChatBot
Efficient fine-tuning approach using the large dataset
"""

import json
import torch
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import random

class ConversationDataset(Dataset):
    """Dataset class for conversation pairs"""
    
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Create input-response pair
        input_text = conv['input']
        response_text = conv['response']
        
        # Format: input <|endoftext|> response <|endoftext|>
        full_text = f"{input_text}{self.tokenizer.eos_token}{response_text}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'labels': encoded['input_ids'].flatten()
        }

class SimpleChatBotTrainer:
    """Simple, efficient trainer for the chatbot"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing simple trainer on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        
        print("✅ Model loaded successfully")
    
    def load_data(self, data_file: str, max_samples: int = 5000):
        """Load and sample training data"""
        print(f"📁 Loading data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Sample for efficient training
        if len(conversations) > max_samples:
            conversations = random.sample(conversations, max_samples)
            print(f"📊 Sampled {max_samples} conversations from {len(conversations)} total")
        
        return conversations
    
    def train(self, train_file: str, epochs: int = 2, batch_size: int = 2, learning_rate: float = 3e-5):
        """Train the model"""
        print("🎯 Starting model training...")
        
        # Load training data
        conversations = self.load_data(train_file, max_samples=5000)
        
        # Create dataset
        dataset = ConversationDataset(conversations, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        print(f"📊 Training on {len(conversations)} conversations")
        print(f"⚙️ Epochs: {epochs}, Batch size: {batch_size}")
        print(f"🔧 Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                
                # Log progress
                if step % 50 == 0:
                    print(f"   Step {step}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"   📊 Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
        
        avg_total_loss = total_loss / (len(dataloader) * epochs)
        print(f"\n✅ Training completed!")
        print(f"📊 Average training loss: {avg_total_loss:.4f}")
        
        return avg_total_loss
    
    def save_model(self, output_dir: str = "models/fine_tuned"):
        """Save the trained model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving model to: {output_path}")
        
        # Save model and tokenizer
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save training metadata
        metadata = {
            "base_model": self.model_name,
            "trained_on": "Cornell Movie Dialogs + Custom Dataset",
            "training_date": datetime.now().isoformat(),
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        with open(output_path / "model_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model saved successfully!")
        return str(output_path)

def main():
    """Main training function"""
    print("🤖 AI ChatBot - Simple Model Training")
    print("="*50)
    
    # Check if training data exists
    train_file = "data/large/train_conversations.json"
    
    if not os.path.exists(train_file):
        print("❌ Training data not found")
        print("📊 Run: python src/large_data_preparation.py")
        return
    
    try:
        # Initialize trainer
        trainer = SimpleChatBotTrainer()
        
        # Train the model
        loss = trainer.train(
            train_file=train_file,
            epochs=2,  # Quick training
            batch_size=2,  # Memory efficient
            learning_rate=3e-5
        )
        
        # Save the trained model
        model_path = trainer.save_model()
        
        print(f"\n🎉 Training pipeline completed!")
        print(f"📁 Trained model: {model_path}")
        print(f"📊 Final loss: {loss:.4f}")
        print(f"🚀 Ready to integrate with chatbot!")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 This might be due to memory constraints or large dataset size")
        print("🔄 Try reducing batch_size or max_samples in the script")
        return None

if __name__ == "__main__":
    main()
