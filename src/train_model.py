#!/usr/bin/env python3
"""
Model Training Script for AI ChatBot
Fine-tunes a conversational model on large datasets
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

# Optional imports with fallbacks
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import wandb  # For experiment tracking
    HAS_TRAINING_DEPS = True
except ImportError:
    HAS_TRAINING_DEPS = False
    print("📦 Training dependencies not found. Install with: pip install transformers datasets wandb")

class ChatBotTrainer:
    """
    Handles model training for the AI ChatBot
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", output_dir="models/trained"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing trainer on {self.device}")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if HAS_TRAINING_DEPS:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model and tokenizer loaded successfully")
        else:
            print("❌ Training dependencies not available")
    
    def load_training_data(self, data_file: str) -> List[Dict]:
        """Load training data from JSON file"""
        print(f"📁 Loading training data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        print(f"💬 Loaded {len(conversations)} conversation pairs")
        return conversations
    
    def prepare_training_dataset(self, conversations: List[Dict], max_length: int = 512):
        """Prepare dataset for training"""
        if not HAS_TRAINING_DEPS:
            print("❌ Cannot prepare training dataset without dependencies")
            return None
        
        print("🔄 Preparing training dataset...")
        
        # Format conversations for training
        formatted_data = []
        for conv in conversations:
            input_text = conv['input']
            response_text = conv['response']
            
            # Create training text in format: input + response
            full_text = f"{input_text}{self.tokenizer.eos_token}{response_text}{self.tokenizer.eos_token}"
            
            # Tokenize and check length
            tokens = self.tokenizer.encode(full_text)
            if len(tokens) <= max_length:
                formatted_data.append(full_text)
        
        print(f"✅ Prepared {len(formatted_data)} training examples")
        
        # Create HuggingFace dataset
        dataset = Dataset.from_dict({"text": formatted_data})
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train_model(self, train_dataset, val_dataset=None, epochs=3, batch_size=4, learning_rate=5e-5):
        """Train the model on the dataset"""
        if not HAS_TRAINING_DEPS:
            print("❌ Cannot train model without dependencies")
            return None
        
        print("🎯 Starting model training...")
        print(f"📊 Training examples: {len(train_dataset)}")
        print(f"⚙️ Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            logging_dir='./logs',
            report_to=None,  # Disable wandb for now
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Start training
        print("🚀 Training started...")
        train_result = trainer.train()
        
        # Save the model
        print("💾 Saving trained model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "training_examples": len(train_dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_loss": train_result.training_loss,
            "timestamp": datetime.now().isoformat(),
            "device": self.device
        }
        
        info_file = self.output_dir / "training_info.json"
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {self.output_dir}")
        print(f"📊 Final training loss: {train_result.training_loss:.4f}")
        
        return str(self.output_dir)
    
    def quick_train(self, data_file: str):
        """Quick training pipeline"""
        print("⚡ Starting quick training pipeline...")
        
        # Load data
        conversations = self.load_training_data(data_file)
        
        # Limit dataset size for quick training (adjust based on your system)
        max_examples = min(10000, len(conversations))  # Use up to 10k examples
        conversations = conversations[:max_examples]
        
        print(f"🎯 Using {len(conversations)} conversations for training")
        
        # Prepare dataset
        train_dataset = self.prepare_training_dataset(conversations)
        
        if train_dataset is None:
            return None
        
        # Train with lightweight settings
        model_path = self.train_model(
            train_dataset=train_dataset,
            epochs=2,  # Quick training
            batch_size=2,  # Small batch for memory efficiency
            learning_rate=3e-5
        )
        
        return model_path

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing training dependencies...")
    
    dependencies = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "wandb"
    ]
    
    for dep in dependencies:
        try:
            os.system(f"pip install {dep}")
            print(f"✅ Installed {dep}")
        except Exception as e:
            print(f"❌ Failed to install {dep}: {e}")

def main():
    """Main training function"""
    print("🤖 AI ChatBot - Model Training")
    print("="*50)
    
    # Check if we have training data
    train_data_file = "data/large/train_conversations.json"
    
    if not os.path.exists(train_data_file):
        print("📊 Training data not found. Preparing large dataset first...")
        
        # Import and run large data preparation
        from large_data_preparation import main as prepare_data
        dataset_file, train_file, val_file = prepare_data()
        
        if train_file:
            train_data_file = train_file
        else:
            print("❌ Could not prepare training data")
            return
    
    # Check dependencies
    if not HAS_TRAINING_DEPS:
        print("❌ Training dependencies not found")
        print("📦 Run: pip install transformers datasets torch accelerate")
        return
    
    # Initialize trainer
    trainer = ChatBotTrainer()
    
    # Start training
    model_path = trainer.quick_train(train_data_file)
    
    if model_path:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Trained model saved to: {model_path}")
        print(f"🚀 You can now use the trained model in your chatbot!")
        
        # Update the chatbot to use the trained model
        print("\n🔄 To use the trained model:")
        print("1. Update src/chatbot.py to load from models/trained/")
        print("2. Test with: python test_chatbot.py")
        print("3. Redeploy: netlify deploy --prod")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()
