#!/usr/bin/env python3
"""
Large Dataset Preparation for AI ChatBot
Downloads and processes large conversational datasets for training
"""

import os
import json
import requests
import zipfile
import tarfile
import pandas as pd
from typing import List, Dict, Tuple
import re
from pathlib import Path
import random

# Optional imports with fallbacks
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import nltk
    HAS_ADVANCED_DEPS = True
except ImportError:
    HAS_ADVANCED_DEPS = False
    print("📦 Installing required packages...")

class LargeDatasetProcessor:
    """
    Downloads and processes large conversational datasets
    """
    
    def __init__(self, data_dir="data/large"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and info
        self.datasets = {
            "cornell_movie_dialogs": {
                "url": "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
                "size": "~50MB",
                "conversations": "~220k",
                "description": "Conversations from movie scripts"
            },
            "personachat": {
                "huggingface": "bavard/personachat_truecased",
                "size": "~16MB", 
                "conversations": "~10k",
                "description": "Persona-based conversations"
            },
            "dailydialog": {
                "huggingface": "daily_dialog",
                "size": "~20MB",
                "conversations": "~13k",
                "description": "Daily conversation topics"
            },
            "empathetic_dialogues": {
                "huggingface": "empathetic_dialogues",
                "size": "~25MB",
                "conversations": "~25k", 
                "description": "Empathetic conversations"
            },
            "blended_skill_talk": {
                "huggingface": "blended_skill_talk",
                "size": "~30MB",
                "conversations": "~27k",
                "description": "Multi-skill conversations"
            }
        }
        
        print("🚀 Large Dataset Processor initialized")
        print(f"📁 Data directory: {self.data_dir}")
    
    def list_available_datasets(self):
        """List all available datasets"""
        print("\n📊 Available Large Datasets:")
        print("="*60)
        
        for name, info in self.datasets.items():
            print(f"🗂️  {name.upper().replace('_', ' ')}")
            print(f"   📏 Size: {info['size']}")
            print(f"   💬 Conversations: {info['conversations']}")
            print(f"   📝 Description: {info['description']}")
            print()
    
    def download_cornell_movie_dialogs(self) -> List[Dict]:
        """Download and process Cornell Movie Dialogs corpus"""
        print("🎬 Downloading Cornell Movie Dialogs...")
        
        url = self.datasets["cornell_movie_dialogs"]["url"]
        zip_path = self.data_dir / "cornell_movie_dialogs.zip"
        extract_path = self.data_dir / "cornell_movie_dialogs"
        
        # Download if not exists
        if not zip_path.exists():
            print("📥 Downloading Cornell Movie Dialogs (50MB)...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download complete!")
        
        # Extract if not exists
        if not extract_path.exists():
            print("📦 Extracting archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("✅ Extraction complete!")
        
        # Process the data
        return self._process_cornell_dialogs(extract_path)
    
    def _process_cornell_dialogs(self, extract_path: Path) -> List[Dict]:
        """Process Cornell Movie Dialogs data"""
        print("🔄 Processing Cornell Movie Dialogs...")
        
        # Find the extracted folder
        extracted_folders = list(extract_path.glob("cornell*"))
        if not extracted_folders:
            raise Exception("Could not find extracted Cornell dialogs folder")
        
        dialog_folder = extracted_folders[0]
        
        # Read movie lines
        lines_file = dialog_folder / "movie_lines.txt"
        conversations_file = dialog_folder / "movie_conversations.txt"
        
        if not lines_file.exists() or not conversations_file.exists():
            raise Exception("Required Cornell dialog files not found")
        
        # Parse movie lines
        lines_dict = {}
        try:
            with open(lines_file, 'r', encoding='latin1') as f:
                for line in f:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 5:
                        line_id = parts[0]
                        text = parts[4]
                        lines_dict[line_id] = self._clean_text(text)
        except Exception as e:
            print(f"⚠️ Error reading lines file: {e}")
        
        # Parse conversations
        conversations = []
        try:
            with open(conversations_file, 'r', encoding='latin1') as f:
                for line in f:
                    parts = line.strip().split(' +++$+++ ')
                    if len(parts) >= 4:
                        line_ids = eval(parts[3])  # List of line IDs
                        
                        # Create conversation pairs
                        for i in range(len(line_ids) - 1):
                            input_id = line_ids[i]
                            response_id = line_ids[i + 1]
                            
                            if input_id in lines_dict and response_id in lines_dict:
                                conversations.append({
                                    'input': lines_dict[input_id],
                                    'response': lines_dict[response_id],
                                    'source': 'cornell_movie_dialogs'
                                })
        except Exception as e:
            print(f"⚠️ Error reading conversations file: {e}")
        
        print(f"✅ Processed {len(conversations)} Cornell movie conversations")
        return conversations
    
    def download_huggingface_dataset(self, dataset_name: str) -> List[Dict]:
        """Download dataset from Hugging Face"""
        if not HAS_ADVANCED_DEPS:
            print("❌ Advanced dependencies not available for Hugging Face datasets")
            return []
        
        print(f"🤗 Downloading {dataset_name} from Hugging Face...")
        
        try:
            if dataset_name == "personachat":
                dataset = load_dataset("bavard/personachat_truecased")
            elif dataset_name == "dailydialog":
                dataset = load_dataset("daily_dialog")
            elif dataset_name == "empathetic_dialogues":
                dataset = load_dataset("empathetic_dialogues")
            elif dataset_name == "blended_skill_talk":
                dataset = load_dataset("blended_skill_talk")
            else:
                print(f"❌ Unknown dataset: {dataset_name}")
                return []
            
            return self._process_huggingface_dataset(dataset, dataset_name)
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return []
    
    def _process_huggingface_dataset(self, dataset, dataset_name: str) -> List[Dict]:
        """Process Hugging Face dataset into conversation pairs"""
        conversations = []
        
        try:
            if dataset_name == "dailydialog":
                # Process Daily Dialog
                train_data = dataset['train']
                for example in train_data:
                    dialog = example['dialog']
                    for i in range(len(dialog) - 1):
                        if dialog[i].strip() and dialog[i+1].strip():
                            conversations.append({
                                'input': self._clean_text(dialog[i]),
                                'response': self._clean_text(dialog[i+1]),
                                'source': 'daily_dialog'
                            })
            
            elif dataset_name == "personachat":
                # Process PersonaChat
                train_data = dataset['train']
                for example in train_data:
                    history = example.get('history', [])
                    candidates = example.get('candidates', [])
                    
                    if history and candidates:
                        input_text = " ".join(history[-2:])  # Last 2 turns
                        response_text = candidates[0] if candidates else ""
                        
                        if input_text.strip() and response_text.strip():
                            conversations.append({
                                'input': self._clean_text(input_text),
                                'response': self._clean_text(response_text),
                                'source': 'personachat'
                            })
            
            elif dataset_name == "empathetic_dialogues":
                # Process Empathetic Dialogues
                train_data = dataset['train']
                for example in train_data:
                    utterance = example.get('utterance', '')
                    response = example.get('utterance', '')  # Simplified
                    
                    if utterance.strip() and len(utterance.split()) > 2:
                        conversations.append({
                            'input': self._clean_text(utterance),
                            'response': self._clean_text(response),
                            'source': 'empathetic_dialogues'
                        })
            
            elif dataset_name == "blended_skill_talk":
                # Process Blended Skill Talk
                train_data = dataset['train']
                for example in train_data:
                    previous_utterance = example.get('previous_utterance', [])
                    free_message = example.get('free_message', '')
                    
                    if previous_utterance and free_message:
                        input_text = " ".join(previous_utterance[-1:])
                        if input_text.strip() and free_message.strip():
                            conversations.append({
                                'input': self._clean_text(input_text),
                                'response': self._clean_text(free_message),
                                'source': 'blended_skill_talk'
                            })
        
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
        
        print(f"✅ Processed {len(conversations)} conversations from {dataset_name}")
        return conversations
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Remove very short or very long messages
        if len(text) < 3 or len(text) > 500:
            return ""
        
        return text
    
    def download_all_datasets(self) -> List[Dict]:
        """Download and process all available datasets"""
        print("🚀 Downloading ALL available datasets...")
        print("⚠️  This will download ~150MB of data and may take 10-15 minutes")
        
        all_conversations = []
        
        # Download Cornell Movie Dialogs
        try:
            cornell_conversations = self.download_cornell_movie_dialogs()
            all_conversations.extend(cornell_conversations)
        except Exception as e:
            print(f"❌ Failed to download Cornell dialogs: {e}")
        
        # Download Hugging Face datasets
        if HAS_ADVANCED_DEPS:
            hf_datasets = ["dailydialog", "personachat", "empathetic_dialogues", "blended_skill_talk"]
            
            for dataset_name in hf_datasets:
                try:
                    hf_conversations = self.download_huggingface_dataset(dataset_name)
                    all_conversations.extend(hf_conversations)
                except Exception as e:
                    print(f"❌ Failed to download {dataset_name}: {e}")
        
        # Remove duplicates and shuffle
        unique_conversations = list({json.dumps(conv, sort_keys=True): conv for conv in all_conversations}.values())
        random.shuffle(unique_conversations)
        
        print(f"\n🎉 Downloaded and processed {len(unique_conversations)} unique conversations!")
        print(f"📊 Dataset breakdown:")
        
        # Count by source
        source_counts = {}
        for conv in unique_conversations:
            source = conv.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            print(f"   📂 {source}: {count:,} conversations")
        
        return unique_conversations
    
    def save_processed_dataset(self, conversations: List[Dict], filename: str = "large_conversations.json"):
        """Save processed conversations to file"""
        output_file = self.data_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(conversations)} conversations to: {output_file}")
        return str(output_file)
    
    def create_training_splits(self, conversations: List[Dict], train_ratio: float = 0.8):
        """Split data into training and validation sets"""
        random.shuffle(conversations)
        
        split_idx = int(len(conversations) * train_ratio)
        train_data = conversations[:split_idx]
        val_data = conversations[split_idx:]
        
        # Save splits
        train_file = self.save_processed_dataset(train_data, "train_conversations.json")
        val_file = self.save_processed_dataset(val_data, "val_conversations.json")
        
        print(f"📊 Training split: {len(train_data)} conversations")
        print(f"📊 Validation split: {len(val_data)} conversations")
        
        return train_file, val_file

def main():
    """Main execution function"""
    processor = LargeDatasetProcessor()
    
    print("🤖 AI ChatBot - Large Dataset Preparation")
    print("="*50)
    
    # Show available datasets
    processor.list_available_datasets()
    
    print("🚀 Starting large dataset download and processing...")
    print("⏳ This may take 10-15 minutes depending on your internet connection")
    
    # Download all datasets
    all_conversations = processor.download_all_datasets()
    
    if all_conversations:
        # Save complete dataset
        dataset_file = processor.save_processed_dataset(all_conversations)
        
        # Create training splits
        train_file, val_file = processor.create_training_splits(all_conversations)
        
        print(f"\n🎉 Large dataset preparation complete!")
        print(f"📁 Complete dataset: {dataset_file}")
        print(f"📁 Training data: {train_file}")
        print(f"📁 Validation data: {val_file}")
        print(f"💬 Total conversations: {len(all_conversations):,}")
        
        return dataset_file, train_file, val_file
    else:
        print("❌ No datasets were successfully downloaded")
        return None, None, None

if __name__ == "__main__":
    main()
