#!/usr/bin/env python3
"""
Setup script for AI ChatBot
Handles installation and initial setup
"""

import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", platform.python_version())
        return False
    
    print(f"✅ Python {platform.python_version()} detected")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("📦 Creating virtual environment...")
    
    success, stdout, stderr = run_command("python -m venv venv")
    if success:
        print("✅ Virtual environment created")
        return True
    else:
        print(f"❌ Failed to create virtual environment: {stderr}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📥 Installing dependencies...")
    
    # Determine the activation script based on OS
    if platform.system() == "Windows":
        pip_command = "venv\\Scripts\\pip install -r requirements.txt"
    else:
        pip_command = "venv/bin/pip install -r requirements.txt"
    
    success, stdout, stderr = run_command(pip_command)
    if success:
        print("✅ Dependencies installed successfully")
        return True
    else:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False

def prepare_data():
    """Prepare training data"""
    print("📊 Preparing training data...")
    
    # Determine the activation script based on OS
    if platform.system() == "Windows":
        python_command = "venv\\Scripts\\python src/data_preparation.py"
    else:
        python_command = "venv/bin/python src/data_preparation.py"
    
    success, stdout, stderr = run_command(python_command)
    if success:
        print("✅ Training data prepared")
        print(stdout)
        return True
    else:
        print(f"⚠️ Data preparation had issues: {stderr}")
        print("📋 Continuing with basic functionality...")
        return True  # Continue even if data prep fails

def main():
    """Main setup function"""
    print("🤖 AI ChatBot Setup - Future-O-Tech Event")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not create_virtual_environment():
            sys.exit(1)
    else:
        print("📦 Virtual environment already exists")
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️ Continuing without full dependencies...")
    
    # Prepare data
    prepare_data()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To run the chatbot:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("   python app.py")
    print("\n🌐 Then visit: http://localhost:5000")
    print("\n📚 Check README.md for detailed documentation")

if __name__ == "__main__":
    main()
