#!/bin/bash

# GitHub Repository Setup Script for AI ChatBot
# Future-O-Tech Event

echo "🚀 Setting up GitHub repository for AI ChatBot..."

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "📥 Please install it first: brew install gh"
    echo "🔑 Then authenticate: gh auth login"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "🔑 Please authenticate with GitHub first:"
    echo "   gh auth login"
    exit 1
fi

# Create the repository
echo "📦 Creating GitHub repository..."
gh repo create ai-chatbot-future-o-tech \
    --public \
    --description "AI-powered chatbot for Future-O-Tech event using natural language processing and machine learning" \
    --clone=false

# Add remote origin
echo "🔗 Adding remote origin..."
git remote add origin https://github.com/$(gh api user --jq .login)/ai-chatbot-future-o-tech.git

# Push to GitHub
echo "📤 Pushing code to GitHub..."
git push -u origin main

echo "✅ GitHub repository created successfully!"
echo "🌐 Repository URL: https://github.com/$(gh api user --jq .login)/ai-chatbot-future-o-tech"
echo ""
echo "🎯 Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run the chatbot: python app.py"
echo "3. Visit: http://localhost:5000"
echo ""
echo "👥 Team members can now clone the repository and contribute!"
