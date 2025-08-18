# 🐙 GitHub Repository Setup

Since GitHub CLI is not installed, follow these steps to create your GitHub repository manually:

## Method 1: Using GitHub Website (Recommended)

### Step 1: Create Repository on GitHub
1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `ai-chatbot-future-o-tech`
   - **Description**: `AI-powered chatbot for Future-O-Tech event using natural language processing and machine learning`
   - **Visibility**: Public ✅
   - **DO NOT** initialize with README (we already have one)
5. Click "Create repository"

### Step 2: Connect Your Local Repository
After creating the repository on GitHub, run these commands in your terminal:

```bash
# Navigate to your project directory
cd "/Users/chiragdawra/Desktop/Ai ChatBot"

# Add the GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/ai-chatbot-future-o-tech.git

# Push your code to GitHub
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

## Method 2: Install GitHub CLI (Optional)

If you want to use the automated script:

```bash
# Install GitHub CLI
brew install gh

# Authenticate with GitHub
gh auth login

# Run the setup script
./setup_github.sh
```

## ✅ Verification

After setting up the repository, you should be able to:
1. See your code on GitHub at: `https://github.com/YOUR_USERNAME/ai-chatbot-future-o-tech`
2. Share the repository link with your team members
3. Enable collaboration by adding teammates as collaborators

## 👥 Adding Team Members

1. Go to your repository on GitHub
2. Click "Settings" tab
3. Click "Collaborators" in the left sidebar
4. Click "Add people"
5. Enter your teammates' GitHub usernames
6. Choose "Write" permission level
7. Send invitations

## 🎯 Next Steps

After setting up GitHub:
1. Install dependencies: `pip install -r requirements.txt`
2. Test the chatbot: `python test_chatbot.py`
3. Run the application: `python app.py`
4. Visit: http://localhost:5000

Your AI chatbot is ready for the Future-O-Tech event! 🚀
