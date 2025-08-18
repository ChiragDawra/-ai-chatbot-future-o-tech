# 🚀 Deployment Guide - AI ChatBot

## 🏠 Local Development

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (optional - has fallbacks)
python src/data_preparation.py

# 3. Run the application
python app.py

# 4. Visit the chatbot
# Open browser to: http://localhost:5000
```

### With Virtual Environment (Recommended)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the chatbot
python app.py
```

## 🧪 Testing

### Run Test Suite
```bash
# Test chatbot functionality
python test_chatbot.py

# Test web interface
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Manual Testing Checklist
- [ ] Chatbot responds to greetings
- [ ] FAQ system works correctly
- [ ] Web interface loads properly
- [ ] Messages display in real-time
- [ ] Error handling works
- [ ] Mobile responsiveness

## 🌐 Production Deployment

### Option 1: Heroku (Free Tier)
```bash
# 1. Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# 2. Create Procfile
echo "web: gunicorn app:app" > Procfile

# 3. Create Heroku app
heroku create your-chatbot-name

# 4. Deploy
git push heroku main
```

### Option 2: Railway (Free Tier)
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway link
railway up
```

### Option 3: Render (Free Tier)
1. Connect your GitHub repository to Render
2. Choose "Web Service"
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python app.py`

## ⚙️ Configuration

### Environment Variables
Create a `.env` file for production:
```
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
PORT=5000
```

### Production Settings
Update `app.py` for production:
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

## 📊 Monitoring

### Health Check
- Endpoint: `/health`
- Expected response: `{"status": "healthy", "chatbot": "ready"}`

### Logs
- Conversation logs: `data/conversation_logs.jsonl`
- Application logs: Console output

## 🔧 Troubleshooting

### Common Issues

**Issue**: Dependencies not installing
```bash
# Solution: Upgrade pip
pip install --upgrade pip
pip install -r requirements.txt
```

**Issue**: Model not loading
```bash
# Solution: Check internet connection and disk space
# The app will fall back to rule-based responses
```

**Issue**: Port already in use
```bash
# Solution: Use different port
python app.py --port 5001
# Or kill existing process:
lsof -ti:5000 | xargs kill
```

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **Storage**: 2GB+ free space (for model downloads)

## 🎯 Performance Optimization

### For Better Response Times
1. **Use GPU**: If available, the app will automatically use CUDA
2. **Model Caching**: Models are cached after first load
3. **FAQ Priority**: FAQ responses are faster than AI generation

### For Production
1. **Use gunicorn**: Better than Flask's development server
2. **Redis Cache**: Add Redis for conversation caching
3. **Load Balancer**: Use nginx for multiple instances

## 🛡️ Security

### For Production Deployment
- [ ] Change default secret key
- [ ] Enable HTTPS
- [ ] Add rate limiting
- [ ] Sanitize user inputs
- [ ] Monitor for abuse

## 📱 Mobile Optimization

The chatbot is already mobile-responsive, but for better mobile experience:
- Consider PWA (Progressive Web App) features
- Add touch gestures
- Optimize for small screens
- Test on various devices

---

**Need help?** Check the README.md or create an issue on GitHub!
