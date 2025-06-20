# 🚀 Deployment Guide for Intelliread

## Option 1: Streamlit Cloud (Recommended - Free)

### Step 1: Prepare Your Repository
1. Make sure your code is in a GitHub repository
2. Ensure `requirements.txt` is in the root directory
3. Verify `main.py` is your main Streamlit file

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to: `main.py`
6. Click "Deploy"

### Step 3: Configure Secrets
In Streamlit Cloud dashboard, add these secrets:
```toml
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
```

### Step 4: Get Your URL
- Your app will be available at: `https://your-app-name.streamlit.app`
- This URL will be permanent and can be added to your resume

## Option 2: Heroku (Alternative)

### Step 1: Create Required Files
Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

Create `Procfile`:
```
web: sh setup.sh && streamlit run main.py
```

### Step 2: Deploy to Heroku
```bash
heroku create your-app-name
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## Option 3: Railway (Modern Alternative)

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables for API keys
4. Deploy automatically

## 🔧 Environment Variables Setup

For any deployment, you'll need these environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key

## 📝 Resume-Ready URL Format

Once deployed, your URL will look like:
- **Streamlit Cloud**: `https://intelliread-ai.streamlit.app`
- **Heroku**: `https://your-app-name.herokuapp.com`
- **Railway**: `https://your-app-name.railway.app`

## 🎯 Tips for Resume

1. **Choose a memorable name**: Use something like `intelliread-ai` or `pdf-ai-assistant`
2. **Test thoroughly**: Make sure the app works before adding to resume
3. **Add description**: Include a brief description of what the app does
4. **Keep it updated**: Maintain the deployment with latest features

## 🆘 Troubleshooting

### Common Issues:
- **Import errors**: Check `requirements.txt` has all dependencies
- **API key errors**: Verify secrets are properly configured
- **Memory issues**: Consider using lighter models for deployment

### Support:
- Streamlit Cloud: [docs.streamlit.io](https://docs.streamlit.io)
- Heroku: [devcenter.heroku.com](https://devcenter.heroku.com)
- Railway: [docs.railway.app](https://docs.railway.app)

## 🎉 Success!

Once deployed, you'll have a professional URL like:
**`https://intelliread-ai.streamlit.app`**

Perfect for your resume and portfolio! 🚀 