# Vercel Deployment Guide

## Important Note

**Vercel has limitations for Flask/PyTorch applications:**

1. **Serverless Function Size Limit**: 50MB (your PyTorch model is 200MB)
2. **Execution Time Limit**: 10 seconds for Hobby plan
3. **Memory Limit**: 1024MB for Hobby plan

## Why You're Getting 404 Error

The 404 error occurs because:
- The model file (`plant_disease_model_1_latest.pt`) is too large for Vercel
- PyTorch dependencies are too heavy for serverless functions
- Flask app structure needs to be in the `api` folder for Vercel

## Recommended Alternatives

### Option 1: Render (Recommended)
Render supports Flask apps with large dependencies and files.

1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r "Flask Deployed App/requirements.txt"`
   - **Start Command**: `gunicorn --chdir "Flask Deployed App" app:app`
   - **Environment**: Python 3

### Option 2: Railway
Railway also supports Flask with PyTorch.

1. Go to [railway.app](https://railway.app)
2. Create new project from GitHub
3. It will auto-detect Flask and deploy

### Option 3: Heroku
Traditional but reliable for Flask apps.

1. Install Heroku CLI
2. Run:
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: PythonAnywhere
Free tier available, good for Flask apps.

1. Upload your code to PythonAnywhere
2. Configure WSGI file
3. Set working directory

## If You Still Want to Use Vercel

You would need to:
1. Host the model file externally (AWS S3, Google Drive, etc.)
2. Download the model on first request
3. Use a smaller model or quantized version
4. Optimize dependencies

## Files Created for Vercel

- `vercel.json` - Vercel configuration
- `api/index.py` - Vercel entry point
- `requirements.txt` - Root dependencies

## Next Steps

I recommend using **Render** or **Railway** instead of Vercel for this Flask + PyTorch application.
