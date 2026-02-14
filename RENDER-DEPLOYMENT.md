# Render.com Deployment Guide

## Prerequisites
- GitHub repository with the flattened structure
- Render.com account (free tier available)

## Deployment Steps

### 1. Create New Web Service
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `GuptaSigma/SMART-Resume-Analyzer`
4. Configure the service:
   - **Name**: `smart-resume-analyzer` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or paid for better performance)

### 2. Environment Variables (Optional)
Add these in Render Dashboard → Environment:
- `SESSION_SECRET`: A random secret key for Flask sessions
- `UPLOAD_DIR`: `/tmp` (recommended for serverless/ephemeral storage)
- `DATABASE_URL`: For production database (SQLite is default)

### 3. NLTK Data Setup
The build command automatically downloads required NLTK data:
- `stopwords`: For text processing
- `punkt`: For sentence tokenization
- `averaged_perceptron_tagger`: For POS tagging
- `wordnet`: For semantic analysis

If you need additional NLTK data, add it to the build command.

**Note on Grammar Checking**: The `language-tool-python` package requires Java Runtime Environment (JRE) to be installed. Since Java is not available by default on Render's free tier, grammar checking will be disabled. The application will work normally with a warning message. If you need grammar checking, you'll need to either:
- Use a paid tier with custom Docker container that includes Java
- Remove `language-tool-python` from `requirements.txt` to avoid the warning

### 4. Deploy
1. Click "Create Web Service"
2. Render will automatically deploy from your main branch
3. Monitor the deployment logs for any errors
4. Once deployed, your app will be available at: `https://your-service-name.onrender.com`

## Automatic Deployments
Render automatically redeploys when you push to your main branch.

## Custom Domain Setup
1. Go to Settings → Custom Domain in Render dashboard
2. Add your custom domain
3. Update your DNS records as instructed by Render

## Important Notes
- **Uploads Directory**: The `/tmp` directory is ephemeral on Render. Files uploaded during runtime will be lost on redeploy. For persistent storage, consider using cloud storage (S3, Cloudinary, etc.)
- **Database**: SQLite works but is also ephemeral. For production, use PostgreSQL (free tier available on Render)
- **Cold Starts**: Free tier instances may spin down after inactivity and take 30-60 seconds to wake up

## Troubleshooting
- Check deployment logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify NLTK data downloaded successfully in build logs
- Check that `app:app` correctly references the Flask app instance

## Upgrading to Paid Plans
For production use, consider:
- Paid instance type for better performance (no cold starts)
- PostgreSQL database for persistence
- Custom domain with SSL (automatic with Render)
