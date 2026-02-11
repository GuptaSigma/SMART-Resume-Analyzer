# Vercel Deployment Guide

## Quick Deploy Steps

### Via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy to production
vercel --prod
```

### Via Vercel Dashboard
1. Go to https://vercel.com
2. Click "Add New Project"
3. Select GitHub repository: `GuptaSigma/SMART-Resume-Analyzer`
4. Click "Deploy"

## Custom Domain Setup
1. Go to Project Settings → Domains in Vercel dashboard
2. Add domain: `opeoluwaadeyericlub.tech`
3. Follow Vercel's DNS configuration instructions

## Environment Variables
If your app needs environment variables, add them in:
- Vercel Dashboard → Project Settings → Environment Variables

## Troubleshooting
- Check Vercel deployment logs for errors
- Ensure all dependencies are in `requirements.txt`
- Verify Flask app imports correctly from nested directory
