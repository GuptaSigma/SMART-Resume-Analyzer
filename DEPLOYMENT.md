# Deployment Guide

## Option 1: Vercel Serverless (Current Setup)

### Notes and Limitations
- File uploads and heavy processing can hit serverless timeouts.
- SQLite on Vercel is ephemeral; data is not persistent.
- NLTK downloads may fail on cold starts. If that happens, bundle data or switch to a persistent host.

### Vercel Steps
1. Push changes to GitHub.
2. Vercel Dashboard -> Add New Project -> Import repo.
3. Framework Preset: Other
4. Root Directory: repo root
5. Build Command: leave empty
6. Output Directory: leave empty
7. Environment Variables (Project Settings):
   - SESSION_SECRET=your-secret
   - UPLOAD_DIR=/tmp
   - DATABASE_URL=sqlite:////tmp/resume_analyzer.db
8. Deploy.

### Domain Setup (opeoluwaadeyericlub.tech)
- Add domain in Vercel.
- DNS Records:
  - Type A, Name @, Value 76.76.21.21
  - Type CNAME, Name www, Value cname.vercel-dns.com
- SSL will be issued automatically by Vercel once DNS propagates.

## Option 2: Docker (VPS or Local)

### Build and Run
```bash
docker build -t smart-hire-analyzer .
docker run -p 5000:5000 \
  -e SESSION_SECRET=change-me \
  -e UPLOAD_DIR=/app/instance \
  -e DATABASE_URL=sqlite:////app/resume_analyzer.db \
  smart-hire-analyzer
```

### Docker Compose
```bash
docker-compose up -d
```

### Reverse Proxy (Nginx Example)
```nginx
server {
    listen 80;
    server_name opeoluwaadeyericlub.tech www.opeoluwaadeyericlub.tech;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL (Let's Encrypt)
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d opeoluwaadeyericlub.tech -d www.opeoluwaadeyericlub.tech
```
