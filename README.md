# SMART-Resume-Analyzer
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SMART-Resume-Analyzer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<h1>SMART-Resume-Analyzer</h1>
<p>Multi-resume parsing, job-role matching, AI content check, and HR email automation.</p>

<h2>Highlights</h2>
<ul>
  <li>Multi-upload + optional JD, role-aware match score.</li>
  <li>≥75% = Shortlisted, else Not Selected.</li>
  <li>AI% vs Human% + confidence with feature flags.</li>
  <li>Gmail SMTP (TLS) for congratulations/rejection emails.</li>
  <li>SQLite default; Postgres via DATABASE_URL.</li>
</ul>

<h2>Run (Local)</h2>
<pre>
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -U pip
pip install flask flask-sqlalchemy sqlalchemy werkzeug psycopg2-binary \
            pymupdf PyPDF2 python-docx reportlab textblob \
            numpy scipy scikit-learn nltk textstat language-tool-python
export FLASK_APP=app.py
flask run
# http://localhost:5000
</pre>

<h2>Routes</h2>
<ul>
  <li>GET /</li>
  <li>POST /analyze</li>
  <li>POST /send_email</li>
</ul>

<p>Repo: add your GitHub link here.</p>
</body>
</html>
