# 📄 SMART Resume Analyzer - AI Powered ATS

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**SMART Resume Analyzer** एक advanced recruitment tool है जो AI का उपयोग करके Resumes को analyze करता है, AI-generated content को detect करता है, और Job Descriptions के साथ Match Score प्रदान करता है.

🚀 **Live Demo:** [resume.opeoluwaadeyericlub.tech](https://resume.opeoluwaadeyericlub.tech)

---

## ✨ Key Features

- ✅ **Deep Resume Parsing:** PDF और Docx files से text extract करता है.
- 🤖 **AI Content Detection:** पता लगाता है कि resume AI से लिखा गया है या इंसान ने.
- 🎯 **ATS Matching:** Job Role और Skills के आधार पर Resume Match Score देता है.
- 📧 **Automated Emailing:** Shortlisted और Rejected candidates को direct email भेजने की सुविधा.
- 📊 **PDF Reports:** HR के लिए पूरी analysis की PDF report generate करता है.
- 🛠 **Custom Requirements:** HR अपनी specific requirements add कर सकता है.

---

## 🛠 Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **Database:** SQLite (SQLAlchemy)
- **AI/ML:** NLTK, Scikit-learn, Textstat
- **PDF Handling:** PyMuPDF (Fitz), ReportLab, PyPDF2
- **Deployment:** Render.com

---

## 📁 Folder Structure

```text
SMART-Resume-Analyzer/
├── app.py              # Main Flask Entry Point
├── resume_analyzer.py  # Resume Analysis Logic
├── ai_detector.py      # AI Detection Engine
├── models.py           # Database Models
├── templates/          # HTML Interface (Jinja2)
├── static/             # CSS, JS, and Images
├── uploads/            # Temporary Resume Storage
├── Procfile            # Render Deployment Config
└── requirements.txt    # Project Dependencies
```

---

## 🚀 Installation & Local Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/GuptaSigma/SMART-Resume-Analyzer.git
   cd SMART-Resume-Analyzer
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```bash
   python app.py
   ```
   Website will be live at `http://127.0.0.1:5000`

---

## ☁️ Deployment on Render

यह project Render पर deploy करने के लिए optimized है.

- **Build Command:**
  `pip install --upgrade pip && pip install -r requirements.txt && python -m nltk.downloader punkt stopwords averaged_perceptron_tagger vader_lexicon punkt_tab`
- **Start Command:**
  `gunicorn app:app`
- **Environment Variable:**
  `PYTHON_VERSION` = `3.10.0`

---

## 👤 Author

**Sigma (Gupta Sagar)**  
- GitHub: [@GuptaSigma](https://github.com/GuptaSigma)
- Website: [opeoluwaadeyericlub.tech](https://opeoluwaadeyericlub.tech)

---
⭐️ **If you like this project, give it a star!**
