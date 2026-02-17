# Yeh code `app.py` file ke liye hai.

import os
import logging
from pathlib import Path
from datetime import datetime
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# Local modules
from extensions import db
from resume_analyzer import ResumeAnalyzer
from email_service import EmailService
from ai_detector import AIDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# configure the database (absolute path prevents multiple DB files)
basedir = os.path.abspath(os.path.dirname(__file__))
default_db_path = os.path.join(basedir, "resume_analyzer.db")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{default_db_path}")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_recycle": 300, "pool_pre_ping": True}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16MB max file size

# Uploads directory (override in serverless with UPLOAD_DIR=/tmp)
upload_root = os.environ.get("UPLOAD_DIR", app.instance_path)
UPLOAD_DIR = Path(upload_root) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Uploads directory: {UPLOAD_DIR}")

# initialize DB
db.init_app(app)

# ensure tables exist
with app.app_context():
    import models # register models
    db.create_all()
    logging.info("Database tables ensured/created.")

# services
pdf_max_pages = 10
try:
    pdf_max_pages = int(os.environ.get("PDF_MAX_PAGES", "10"))
except Exception:
    pdf_max_pages = 10

# AI detector ko bina kisi override ke initialize karein
resume_analyzer = ResumeAnalyzer(pdf_max_pages=pdf_max_pages)
email_service = EmailService()
ai_detector = AIDetector()  # Yeh ab bina hardcoded values ke kaam karega

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# ==================== Resume Analysis ====================
@app.route('/analyze', methods=['POST'])
def analyze_resumes():
    try:
        company_name = request.form.get('company_name')
        hr_name = request.form.get('hr_name')
        hr_email = request.form.get('hr_email')
        email_password = request.form.get('email_password')
        target_role = request.form.get('target_role')
        hr_requirements = request.form.get('hr_requirements', '')

        if not all([company_name, hr_name, hr_email, email_password]):
            flash('Please fill in all required company information fields.', 'error')
            return redirect(url_for('index'))

        resume_files = request.files.getlist('resume_files')
        jd_file = request.files.get('jd_file')

        if not resume_files or resume_files[0].filename == '':
            flash('Please upload at least one resume file.', 'error')
            return redirect(url_for('index'))

        jd_content = ""
        if jd_file and jd_file.filename and allowed_file(jd_file.filename):
            try:
                # Save JD temporarily to reuse extractor uniformly
                jd_secure = secure_filename(jd_file.filename)
                jd_path = UPLOAD_DIR / f"JD_{jd_secure}"
                jd_file.stream.seek(0)
                jd_file.save(jd_path)
                with jd_path.open("rb") as f:
                    jd_content = resume_analyzer.extract_text_from_file(f)
                try:
                    jd_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"Error processing JD file: {e}")
                flash('Error processing job description file. Proceeding without it.', 'warning')

        email_service.setup(hr_email, email_password)

        # Create a new AnalysisResult record
        import models
        new_analysis_result = models.AnalysisResult(
            company_name=company_name,
            hr_name=hr_name,
            hr_email=hr_email,
            total_processed=0,
            shortlisted_count=0
        )
        db.session.add(new_analysis_result)
        db.session.commit()

        candidates = []
        total_processed = 0
        ai_detected_count = 0
        shortlisted_count = 0

        for file in resume_files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    # Save the uploaded resume to disk (stable path for later AI check)
                    secured_name = secure_filename(file.filename)
                    save_path = UPLOAD_DIR / secured_name
                    file.stream.seek(0)
                    file.save(save_path)
                    logger.info(f"Saved upload: {save_path}")

                    # Extract text for analysis using the saved file
                    with save_path.open("rb") as f:
                        candidate_data = resume_analyzer.analyze_resume(
                            file=f,
                            target_role=target_role,
                            jd_content=jd_content,
                            hr_requirements=hr_requirements
                        )

                    # Extract text again for AI detection (separate pass; keeps concerns clear)
                    with save_path.open("rb") as f:
                        text_content = resume_analyzer.extract_text_from_file(f)

                    # Run AI detector on the actual resume text (NO override/hard-code)
                    ai_analysis = ai_detector.analyze_resume(
                        text_content,
                        candidate_data.get('name', 'Unknown'),
                        secured_name
                    )
                    is_ai_generated = ai_analysis.get('is_ai_generated', False)

                    is_shortlisted = candidate_data.get('match_score', 0) >= 75
                    if is_shortlisted:
                        shortlisted_count += 1

                    # Save to DB with the same, stable filename
                    new_candidate = models.Candidate(
                        analysis_id=new_analysis_result.id,
                        name=candidate_data.get('name'),
                        email=candidate_data.get('email'),
                        phone=candidate_data.get('phone'),
                        match_score=candidate_data.get('match_score'),
                        target_role=candidate_data.get('target_role'),
                        experience_years=candidate_data.get('experience_years', 0),
                        education=candidate_data.get('education'),
                        university=candidate_data.get('university'),
                        is_shortlisted=is_shortlisted,
                        is_ai_generated=is_ai_generated,
                        resume_filename=secured_name,
                        # --- AI Analysis features saved here ---
                        ai_percentage=ai_analysis.get('ai_percentage'),
                        ai_confidence=ai_analysis.get('ai_confidence'),
                        ai_features=ai_analysis.get('features')
                    )
                    db.session.add(new_candidate)
                    db.session.commit()

                    # Ensure UI data matches DB
                    candidate_data['resume_filename'] = secured_name
                    candidate_data['is_ai_generated'] = is_ai_generated

                    # Optionally include AI summary in the in-memory list (for results page)
                    candidate_data['ai_percentage'] = ai_analysis.get('ai_percentage')
                    candidate_data['human_percentage'] = ai_analysis.get('human_percentage')
                    candidate_data['ai_confidence'] = ai_analysis.get('ai_confidence')

                    candidates.append(candidate_data)
                    total_processed += 1
                    if is_ai_generated:
                        ai_detected_count += 1

                except Exception as e:
                    logging.exception(f"Error processing {file.filename}: {e}")
                    flash(f'Error processing {file.filename}: {str(e)}', 'warning')

        new_analysis_result.total_processed = total_processed
        new_analysis_result.shortlisted_count = shortlisted_count
        db.session.commit()

        if not candidates:
            flash('No resumes could be processed successfully.', 'error')
            return redirect(url_for('index'))

        shortlisted = [c for c in candidates if c.get('match_score', 0) >= 75]
        not_selected = [c for c in candidates if c.get('match_score', 0) < 75]

        return render_template(
            'results.html',
            company_name=company_name,
            hr_name=hr_name,
            hr_email=hr_email,
            shortlisted=shortlisted,
            not_selected=not_selected,
            total_processed=total_processed,
            ai_detected=ai_detected_count
        )
    except Exception as e:
        logging.exception(f"Error in analyze_resumes: {e}")
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

# ==================== Email Routes ====================
@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.get_json() or {}
        email = data.get('email')
        name = data.get('name')
        email_type = data.get('type')

        if not all([email, name, email_type]):
            return jsonify({'success': False, 'message': 'Missing required fields'})

        import models
        # Find the most recent candidate record for this email+name
        candidate_record = models.Candidate.query.filter_by(email=email, name=name).order_by(models.Candidate.created_at.desc()).first()
        if not candidate_record:
            return jsonify({'success': False, 'message': 'Candidate not found'})

        # Prevent wrong email type to wrong list
        if email_type == 'congratulations' and not candidate_record.is_shortlisted:
            return jsonify({'success': False, 'message': 'Cannot send congratulations to a not-selected candidate'})
        if email_type == 'rejection' and candidate_record.is_shortlisted:
            return jsonify({'success': False, 'message': 'Cannot send rejection to a shortlisted candidate'})

        # Idempotency: prevent duplicate emails
        if candidate_record.email_sent:
            return jsonify({'success': False, 'message': 'Email already sent to this candidate'})

        success = email_service.send_email(email, name, email_type)

        if success:
            candidate_record.email_sent = True
            candidate_record.sent_on = datetime.now()
            db.session.commit()
            return jsonify({'success': True, 'message': 'Email sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to send email'})
    except Exception as e:
        logging.exception(f"Error sending email: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/send_all_rejections', methods=['POST'])
def send_all_rejections():
    try:
        data = request.get_json() or {}
        emails = data.get('emails', [])
        if not emails:
            return jsonify({'success': False, 'message': 'No emails provided'})

        import models
        # Only not-selected and not already emailed
        candidates = models.Candidate.query.filter(
            models.Candidate.email.in_(emails),
            models.Candidate.is_shortlisted.is_(False)
        ).all()

        sent_count = 0
        skipped_count = 0

        for c in candidates:
            if not c.email or c.email_sent:
                skipped_count += 1
                continue
            if email_service.send_email(c.email, c.name, "rejection"):
                c.email_sent, c.sent_on = True, datetime.now()
                sent_count += 1
            else:
                skipped_count += 1
        db.session.commit()

        return jsonify({'success': True, 'sent_count': sent_count, 'skipped_count': skipped_count, 'total_count': len(candidates)})
    except Exception as e:
        logging.exception(f"Error sending all rejection emails: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/send_all_congratulations', methods=['POST'])
def send_all_congratulations():
    try:
        data = request.get_json() or {}
        emails = data.get('emails', [])
        if not emails:
            return jsonify({'success': False, 'message': 'No emails provided'})

        import models
        # Only shortlisted and not already emailed
        candidates = models.Candidate.query.filter(
            models.Candidate.email.in_(emails),
            models.Candidate.is_shortlisted.is_(True)
        ).all()

        sent_count = 0
        skipped_count = 0

        for c in candidates:
            if not c.email or c.email_sent:
                skipped_count += 1
                continue
            if email_service.send_email(c.email, c.name, "congratulations"):
                c.email_sent, c.sent_on = True, datetime.now()
                sent_count += 1
            else:
                skipped_count += 1
        db.session.commit()

        return jsonify({'success': True, 'sent_count': sent_count, 'skipped_count': skipped_count, 'total_count': len(candidates)})
    except Exception as e:
        logging.exception(f"Error sending all congratulatory emails: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Helper API: Check if selected candidates are ready (emails sent)
@app.route('/api/selected_ready/<company_name>')
def selected_ready(company_name):
    try:
        import models
        analysis_result = models.AnalysisResult.query.filter_by(company_name=company_name).order_by(models.AnalysisResult.created_at.desc()).first()
        if not analysis_result:
            return jsonify({'success': False, 'ready_count': 0, 'message': 'No analysis found'}), 404

        ready_count = models.Candidate.query.filter_by(
            analysis_id=analysis_result.id,
            is_shortlisted=True,
            email_sent=True
        ).filter(models.Candidate.sent_on.isnot(None)).count()

        return jsonify({'success': True, 'ready_count': ready_count})
    except Exception as e:
        logging.exception(f"Error checking selected readiness: {e}")
        return jsonify({'success': False, 'ready_count': 0, 'message': str(e)}), 500

# ==================== AI Check ====================
# UI calls /ai_check/<filename>; query by resume_filename
@app.route('/ai_check/<path:filename>')
def ai_check(filename):
    try:
        import models
        secured = secure_filename(filename)
        candidate_data = models.Candidate.query.filter_by(resume_filename=secured).first()
        if not candidate_data:
            candidate_data = models.Candidate.query.filter_by(resume_filename=filename).first()
        if not candidate_data:
            flash('Candidate not found in analysis results.', 'error')
            return redirect(url_for('index'))

        # Recompute AI analysis using the saved resume file (no hard-coded scores)
        file_path = UPLOAD_DIR / (candidate_data.resume_filename or secured)
        if not file_path.exists():
            # Agar file nahi milti, toh user ko batayein aur gracefully fall back karein.
            logging.warning(f"Saved resume file not found for AI check: {file_path}")
            flash('Original file not found. Analysis results may not be accurate.', 'warning')
            return redirect(url_for('index'))

        # Extract text and run detector
        with file_path.open("rb") as f:
            text_content = resume_analyzer.extract_text_from_file(f)

        ai_analysis = ai_detector.analyze_resume(
            text_content,
            candidate_data.name or "Unknown",
            candidate_data.resume_filename or secured
        )

        # Render real analysis
        return render_template('ai_check.html', analysis=ai_analysis)
    except Exception as e:
        logging.exception(f"Error in AI check for {filename}: {e}")
        flash(f'An unexpected error occurred during AI check: {str(e)}', 'error')
        return redirect(url_for('index'))

# ==================== PDF Reports ====================
@app.route('/download_report/<company_name>')
def download_report(company_name):
    try:
        import models
        analysis_result = models.AnalysisResult.query.filter_by(company_name=company_name).order_by(models.AnalysisResult.created_at.desc()).first()
        if not analysis_result:
            flash('No analysis results found for download.', 'error')
            return redirect(url_for('index'))

        all_candidates = models.Candidate.query.filter_by(analysis_id=analysis_result.id).all()
        shortlisted = [c for c in all_candidates if c.is_shortlisted]
        not_selected = [c for c in all_candidates if not c.is_shortlisted]

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        flowables.append(Paragraph(f"<b>Complete Resume Analysis Report for {analysis_result.company_name}</b>", styles['Title']))
        flowables.append(Spacer(1, 12))
        flowables.append(Paragraph(f"<b>HR:</b> {analysis_result.hr_name} | <b>Date:</b> {analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        flowables.append(Paragraph(f"<b>Total Processed:</b> {analysis_result.total_processed}", styles['Normal']))
        flowables.append(Spacer(1, 24))

        # Shortlisted table
        flowables.append(Paragraph("<b><u>Shortlisted Candidates</u></b>", styles['h2']))
        flowables.append(Spacer(1, 12))
        if shortlisted:
            data = [['S.No.', 'Name', 'Match Score', 'Experience', 'Education', 'AI Detected', 'Email Sent']]
            for i, c in enumerate(shortlisted, 1):
                email_status = "Yes" if (c.email_sent and c.sent_on) else "No"
                data.append([str(i), Paragraph(c.name or "N/A", styles['Normal']),
                              f"{c.match_score}%", f"{c.experience_years} years",
                              c.education or "N/A", "Yes" if c.is_ai_generated else "No",
                              email_status])
            table = Table(data, colWidths=[0.5*inch, 1.5*inch, 0.75*inch, 1*inch, 1.75*inch, 0.75*inch, 0.75*inch])
            table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN',(0,0),(-1,-1),'CENTER')]))
            flowables.append(table)
        else:
            flowables.append(Paragraph("<i>No candidates shortlisted.</i>", styles['Italic']))
        flowables.append(Spacer(1, 48))

        # Not selected table
        flowables.append(Paragraph("<b><u>Not Selected Candidates</u></b>", styles['h2']))
        flowables.append(Spacer(1, 12))
        if not_selected:
            data = [['S.No.', 'Name', 'Match Score', 'Experience', 'Education', 'AI Detected', 'Email Sent']]
            for i, c in enumerate(not_selected, 1):
                email_status = "Yes" if (c.email_sent and c.sent_on) else "No"
                data.append([str(i), Paragraph(c.name or "N/A", styles['Normal']),
                              f"{c.match_score}%", f"{c.experience_years} years",
                              c.education or "N/A", "Yes" if c.is_ai_generated else "No",
                              email_status])
            table = Table(data, colWidths=[0.5*inch, 1.5*inch, 0.75*inch, 1*inch, 1.75*inch, 0.75*inch, 0.75*inch])
            table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN',(0,0),(-1,-1),'CENTER')]))
            flowables.append(table)
        else:
            flowables.append(Paragraph("<i>All candidates shortlisted.</i>", styles['Italic']))

        doc.build(flowables)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{company_name}_full_report.pdf", mimetype='application/pdf')
    except Exception as e:
        logging.exception(f"Error creating PDF: {e}")
        flash(f'Error creating PDF: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download_all_selected/<company_name>')
def download_all_selected(company_name):
    try:
        import models
        analysis_result = models.AnalysisResult.query.filter_by(company_name=company_name).order_by(models.AnalysisResult.created_at.desc()).first()
        if not analysis_result:
            flash('No analysis results found for download.', 'error')
            return redirect(url_for('index'))

        # Only shortlisted whose emails were sent (and recorded with timestamp)
        selected_candidates = models.Candidate.query.filter_by(
            analysis_id=analysis_result.id, is_shortlisted=True, email_sent=True
        ).filter(models.Candidate.sent_on.isnot(None)).all()

        if not selected_candidates:
            flash('No shortlisted candidates have been sent an email yet. Please send emails first.', 'warning')
            return redirect(url_for('index'))

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        flowables.append(Paragraph(f"<b>Selected Candidates Report for {analysis_result.company_name}</b>", styles['Title']))
        flowables.append(Spacer(1, 12))
        flowables.append(Paragraph(f"<b>HR:</b> {analysis_result.hr_name} | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        flowables.append(Paragraph(f"<b>Total Selected:</b> {len(selected_candidates)}", styles['Normal']))
        flowables.append(Spacer(1, 24))

        data = [['S.No.', 'Name', 'Match Score', 'Experience', 'Education', 'AI Detected', 'Email Sent On']]
        for i, c in enumerate(selected_candidates, 1):
            sent_date = c.sent_on.strftime("%Y-%m-%d %H:%M") if c.sent_on else "N/A"
            data.append([str(i), Paragraph(c.name or "N/A", styles['Normal']),
                          f"{c.match_score}%", f"{c.experience_years} years",
                          c.education or "N/A", "Yes" if c.is_ai_generated else "No", sent_date])
        table = Table(data, colWidths=[0.5*inch, 1.5*inch, 0.75*inch, 1*inch, 1.75*inch, 0.75*inch, 1.25*inch])
        table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('ALIGN',(0,0),(-1,-1),'CENTER')]))
        flowables.append(table)

        doc.build(flowables)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{company_name}_selected_report.pdf", mimetype='application/pdf')
    except Exception as e:
        logging.exception(f"Error creating PDF: {e}")
        flash(f'Error creating PDF: {str(e)}', 'error')
        return redirect(url_for('index'))

# ==================== Error Handlers ====================
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ==================== Run App ====================
if __name__ == '__main__':
    with app.app_context():
        import models # ensure models are registered
        db.create_all()
        logging.info("Database tables ensured on startup.")
    # Ensure uploads dir exists at runtime
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
