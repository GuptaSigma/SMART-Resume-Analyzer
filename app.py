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
        
        ai_generated_count = len([c for c in all_candidates if c.is_ai_generated])
        human_written_count = len(all_candidates) - ai_generated_count

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Define styles
        title_style = styles['Title'].clone('CustomTitle')
        title_style.textColor = colors.HexColor('#003366')
        title_style.fontSize = 20
        title_style.alignment = 1 # Center

        h2_style = styles['h2'].clone('CustomH2')
        h2_style.textColor = colors.HexColor('#003366')
        
        normal_style = styles['Normal']
        
        flowables = []

        # Title
        flowables.append(Paragraph("Resume Analysis Report", title_style))
        flowables.append(Spacer(1, 12))

        # Company Info Table
        info_data = [
            ["Company Name:", analysis_result.company_name],
            ["HR Name:", analysis_result.hr_name],
            ["HR Email:", analysis_result.hr_email],
            ["Analysis Date:", analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        flowables.append(info_table)
        flowables.append(Spacer(1, 24))

        # Summary Statistics
        flowables.append(Paragraph("Summary Statistics", h2_style))
        flowables.append(Spacer(1, 6))
        stats_data = [
            ["Total Resumes Processed:", str(analysis_result.total_processed)],
            ["Shortlisted Candidates:", str(len(shortlisted))],
            ["Not Selected:", str(len(not_selected))]
        ]
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#E6F2FF')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        flowables.append(stats_table)
        flowables.append(Spacer(1, 24))

        # Candidate Table Style
        candidate_table_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8B0000')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ('PADDING', (0,0), (-1,-1), 6)
        ])

        # Shortlisted candidates
        flowables.append(Paragraph("Shortlisted Candidates", h2_style))
        flowables.append(Spacer(1, 6))
        if shortlisted:
            candidates_data = [['Name', 'Email', 'Score', 'Education']]
            for c in shortlisted:
                candidates_data.append([
                    Paragraph(c.name or "N/A", normal_style),
                    Paragraph(c.email or "N/A", normal_style),
                    f"{c.match_score}%",
                    Paragraph(c.education or "N/A", normal_style)
                ])
            cand_table = Table(candidates_data, colWidths=[2*inch, 2*inch, 1*inch, 2*inch])
            cand_table.setStyle(candidate_table_style)
            flowables.append(cand_table)
        else:
            flowables.append(Paragraph("<i>No candidates shortlisted.</i>", styles['Italic']))
        flowables.append(Spacer(1, 24))

        # Not Selected candidates
        flowables.append(Paragraph("Not Selected Candidates", h2_style))
        flowables.append(Spacer(1, 6))
        if not_selected:
            not_selected_data = [['Name', 'Email', 'Score', 'Education']]
            for c in not_selected:
                not_selected_data.append([
                    Paragraph(c.name or "N/A", normal_style),
                    Paragraph(c.email or "N/A", normal_style),
                    f"{c.match_score}%",
                    Paragraph(c.education or "N/A", normal_style)
                ])
            ns_table = Table(not_selected_data, colWidths=[2*inch, 2*inch, 1*inch, 2*inch])
            ns_table.setStyle(candidate_table_style)
            flowables.append(ns_table)
        else:
            flowables.append(Paragraph("<i>All candidates were shortlisted!</i>", styles['Italic']))
        flowables.append(Spacer(1, 24))

        # AI Detection Summary
        flowables.append(Paragraph("AI Detection Summary", h2_style))
        flowables.append(Spacer(1, 6))
        ai_data = [
            ["Total Candidates Analyzed:", str(len(all_candidates))],
            ["AI-Generated Resumes Detected:", str(ai_generated_count)],
            ["Human-Written Resumes:", str(human_written_count)]
        ]
        ai_table = Table(ai_data, colWidths=[3*inch, 3*inch])
        ai_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFF2CC')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        flowables.append(ai_table)

        doc.build(flowables)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"{company_name}_analysis_report.pdf", mimetype='application/pdf')
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

        # Only shortlisted whose emails were sent
        selected_candidates = models.Candidate.query.filter_by(
            analysis_id=analysis_result.id, is_shortlisted=True, email_sent=True
        ).filter(models.Candidate.sent_on.isnot(None)).all()

        if not selected_candidates:
            flash('No shortlisted candidates have been sent an email yet.', 'warning')
            return redirect(url_for('index'))

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Define styles
        title_style = styles['Title'].clone('CustomTitle')
        title_style.textColor = colors.HexColor('#003366')
        title_style.fontSize = 20
        title_style.alignment = 1

        h2_style = styles['h2'].clone('CustomH2')
        h2_style.textColor = colors.HexColor('#003366')
        
        normal_style = styles['Normal']
        
        flowables = []

        # Title
        flowables.append(Paragraph("Selected Candidates Report", title_style))
        flowables.append(Spacer(1, 12))

        # Company Info Table
        info_data = [
            ["Company Name:", analysis_result.company_name],
            ["HR Name:", analysis_result.hr_name],
            ["Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        flowables.append(info_table)
        flowables.append(Spacer(1, 24))

        # Selected Table
        flowables.append(Paragraph("Selected Candidates (Email Sent)", h2_style))
        flowables.append(Spacer(1, 6))
        
        data = [['Name', 'Email', 'Score', 'Education', 'Sent On']]
        for c in selected_candidates:
            sent_date = c.sent_on.strftime("%Y-%m-%d %H:%M") if c.sent_on else "N/A"
            data.append([
                Paragraph(c.name or "N/A", normal_style),
                Paragraph(c.email or "N/A", normal_style),
                f"{c.match_score}%",
                Paragraph(c.education or "N/A", normal_style),
                sent_date
            ])
            
        table = Table(data, colWidths=[1.5*inch, 1.5*inch, 0.75*inch, 1.75*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8B0000')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
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
