from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, make_response
import os
from extensions import db
from models import AnalysisResult, Candidate
from resume_analyzer import ResumeAnalyzer
from ai_detector import AIDetector
from email_service import EmailService
from datetime import datetime
import logging
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
if app.config['SECRET_KEY'] == 'dev-secret-key-change-in-production':
    logger.warning("WARNING: Using default SECRET_KEY. Set SECRET_KEY environment variable in production!")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///resume_analyzer.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)

# Initialize services
resume_analyzer = ResumeAnalyzer()
ai_detector = AIDetector()
email_service = EmailService()

# Create tables
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/analyze_resumes', methods=['POST'])
def analyze_resumes():
    """Process and analyze uploaded resumes"""
    try:
        # Get form data
        company_name = request.form.get('company_name')
        hr_name = request.form.get('hr_name')
        hr_email = request.form.get('hr_email')
        hr_password = request.form.get('email_password')
        target_role = request.form.get('target_role', '')
        hr_requirements = request.form.get('hr_requirements', '')
        
        # Validate required fields
        if not all([company_name, hr_name, hr_email, hr_password]):
            flash('Please fill in all required fields', 'danger')
            return redirect(url_for('index'))
        
        # Get uploaded files
        resume_files = request.files.getlist('resume_files')
        if not resume_files or resume_files[0].filename == '':
            flash('Please upload at least one resume', 'warning')
            return redirect(url_for('index'))
        
        # Process job description file if uploaded
        jd_content = ''
        jd_file = request.files.get('jd_file')
        if jd_file and jd_file.filename:
            try:
                jd_content = resume_analyzer.extract_text_from_file(jd_file)
                logger.info(f"Job description file processed: {jd_file.filename}")
            except Exception as e:
                logger.error(f"Error processing JD file: {e}")
                jd_content = ''
        
        # Setup email service
        email_service.setup(hr_email, hr_password)
        
        # Create analysis result
        analysis_result = AnalysisResult(
            company_name=company_name,
            hr_name=hr_name,
            hr_email=hr_email,
            total_processed=0,
            shortlisted_count=0
        )
        db.session.add(analysis_result)
        db.session.commit()
        
        candidates_data = []
        shortlisted_candidates = []
        not_selected_candidates = []
        
        # Process each resume
        for resume_file in resume_files:
            if resume_file.filename == '':
                continue
                
            try:
                # Analyze resume
                result = resume_analyzer.analyze_resume(
                    resume_file,
                    target_role=target_role,
                    jd_content=jd_content,
                    hr_requirements=hr_requirements
                )
                
                # Extract resume text for AI detection
                resume_file.seek(0)  # Reset file pointer
                resume_text = resume_analyzer.extract_text_from_file(resume_file)
                
                # Run AI detection
                ai_result = ai_detector.analyze_resume(
                    resume_text,
                    result['name'],
                    result['resume_filename']
                )
                
                # Determine if shortlisted (>=75% match score)
                is_shortlisted = result['match_score'] >= 75
                
                # Create candidate record
                candidate = Candidate(
                    analysis_id=analysis_result.id,
                    name=result['name'],
                    email=result.get('email'),
                    phone=result.get('phone'),
                    match_score=result['match_score'],
                    target_role=result['target_role'],
                    experience_years=result['experience_years'],
                    education=result['education'],
                    university=result.get('university', ''),
                    is_shortlisted=is_shortlisted,
                    is_ai_generated=ai_result['is_ai_generated'],
                    resume_filename=result['resume_filename'],
                    ai_percentage=ai_result['ai_percentage'],
                    ai_confidence=ai_result['ai_confidence'],
                    ai_features=ai_result.get('features', {})
                )
                db.session.add(candidate)
                db.session.flush()  # Flush to get candidate.id before commit
                
                # Prepare candidate data for template
                candidate_info = {
                    'id': candidate.id,
                    'name': result['name'],
                    'email': result.get('email', 'N/A'),
                    'phone': result.get('phone', 'N/A'),
                    'match_score': result['match_score'],
                    'target_role': result['target_role'],
                    'experience': result.get('experience', 'N/A'),
                    'experience_years': result['experience_years'],
                    'education': result['education'],
                    'university': result.get('university', 'N/A'),
                    'skills': result.get('skills', []),
                    'missing_skills': result.get('missing_skills', []),
                    'hr_flags': result.get('hr_flags', []),
                    'match_reasons': result.get('match_reasons', []),
                    'is_shortlisted': is_shortlisted,
                    'resume_filename': result['resume_filename'],
                    'ai_percentage': ai_result['ai_percentage'],
                    'ai_confidence': ai_result['ai_confidence'],
                    'is_ai_generated': ai_result['is_ai_generated'],
                    'classification': ai_result['classification'],
                    'github_profiles': result.get('github_profiles', []),
                    'linkedin_profile': result.get('linkedin_profile'),
                    'has_internship': result.get('has_internship', False),
                    'internship_company': result.get('internship_company', ''),
                    'has_hackathon': result.get('has_hackathon', False),
                    'certifications_and_achievements': result.get('certifications_and_achievements', []),
                    'projects': result.get('projects', [])
                }
                
                candidates_data.append(candidate_info)
                
                if is_shortlisted:
                    shortlisted_candidates.append(candidate_info)
                else:
                    not_selected_candidates.append(candidate_info)
                
                analysis_result.total_processed += 1
                if is_shortlisted:
                    analysis_result.shortlisted_count += 1
                
                logger.info(f"Processed resume: {result['name']} - Score: {result['match_score']}")
                
            except Exception as e:
                logger.error(f"Error processing resume {resume_file.filename}: {e}")
                continue
        
        db.session.commit()
        
        # Sort candidates by match score
        shortlisted_candidates.sort(key=lambda x: x['match_score'], reverse=True)
        not_selected_candidates.sort(key=lambda x: x['match_score'], reverse=True)
        
        logger.info(f"Analysis complete: {analysis_result.total_processed} resumes processed, {analysis_result.shortlisted_count} shortlisted")
        
        return render_template('results.html',
                             company_name=company_name,
                             hr_name=hr_name,
                             total_processed=analysis_result.total_processed,
                             shortlisted_count=analysis_result.shortlisted_count,
                             shortlisted_candidates=shortlisted_candidates,
                             not_selected_candidates=not_selected_candidates,
                             target_role=target_role if target_role else 'Auto-detected')
        
    except Exception as e:
        logger.error(f"Error in analyze_resumes: {e}")
        flash(f'An error occurred during analysis: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/send_email', methods=['POST'])
def send_email():
    """Send a single email to a candidate"""
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        email_type = data.get('type')
        
        if not all([email, name, email_type]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        success = email_service.send_email(email, name, email_type)
        
        if success:
            # Update candidate record - use the most recent one if multiple exist
            candidate = Candidate.query.filter_by(email=email, name=name).order_by(Candidate.created_at.desc()).first()
            if candidate:
                candidate.email_sent = True
                candidate.sent_on = datetime.utcnow()
                db.session.commit()
            
            return jsonify({'success': True, 'message': f'Email sent to {name}'})
        else:
            return jsonify({'success': False, 'message': 'Failed to send email'}), 500
            
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/send_all_rejections', methods=['POST'])
def send_all_rejections():
    """Send rejection emails to all provided candidates"""
    try:
        data = request.get_json()
        emails = data.get('emails', [])
        
        if not emails:
            return jsonify({'success': False, 'message': 'No emails provided'}), 400
        
        sent_count = 0
        for email_data in emails:
            email = email_data.get('email')
            name = email_data.get('name')
            
            if email and name:
                success = email_service.send_email(email, name, 'rejection')
                if success:
                    sent_count += 1
                    # Update candidate record - use the most recent one if multiple exist
                    candidate = Candidate.query.filter_by(email=email, name=name).order_by(Candidate.created_at.desc()).first()
                    if candidate:
                        candidate.email_sent = True
                        candidate.sent_on = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Sent {sent_count} rejection emails',
            'sent_count': sent_count
        })
        
    except Exception as e:
        logger.error(f"Error sending rejection emails: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/send_all_congratulations', methods=['POST'])
def send_all_congratulations():
    """Send congratulatory emails to all provided candidates"""
    try:
        data = request.get_json()
        emails = data.get('emails', [])
        
        if not emails:
            return jsonify({'success': False, 'message': 'No emails provided'}), 400
        
        sent_count = 0
        for email_data in emails:
            email = email_data.get('email')
            name = email_data.get('name')
            
            if email and name:
                success = email_service.send_email(email, name, 'congratulations')
                if success:
                    sent_count += 1
                    # Update candidate record - use the most recent one if multiple exist
                    candidate = Candidate.query.filter_by(email=email, name=name).order_by(Candidate.created_at.desc()).first()
                    if candidate:
                        candidate.email_sent = True
                        candidate.sent_on = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Sent {sent_count} congratulatory emails',
            'sent_count': sent_count
        })
        
    except Exception as e:
        logger.error(f"Error sending congratulatory emails: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/ai_check/<filename>')
def ai_check(filename):
    """Display AI detection analysis for a specific candidate"""
    try:
        # Find candidate by resume filename
        candidate = Candidate.query.filter_by(resume_filename=filename).first()
        
        if not candidate:
            flash('Candidate not found', 'warning')
            return redirect(url_for('index'))
        
        return render_template('ai_check.html',
                             candidate=candidate,
                             ai_percentage=candidate.ai_percentage,
                             ai_confidence=candidate.ai_confidence,
                             ai_features=candidate.ai_features or {},
                             is_ai_generated=candidate.is_ai_generated)
        
    except Exception as e:
        logger.error(f"Error in ai_check: {e}")
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/download_report/<company_name>')
def download_report(company_name):
    """Generate and download a PDF report for the analysis results"""
    try:
        # Get the latest analysis result for this company
        analysis_result = AnalysisResult.query.filter(AnalysisResult.company_name.ilike(company_name)).order_by(AnalysisResult.created_at.desc()).first()
        
        if not analysis_result:
            flash('No analysis found for this company', 'warning')
            return redirect(url_for('index'))
        
        # Get all candidates for this analysis
        candidates = Candidate.query.filter_by(analysis_id=analysis_result.id).all()
        shortlisted = [c for c in candidates if c.is_shortlisted]
        not_selected = [c for c in candidates if not c.is_shortlisted]
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5282'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Add title
        elements.append(Paragraph(f"Resume Analysis Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Company and HR details
        company_data = [
            ['Company Name:', analysis_result.company_name],
            ['HR Name:', analysis_result.hr_name],
            ['HR Email:', analysis_result.hr_email],
            ['Analysis Date:', analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        company_table = Table(company_data, colWidths=[2*inch, 4*inch])
        company_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(company_table)
        elements.append(Spacer(1, 20))
        
        # Statistics
        elements.append(Paragraph("Summary Statistics", heading_style))
        stats_data = [
            ['Total Resumes Processed:', str(analysis_result.total_processed)],
            ['Shortlisted Candidates:', str(analysis_result.shortlisted_count)],
            ['Not Selected:', str(analysis_result.total_processed - analysis_result.shortlisted_count)],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 20))
        
        # Shortlisted Candidates
        if shortlisted:
            elements.append(Paragraph("Shortlisted Candidates", heading_style))
            
            # Header row
            shortlisted_data = [['Name', 'Email', 'Score', 'Education', 'Experience']]
            
            for candidate in sorted(shortlisted, key=lambda x: x.match_score, reverse=True):
                shortlisted_data.append([
                    candidate.name or 'N/A',
                    candidate.email or 'N/A',
                    f"{candidate.match_score:.1f}%",
                    candidate.education or 'N/A',
                    f"{candidate.experience_years} yrs" if candidate.experience_years else 'N/A'
                ])
            
            shortlisted_table = Table(shortlisted_data, colWidths=[1.5*inch, 1.8*inch, 0.8*inch, 1.3*inch, 1*inch])
            shortlisted_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
            ]))
            elements.append(shortlisted_table)
            elements.append(Spacer(1, 20))
        
        # Not Selected Candidates
        if not_selected:
            elements.append(Paragraph("Not Selected Candidates", heading_style))
            
            # Header row
            not_selected_data = [['Name', 'Email', 'Score', 'Education']]
            
            for candidate in sorted(not_selected, key=lambda x: x.match_score, reverse=True):
                not_selected_data.append([
                    candidate.name or 'N/A',
                    candidate.email or 'N/A',
                    f"{candidate.match_score:.1f}%",
                    candidate.education or 'N/A'
                ])
            
            not_selected_table = Table(not_selected_data, colWidths=[2*inch, 2.2*inch, 0.8*inch, 1.5*inch])
            not_selected_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b0000')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffe4e1')]),
            ]))
            elements.append(not_selected_table)
            elements.append(Spacer(1, 20))
        
        # AI Detection Summary
        elements.append(Paragraph("AI Detection Summary", heading_style))
        ai_generated_count = sum(1 for c in candidates if c.is_ai_generated)
        ai_summary_data = [
            ['Total Candidates Analyzed:', str(len(candidates))],
            ['AI-Generated Resumes Detected:', str(ai_generated_count)],
            ['Human-Written Resumes:', str(len(candidates) - ai_generated_count)],
        ]
        
        ai_summary_table = Table(ai_summary_data, colWidths=[3*inch, 3*inch])
        ai_summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff3cd')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(ai_summary_table)
        
        # Build PDF
        doc.build(elements)
        
        # Prepare response
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=resume_analysis_{company_name}_{analysis_result.created_at.strftime("%Y%m%d")}.pdf'
        
        logger.info(f"Generated PDF report for company: {company_name}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        flash(f'An error occurred while generating the report: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/download_all_selected/<company_name>')
def download_all_selected(company_name):
    """Download CSV file with all shortlisted candidates"""
    try:
        # Get the latest analysis result for this company
        analysis_result = AnalysisResult.query.filter(AnalysisResult.company_name.ilike(company_name)).order_by(AnalysisResult.created_at.desc()).first()
        
        if not analysis_result:
            flash('No analysis found for this company', 'warning')
            return redirect(url_for('index'))
        
        # Get all shortlisted candidates
        shortlisted_candidates = Candidate.query.filter_by(
            analysis_id=analysis_result.id,
            is_shortlisted=True
        ).order_by(Candidate.match_score.desc()).all()
        
        if not shortlisted_candidates:
            flash('No shortlisted candidates found', 'warning')
            return redirect(url_for('index'))
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Name', 'Email', 'Phone', 'Match Score (%)', 
            'Target Role', 'Experience (Years)', 'Education', 
            'University', 'Resume Filename', 'AI Generated'
        ])
        
        # Write candidate data
        for candidate in shortlisted_candidates:
            writer.writerow([
                candidate.name or 'N/A',
                candidate.email or 'N/A',
                candidate.phone or 'N/A',
                f"{candidate.match_score:.2f}",
                candidate.target_role or 'N/A',
                candidate.experience_years if candidate.experience_years is not None else 'N/A',
                candidate.education or 'N/A',
                candidate.university or 'N/A',
                candidate.resume_filename or 'N/A',
                'Yes' if candidate.is_ai_generated else 'No'
            ])
        
        # Prepare response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=shortlisted_candidates_{company_name}_{analysis_result.created_at.strftime("%Y%m%d")}.csv'
        
        logger.info(f"Generated CSV for shortlisted candidates: {company_name}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        flash(f'An error occurred while generating the CSV: {str(e)}', 'danger')
        return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Debug mode should be controlled via environment variable
    # Never enable debug=True in production as it poses security risks
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(debug=debug_mode)
