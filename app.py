from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
from extensions import db
from models import AnalysisResult, Candidate
from resume_analyzer import ResumeAnalyzer
from ai_detector import AIDetector
from email_service import EmailService
from datetime import datetime
import logging

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
