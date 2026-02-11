import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailService:
    def __init__(self):
        self.smtp_server = None
        self.hr_email = None
        self.hr_password = None
    
    def setup(self, hr_email, hr_password):
        """Setup email service with HR credentials"""
        self.hr_email = hr_email
        self.hr_password = hr_password
    
    def send_email(self, to_email, candidate_name, email_type):
        """Send a single email to a candidate"""
        try:
            if not all([self.hr_email, self.hr_password, to_email]):
                logging.error("Missing email configuration")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.hr_email
            msg['To'] = to_email
            msg['Subject'] = self._get_subject(email_type, candidate_name)
            
            # Get email body
            body = self._get_email_body(email_type, candidate_name)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email using Gmail SMTP
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.hr_email, self.hr_password)
                server.send_message(msg)
            
            logging.info(f"Email sent successfully to {to_email} for type: {email_type}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending email to {to_email}: {e}")
            return False

    def send_all_rejections(self, candidates):
        """Send rejection emails to a list of candidates"""
        success_count = 0
        total_count = len(candidates)
        for candidate in candidates:
            if self.send_email(candidate['email'], candidate['name'], 'rejection'):
                success_count += 1
        return {"success": True, "sent_count": success_count, "total_count": total_count}

    def send_all_congratulations(self, candidates):
        """Send congratulatory emails to a list of shortlisted candidates"""
        success_count = 0
        total_count = len(candidates)
        for candidate in candidates:
            if self.send_email(candidate['email'], candidate['name'], 'congratulations'):
                success_count += 1
        return {"success": True, "sent_count": success_count, "total_count": total_count}

    def _get_subject(self, email_type, candidate_name):
        """Get email subject based on type"""
        if email_type == 'congratulations':
            return f"Congratulations {candidate_name}! - Resume Shortlisted"
        elif email_type == 'rejection':
            return f"Thank you for your interest - Application Update"
        else:
            return "Application Update"
    
    def _get_email_body(self, email_type, candidate_name):
        """Get email body based on type"""
        if email_type == 'congratulations':
            return f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #28a745;">Congratulations, {candidate_name}!</h2>
                    
                    <p>We are pleased to inform you that your resume has been shortlisted for our current opening.</p>
                    
                    <p>Our AI-powered resume analysis system has identified you as a strong candidate based on:</p>
                    <ul>
                        <li>Relevant skills and experience</li>
                        <li>Educational background</li>
                        <li>Overall profile match</li>
                    </ul>
                    
                    <p>Our HR team will contact you soon for the next steps in the recruitment process.</p>
                    
                    <p>Thank you for your interest in joining our team!</p>
                    
                    <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                        <p style="margin: 0;"><strong>Contact:</strong> </p>
                        <p style="margin: 0;"><em>This email was sent by our AI Resume Analyzer system.</em></p>
                    </div>
                </div>
            </body>
            </html>
            """
        elif email_type == 'rejection':
            return f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #dc3545;">Thank you for your application, {candidate_name}</h2>
                    
                    <p>Thank you for taking the time to submit your resume for our current opening.</p>
                    
                    <p>After careful consideration using our AI-powered analysis system, we have decided to move forward with other candidates whose profiles more closely match our current requirements.</p>
                    
                    <p>We encourage you to:</p>
                    <ul>
                        <li>Keep enhancing your skills</li>
                        <li>Apply for future openings that match your profile</li>
                        <li>Stay connected with us for upcoming opportunities</li>
                    </ul>
                    
                    <p>We wish you the best in your job search and future endeavors.</p>
                    
                    <div style="margin-top: 30px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                        <p style="margin: 0;"><strong>Contact:</strong> </p>
                        <p style="margin: 0;"><em>This email was sent by our AI Resume Analyzer system.</em></p>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            return f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2>Application Update</h2>
                    <p>Dear {candidate_name},</p>
                    <p>Thank you for your application. We have received your resume and will review it shortly.</p>
                    <p>Best regards,<br>HR Team</p>
                </div>
            </body>
            </html>
            """
