from datetime import datetime
from extensions import db
from sqlalchemy.dialects.postgresql import JSON  # Or use a generic JSON type for other databases
from sqlalchemy.types import TypeDecorator, String
import json

class JSONEncodedDict(TypeDecorator):
    impl = String

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_result'
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    hr_name = db.Column(db.String(200), nullable=False)
    hr_email = db.Column(db.String(200), nullable=False)
    total_processed = db.Column(db.Integer, default=0)
    shortlisted_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    candidates = db.relationship('Candidate', backref='analysis_result', lazy=True)
    def __repr__(self):
        return f'<AnalysisResult {self.company_name}>'

class Candidate(db.Model):
    __tablename__ = 'candidate'
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_result.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    match_score = db.Column(db.Float, nullable=False)
    target_role = db.Column(db.String(200))
    experience_years = db.Column(db.Integer, default=0)
    education = db.Column(db.String(200))
    university = db.Column(db.String(200))
    is_shortlisted = db.Column(db.Boolean, default=False)
    is_ai_generated = db.Column(db.Boolean, default=False)
    resume_filename = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    email_sent = db.Column(db.Boolean, default=False)
    sent_on = db.Column(db.DateTime)
    # --- ADDED AI-SPECIFIC COLUMNS HERE ---
    ai_percentage = db.Column(db.Float)
    ai_confidence = db.Column(db.Float)
    ai_features = db.Column(JSONEncodedDict)
    def __repr__(self):
        return f'<Candidate {self.name}>'
