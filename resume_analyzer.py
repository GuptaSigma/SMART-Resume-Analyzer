import logging
from dataclasses import dataclass, asdict, field
from io import BytesIO
from typing import List, Dict, Optional, Any, Tuple
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from docx import Document
except Exception:
    Document = None

# Logging setup
logger = logging.getLogger("resume_analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Data classes
@dataclass
class SkillHit:
    name: str
    category: str = "technical"
    match: bool = True
    is_hr_requirement: bool = False  # ✅ NEW: Flag for HR requirement highlighting

@dataclass
class GitHubProfile:
    url: str
    source: str

@dataclass
class CandidateReport:
    name: str
    email: Optional[str]
    phone: Optional[str]
    education: str
    university: str
    experience_years: int
    skills: List[SkillHit]
    has_internship: bool
    internship_company: str
    has_hackathon: bool
    has_certifications_or_achievements: bool
    certifications_and_achievements: List[str]
    github_profiles: List[GitHubProfile] = field(default_factory=list)
    linkedin_profile: Optional[str] = None
    target_role: str = "Software Developer"
    resume_filename: Optional[str] = None
    resume_text: str = ""
    is_ai_generated: bool = False
    match_score: int = 0
    match_reasons: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    is_fresher: bool = False
    projects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["skills"] = [asdict(s) for s in self.skills]
        d["github_profiles"] = [asdict(g) for g in self.github_profiles]
        return d

# Helper functions
def _escape_for_word_boundary(term: str) -> str:
    if term.lower() == "c++":
        return r"c\s*\+\s*\+"
    if term.lower() == "c#":
        return r"c\s*#"
    if term.lower() == "node.js":
        return r"node\.?js"
    return re.escape(term)

def _compile_word_regex(term: str) -> re.Pattern:
    pattern = _escape_for_word_boundary(term)
    return re.compile(fr"(?<![\w+]){pattern}(?![\w+])", re.IGNORECASE)

def _calculate_months_between(start_str: str, end_str: str) -> int:
    """Calculates the number of months between two date strings."""
    try:
        formats_to_try = ['%b %Y', '%Y', '%d%b%Y', '%B %Y']
        start_date = None
        for fmt in formats_to_try:
            try:
                start_date = datetime.strptime(start_str.strip(), fmt)
                break
            except ValueError:
                continue
        if start_date is None:
            return 0

        if 'present' in end_str.lower() or 'current' in end_str.lower():
            end_date = datetime.now()
        else:
            end_date = None
            for fmt in formats_to_try:
                try:
                    end_date = datetime.strptime(end_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            if end_date is None:
                return 0

        diff = relativedelta(end_date, start_date)
        return diff.years * 12 + diff.months + 1
    except Exception:
        return 0

# Main class
class ResumeAnalyzer:
    def __init__(self, *, pdf_max_pages: int = 10):
        self.pdf_max_pages = pdf_max_pages
        self.skills_database: Dict[str, List[str]] = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'typescript', 'c'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'bootstrap'],
            'database': ['mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'cassandra', 'sql'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'deep learning', 'machine learning', 'data science'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'trello']
        }

        self.role_skills: Dict[str, List[str]] = {
            'Data Analyst': ['python', 'sql', 'excel', 'tableau', 'powerbi', 'pandas', 'numpy', 'matplotlib', 'statistics', 'data visualization'],
            'Software Developer': ['programming', 'python', 'java', 'javascript', 'git', 'debugging', 'algorithms', 'data structures', 'testing', 'object-oriented programming'],
            'Web Developer': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node.js', 'php', 'wordpress', 'responsive design'],
            'Full Stack Developer': ['frontend', 'backend', 'database', 'api', 'html', 'css', 'javascript', 'python', 'c', 'java', 'c++', 'sql'],
            'UI/UX Designer': ['figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'wireframing', 'prototyping', 'user research', 'usability testing', 'personas'],
            'Graphic Designer': ['photoshop', 'illustrator', 'indesign', 'coreldraw', 'after effects', 'premiere pro', 'typography', 'branding', 'canva', 'creative suite'],
            'Cybersecurity Analyst': ['network security', 'penetration testing', 'vulnerability assessment', 'firewalls', 'encryption', 'intrusion detection', 'risk management', 'security audits', 'incident response', 'security policies'],
            'Cloud Engineer': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'cloud architecture', 'cloud migration', 'automation', 'monitoring'],
            'AI/ML Engineer': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'data science', 'natural language processing', 'model deployment', 'hyperparameter tuning', 'neural networks'],
            'DevOps Engineer': ['docker', 'kubernetes', 'jenkins', 'git', 'linux', 'terraform', 'ansible', 'ci/cd', 'monitoring', 'configuration management'],
            'Mobile Developer': ['react native', 'flutter', 'swift', 'kotlin', 'java', 'objective-c', 'ios', 'android', 'xamarin', 'app deployment'],
            'Product Manager': ['product strategy', 'roadmapping', 'agile', 'scrum', 'user research', 'analytics', 'a/b testing', 'stakeholder management', 'market research', 'feature prioritization']
        }
        
        self._email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        self._url_re = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
        self._github_domain_re = re.compile(r"https?://(?:www\.)?github\.com/[^\s)]+", re.IGNORECASE)
        self._github_username_line_re = re.compile(r"github\s*[:|-]?\s*@?([A-Za-z0-9-]{1,39})", re.IGNORECASE)
        self._linkedin_re = re.compile(
            r"(https?://)?(www\.)?linkedin\.com/(in|pub|company)/[a-zA-Z0-9._-]+/?",
            re.IGNORECASE
        )
        self._linkedin_username_line_re = re.compile(r"linkedin\s*[:|-]?\s*@?([a-zA-Z0-9-._]+)", re.IGNORECASE)
        self._phone_re = re.compile(r'(?:\+91[\s-]?)?\d{5}[\s-]?\d{5}|(?:\+91[\s-]?)?\d{10}')

        self._date_range_re = re.compile(
            r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4})'
            r'\s*(?:to|[-–—])?\s*'
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4}|Present|Current)\b',
            re.IGNORECASE
        )

        self._skill_regex_cache: Dict[str, re.Pattern] = {}
        all_skills = set(s for skills_list in self.skills_database.values() for s in skills_list)
        all_skills.update(s for skills_list in self.role_skills.values() for s in skills_list)
        for s in all_skills:
            if s.lower() not in self._skill_regex_cache:
                self._skill_regex_cache[s.lower()] = _compile_word_regex(s)

    def extract_text_from_file(self, file_obj) -> str:
        filename = getattr(file_obj, 'filename', None) or getattr(file_obj, 'name', 'uploaded')
        lower = str(filename).lower()
        try:
            if lower.endswith('.pdf'):
                return self._extract_from_pdf(file_obj)
            elif lower.endswith(('.docx', '.doc')):
                return self._extract_from_docx(file_obj)
            elif lower.endswith('.txt'):
                data = file_obj.read()
                return data.decode('utf-8', errors='ignore') if isinstance(data, (bytes, bytearray)) else str(data)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            logger.warning(f"Error extracting text from {filename}: {e}")
            return f"Error processing file {filename}. Please ensure file is not corrupted."

    def _extract_from_pdf(self, file_obj) -> str:
        pdf_bytes = file_obj.read()
        if fitz is not None:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                full_text = []
                max_pages = min(self.pdf_max_pages, doc.page_count)
                for page_num in range(max_pages):
                    page = doc.load_page(page_num)
                    full_text.append(page.get_text())
                    links = page.get_links()
                    for link in links:
                        if 'uri' in link:
                            full_text.append(f"\n{link['uri']}\n")
                doc.close()
                extracted_text = "\n".join(full_text)
                if extracted_text.strip():
                    return extracted_text
                logger.warning("No text extracted via PyMuPDF, trying PyPDF2 fallback.")
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}. Trying PyPDF2 fallback.")
        if PyPDF2 is not None:
            try:
                reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                full_text = []
                max_pages = min(self.pdf_max_pages, len(reader.pages))
                for i in range(max_pages):
                    page = reader.pages[i]
                    try:
                        full_text.append(page.extract_text() or "")
                    except Exception:
                        continue
                extracted_text = "\n".join(full_text)
                if extracted_text.strip():
                    return extracted_text
                else:
                    return "PDF file could not be processed - may be image-based or corrupted"
            except Exception as e:
                logger.warning(f"PyPDF2 processing failed: {e}")
                return "PDF file could not be processed due to format issues"
        return "PyMuPDF/PyPDF2 not installed. Cannot read PDF."

    def _extract_from_docx(self, file_obj) -> str:
        if Document is None:
            raise RuntimeError("python-docx not installed. Please install with `pip install python-docx`.")
        try:
            doc = Document(BytesIO(file_obj.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.exception(f"Error reading DOCX: {e}")
            raise

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        emails = self._email_re.findall(text)
        email = emails[0] if emails else None
        phone_match = self._phone_re.search(text)
        phone = phone_match.group(0).strip() if phone_match else None
        return {"email": email, "phone": phone}

    def extract_name(self, text: str) -> str:
        lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
        for line in lines[:5]:
            if 3 < len(line) < 60 and len(line.split()) <= 5 and re.match(r'^[A-Za-z\s.]+$', line):
                return line.title()
        return "Unknown"

    def extract_education(self, text: str) -> Tuple[str, str]:
        text_lower = text.lower()
        education_level = "Not specified"
        university = ""
        
        education_patterns = [
            r'\bph\.?d\b|\bdoctorate\b',
            r'\bmaster\s+of\s+engineering\b|\b(?:m\.?tech|mtech|m\.e\.?|me|m\.?sc|msc|m\.?com|mcom|m\.?a|ma|mba|mca)\b',
            r'\bbachelor\s+of\s+engineering\b|\b(?:b\.?tech|btech|b\.e\.?|be|bachelor of\s+sci[a-z]*e|b\.?sc|bsc|b\.?com|bcom|b\.?a|ba|bba|bca)\b',
            r'\bdiploma\b',
            r'\b(?:higher\s+secondary|12th|twelfth|intermediate)\b',
            r'\b(?:secondary|10th|tenth|matriculation)\b',
            r'\bengineering\b|\bengineer\b'
        ]
        
        education_section_match = re.search(r'education\s*.*?(?:\n\n|\Z)', text_lower, re.DOTALL | re.IGNORECASE)
        education_text = education_section_match.group(0) if education_section_match else text_lower

        for pattern in education_patterns:
            match = re.search(pattern, education_text, re.IGNORECASE)
            if match:
                education_level = match.group(0).strip().title()
                break
        
        university_patterns = [
            r'([^,\n]+?)(?:\s*(?:university|college|institute|polytechnic|iit|nit))',
        ]
        
        search_area = education_text
        if education_level != "Not specified":
            edu_start_index = text_lower.find(education_level.lower())
            if edu_start_index != -1:
                search_area = text_lower[max(0, edu_start_index - 50):edu_start_index + 100]

        for pattern in university_patterns:
            matches = re.findall(pattern, search_area, re.IGNORECASE)
            if matches:
                full_name = max(matches, key=len).strip()
                keywords_to_append = ['university', 'college', 'institute', 'polytechnic', 'iit', 'nit']
                for kw in keywords_to_append:
                    if kw in search_area:
                        full_name = full_name + ' ' + kw
                        break
                university = full_name.strip().title()
                break
        
        if not university and "education" in text_lower:
            lines = text_lower.split('\n')
            edu_section_found = False
            for line in lines:
                if 'education' in line:
                    edu_section_found = True
                    continue
                if edu_section_found:
                    if any(kw in line for kw in ['university', 'college', 'institute', 'school']):
                        clean_line = re.sub(r'\d{4}-\d{4}|present', '', line)
                        clean_line = re.sub(r'(b\.?tech|b\.?e|bachelor|m\.?tech|master|diploma|phd)', '', clean_line, re.IGNORECASE)
                        university = clean_line.strip().title()
                        if university:
                            break
                            
        university = university.strip('.,- ')
        return education_level, university

    def extract_experience_months(self, text: str) -> int:
        experience_text = ""
        section_pattern = re.compile(
            r'\b(EXPERIENCE|PROFESSIONAL EXPERIENCE|WORK EXPERIENCE)\b(.*?)(?=\b(?:PROJECTS|SKILLS|EDUCATION|LANGUAGES|ACTIVITIES)\b|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = section_pattern.search(text)
        if match:
            experience_text = match.group(2)
        else:
            experience_text = text
            
        total_months = 0
        matches = self._date_range_re.findall(experience_text)
        for start_date_str, end_date_str in matches:
            total_months += _calculate_months_between(start_date_str, end_date_str)
        return total_months

    def is_fresher(self, text: str) -> bool:
        fresher_keywords = ['fresher', 'entry level', 'seeking entry-level', 'first professional role']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in fresher_keywords)

    def extract_skills(self, text: str, target_skills: List[str] = None) -> List[SkillHit]:
        found_skills = []
        text_lower = text.lower()
        all_target_skills = set()
        if target_skills:
            all_target_skills.update([s.lower() for s in target_skills])
        else:
            all_target_skills.update([s.lower() for skills_list in self.skills_database.values() for s in skills_list])

        found_skill_names = set()
        for skill_lower in all_target_skills:
            if skill_lower in self._skill_regex_cache:
                regex = self._skill_regex_cache[skill_lower]
                if regex.search(text):
                    found_skill_names.add(skill_lower)
            elif skill_lower in text_lower:
                found_skill_names.add(skill_lower)
        
        for skill_name in found_skill_names:
            category = "technical"
            for cat, skills_list in self.skills_database.items():
                if skill_name in [s.lower() for s in skills_list]:
                    category = cat
                    break
            found_skills.append(SkillHit(name=skill_name.title(), category=category, match=True, is_hr_requirement=False))
        
        return found_skills
        
    def extract_github_profiles(self, text: str) -> List[GitHubProfile]:
        profiles = []
        github_urls = self._github_domain_re.findall(text)
        for url in github_urls:
            url = url.rstrip(')')
            profiles.append(GitHubProfile(url=url, source="explicit"))
        lines = text.split('\n')
        for line in lines:
            match = self._github_username_line_re.search(line)
            if match:
                username = match.group(1)
                url = f"https://github.com/{username}"
                if not any(p.url == url for p in profiles):
                    profiles.append(GitHubProfile(url=url, source="inferred"))
        return profiles

    def extract_linkedin_profile(self, text: str) -> Optional[str]:
        url_match = self._linkedin_re.search(text)
        if url_match:
            full_url = url_match.group(0)
            if not full_url.startswith(('http://', 'https://')):
                return 'https://' + full_url
            return full_url

        for line in text.split('\n'):
            username_match = self._linkedin_username_line_re.search(line)
            if username_match and 'linkedin.com' not in username_match.group(0):
                username = username_match.group(1)
                if '@' not in username:
                    return f"https://www.linkedin.com/in/{username}"
        return None

    def extract_certifications(self, text: str) -> List[str]:
        certifications = []
        
        cert_pattern = re.compile(
            r'^(.*(udemy|google|coursera|linkedin learning|aws|microsoft|hubspot|edx|udacity).*|'
            r'.*(certificate|certification|nanodegree|specialization|course)\b.*|'
            r'([\w\s,&]+?)\s*-\s*(udemy|google|coursera|linkedin learning|aws|microsoft|hubspot|edx|udacity))$',
            re.MULTILINE | re.IGNORECASE
        )

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            clean_line = re.sub(r'^[•\-\*]\s*', '', line)
            
            if cert_pattern.match(clean_line):
                certifications.append(clean_line)
                
        return certifications

    def has_internship(self, text: str) -> Tuple[bool, str]:
        internship_keywords = ['intern', 'internship', 'trainee', 'apprentice']
        text_lower = text.lower()
        for keyword in internship_keywords:
            if keyword in text_lower:
                lines = text_lower.split('\n')
                for line in lines:
                    if keyword in line:
                        words = line.split()
                        if 'at' in words:
                            idx = words.index('at')
                            if idx + 1 < len(words):
                                company = ' '.join(words[idx+1:idx+3])
                                return True, company.title()
                        return True, "Not specified"
        return False, ""

    def has_hackathon_experience(self, text: str) -> bool:
        hackathon_keywords = ['hackathon', 'hack-a-thon', 'coding competition', 'programming contest']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in hackathon_keywords)

    def extract_projects(self, text: str) -> List[str]:
        projects = []
        found_projects_normalized = set()

        section_pattern = re.compile(
            r'projects?\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Za-z\s]{2,30}$|\Z)',
            re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        
        match = section_pattern.search(text)
        if not match:
            return []

        project_text_block = match.group(1)
        lines = [line.strip() for line in project_text_block.split("\n") if line.strip()]

        description_starters = re.compile(
            r'^(a|an|the|includes?|designed|provides|built|created|developed|implemented|mimic|features|using|being|a fully)\b',
            re.IGNORECASE
        )
        date_pattern = re.compile(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|\d{4})\b', re.IGNORECASE)

        for line in lines:
            clean_line = re.sub(r'^[\s•\-\–\*]+', '', line).strip()
            if not clean_line:
                continue

            word_count = len(clean_line.split())
            is_description = description_starters.match(clean_line)
            is_just_date = date_pattern.search(clean_line) and word_count < 6

            if not is_description and not is_just_date:
                normalized = re.sub(r'[^a-z0-9 ]+', '', clean_line.lower()).strip()
                if normalized and normalized not in found_projects_normalized:
                    found_projects_normalized.add(normalized)
                    projects.append(clean_line.title())

        return projects
    
    def _predict_role_from_text(self, text: str) -> str:
        text_lower = text.lower()
        role_scores: Dict[str, int] = {role: 0 for role in self.role_skills.keys()}

        for role, skills in self.role_skills.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    role_scores[role] += 1
        
        if not any(role_scores.values()):
            return "Software Developer"
        predicted_role = max(role_scores, key=role_scores.get)
        return predicted_role

    def _count_roles(self, text: str) -> int:
        experience_text = ""
        section_pattern = re.compile(
            r'\b(EXPERIENCE|PROFESSIONAL EXPERIENCE|WORK EXPERIENCE)\b(.*?)(?=\b(?:PROJECTS|SKILLS|EDUCATION|LANGUAGES|ACTIVITIES)\b|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = section_pattern.search(text)
        if match:
            experience_text = match.group(2)
        else:
            experience_text = text

        matches = self._date_range_re.findall(experience_text)
        return len(matches)

    def calculate_match_score(self, candidate_skills: List[SkillHit], target_skills: List[str],
                              experience_years: int, education: str, has_internship: bool, has_hackathon: bool, is_fresher: bool, has_certifications_or_achievements: bool, num_roles: int) -> Tuple[int, List[str]]:
        score = 0
        reasons = []

        weights = {
            "skills": 50,
            "experience": 20,
            "education": 15,
            "internship": 5,
            "hackathon": 5,
            "certifications": 5,
            "multi_role_bonus": 0
        }

        if target_skills:
            matched_skills = [s for s in candidate_skills if s.name.lower() in [t.lower() for t in target_skills]]
            skill_percentage = (len(matched_skills) / len(target_skills)) if len(target_skills) > 0 else 0
            skill_score = round(skill_percentage * weights["skills"])
            score += skill_score
            if skill_score > (weights["skills"] * 0.7):
                reasons.append(f"Strong skill match ({len(matched_skills)}/{len(target_skills)} skills)")
            elif skill_score > (weights["skills"] * 0.4):
                reasons.append(f"Good skill match ({len(matched_skills)}/{len(target_skills)} skills)")

        matched_skill_count = len([s for s in candidate_skills if s.match])
        if matched_skill_count >= 3:
            score += 5
            reasons.append(f"Bonus: Matched {matched_skill_count} key skills")
        
        experience_score = min(experience_years, 5) * (weights["experience"] / 5)
        score += experience_score
        if experience_years >= 5:
            reasons.append(f"Highly experienced ({experience_years} years)")
        elif experience_years >= 1:
            reasons.append(f"Some professional experience ({experience_years} years)")
        elif is_fresher:
            reasons.append("Fresher candidate")

        education_score_map = {
            'bachelor': 1, 'engineer': 1, 'master': 1.25, 'phd': 1.5
        }
        education_level_match = next((kw for kw in education_score_map if kw in education.lower().replace(' ', '_')), None)
        if education_level_match:
            edu_score = education_score_map[education_level_match] * weights["education"]
            score += edu_score
            reasons.append(f"Relevant degree: {education}")

        if num_roles >= 5:
            score += 10
            reasons.append("Bonus: Experience across 5 or more roles")
        elif num_roles >= 3:
            score += 5
            reasons.append("Bonus: Experience across 3 or more roles")

        if has_internship:
            score += weights["internship"]
            reasons.append("Internship experience")

        if has_hackathon:
            score += weights["hackathon"]
            reasons.append("Hackathon experience")
            
        if has_certifications_or_achievements:
            score += weights["certifications"]
            reasons.append("Certifications and achievements")

        final_score = max(min(round(score), 100), 0)
        return final_score, reasons

    def _extract_skills_from_jd(self, jd_content: str) -> List[str]:
        jd_lower = jd_content.lower()
        jd_skills = []
        
        for skill, regex in self._skill_regex_cache.items():
            if regex.search(jd_lower):
                jd_skills.append(skill)
        
        skill_section_pattern = re.compile(
            r'(?:required skills?|must have|qualifications?|technical skills?|skills?|requirements?)[\s:]*([^\n]+(?:\n(?![\n])[^\n]+)*)',
            re.IGNORECASE
        )
        
        match = skill_section_pattern.search(jd_content)
        if match:
            skill_text = match.group(1)
            for skill, regex in self._skill_regex_cache.items():
                if regex.search(skill_text.lower()) and skill not in jd_skills:
                    jd_skills.append(skill)
        
        return list(set(jd_skills))

    def _check_hr_requirements(self, hr_requirements: str, candidate) -> List[str]:
        issues = []
        
        if not hr_requirements or not hr_requirements.strip():
            return issues
        
        reqs = [r.strip() for r in re.split(r'[,;\n]', hr_requirements) if r.strip()]
        
        for req in reqs:
            req_lower = req.lower()
            
            if "min" in req_lower and ("year" in req_lower or "yr" in req_lower or "experience" in req_lower):
                numbers = re.findall(r'\d+', req)
                if numbers:
                    min_years = int(numbers[0])
                    if candidate["experience_years"] < min_years:
                        issues.append(f"⚠️ Experience less than required ({min_years} years, has {candidate['experience_years']} years)")
            
            elif "max" in req_lower and ("year" in req_lower or "yr" in req_lower):
                numbers = re.findall(r'\d+', req)
                if numbers:
                    max_years = int(numbers[0])
                    if candidate["experience_years"] > max_years:
                        issues.append(f"⚠️ Experience exceeds maximum ({max_years} years, has {candidate['experience_years']} years)")
            
            elif any(edu in req_lower for edu in ["b.tech", "btech", "m.tech", "mtech", "bachelor", "master", "phd", "degree"]):
                education_lower = candidate.get("education", "").lower()
                if req_lower not in education_lower:
                    issues.append(f"⚠️ Education requirement not met: {req}")
            
            else:
                clean_req = re.sub(r'\b(mandatory|must have|required|should have)\b', '', req_lower).strip()
                
                if clean_req:
                    skill_found = any(clean_req in s["name"].lower() for s in candidate["skills"])
                    
                    if not skill_found:
                        issues.append(f"❌ Missing required skill: {req}")
        
        return issues

    def analyze_resume(self, file, target_role: str = "", jd_content: str = "", hr_requirements: str = "") -> Dict[str, Any]:
        try:
            filename = getattr(file, 'filename', 'unknown')
            text = self.extract_text_from_file(file)
            if not text or len(text.strip()) < 50:
                raise ValueError("Insufficient text extracted from resume")

            if target_role and target_role.lower() == "other":
                if jd_content:
                    final_target_role = "Custom Role (from JD)"
                    target_skills = self._extract_skills_from_jd(jd_content)
                else:
                    final_target_role = self._predict_role_from_text(text)
                    target_skills = self.role_skills.get(final_target_role, [])
            elif target_role:
                final_target_role = target_role
                if jd_content:
                    target_skills = self._extract_skills_from_jd(jd_content)
                else:
                    target_skills = self.role_skills.get(final_target_role, [])
            else:
                final_target_role = self._predict_role_from_text(text)
                if jd_content:
                    target_skills = self._extract_skills_from_jd(jd_content)
                else:
                    target_skills = self.role_skills.get(final_target_role, [])

            # ✅ NEW: Track HR requirement skills separately
            hr_requirement_skills = []
            if hr_requirements and hr_requirements.strip():
                hr_reqs = [r.strip() for r in re.split(r'[,;\n]', hr_requirements) if r.strip()]
                for req in hr_reqs:
                    req_lower = req.lower()
                    if "year" not in req_lower and "yr" not in req_lower and "experience" not in req_lower:
                        if not any(edu in req_lower for edu in ["b.tech", "btech", "m.tech", "mtech", "bachelor", "master", "phd", "degree"]):
                            clean_skill = re.sub(r'\b(mandatory|must have|required|should have)\b', '', req_lower).strip()
                            if clean_skill:
                                hr_requirement_skills.append(clean_skill)
                                if clean_skill not in [s.lower() for s in target_skills]:
                                    target_skills.append(clean_skill)

            contact_info = self.extract_contact_info(text)
            name = self.extract_name(text)
            education, university = self.extract_education(text)
            experience_months = self.extract_experience_months(text)
            experience_years = experience_months // 12

            is_fresher_status = self.is_fresher(text)
            if is_fresher_status and experience_months > 0:
                is_fresher_status = False
            
            num_roles = self._count_roles(text)
            skills = self.extract_skills(text, target_skills)
            
            # ✅ NEW: Mark HR requirement skills for highlighting
            if hr_requirement_skills:
                hr_requirement_skills_normalized = [s.replace(" ", "") for s in hr_requirement_skills]
                for skill in skills:
                    skill_normalized = skill.name.lower().replace(" ", "")
                    if skill_normalized in hr_requirement_skills_normalized:
                        skill.is_hr_requirement = True
            
            github_profiles = self.extract_github_profiles(text)
            linkedin_profile = self.extract_linkedin_profile(text)
            certifications = self.extract_certifications(text)
            has_intern, intern_company = self.has_internship(text)
            has_hackathon = self.has_hackathon_experience(text)
            has_certifications_or_achievements = len(certifications) > 0
            projects = self.extract_projects(text)

            experience_str = "No Experience"
            if experience_months > 0:
                if experience_months < 12:
                    experience_str = f"{experience_months} months"
                else:
                    years = experience_months // 12
                    months = experience_months % 12
                    if months > 0:
                        experience_str = f"{years} years and {months} months"
                    else:
                        experience_str = f"{years} years"

            match_score, match_reasons = self.calculate_match_score(
                skills, target_skills, experience_years, education,
                has_intern, has_hackathon, is_fresher_status,
                has_certifications_or_achievements, num_roles
            )

            matched_skill_names = {s.name.lower() for s in skills if s.match}
            missing_skills = [skill for skill in target_skills if skill.lower() not in matched_skill_names]

            candidate_data = {
                'name': name,
                'email': contact_info.get('email'),
                'phone': contact_info.get('phone'),
                'education': education,
                'university': university,
                'experience_years': experience_years,
                'skills': [asdict(s) for s in skills],
            }

            hr_flags = self._check_hr_requirements(hr_requirements, candidate_data) if hr_requirements else []

            result = {
                **candidate_data,
                'experience': experience_str,
                'target_role': final_target_role,
                'match_score': match_score,
                'match_reasons': match_reasons,
                'missing_skills': missing_skills,
                'missing_jd_skills': missing_skills,
                'hr_flags': hr_flags,
                'has_internship': has_intern,
                'internship_company': intern_company,
                'has_hackathon': has_hackathon,
                'has_certifications_or_achievements': has_certifications_or_achievements,
                'certifications_and_achievements': certifications,
                'github_profiles': [asdict(g) for g in github_profiles],
                'linkedin_profile': linkedin_profile,
                'resume_filename': filename,
                'resume_text': text[:1000],
                'is_fresher': is_fresher_status,
                'projects': projects,
                'num_roles': num_roles
            }
            return result

        except Exception as e:
            logger.error(f"Error analyzing resume {getattr(file, 'filename', 'unknown')}: {e}")
            raise
