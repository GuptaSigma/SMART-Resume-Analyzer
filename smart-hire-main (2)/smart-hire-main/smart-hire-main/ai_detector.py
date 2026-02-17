import math
import re
import logging
import ssl
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Set
from collections import Counter

import numpy as np
from scipy import stats

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ngrams
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

import textstat

# --- Optional Dependency: LanguageTool ---
# This requires Java. We wrap it in try-except to prevent crashes if not installed.
try:
    import language_tool_python
    # Attempt to initialize; this might fail if Java is missing
    _GRAMMAR_TOOL = language_tool_python.LanguageTool('en-US')
except Exception as e:
    logging.warning(f"LanguageTool could not be initialized (Java might be missing). Grammar checking disabled. Error: {e}")
    _GRAMMAR_TOOL = None

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- NLTK Resource Management ---
def _ensure_nltk():
    """
    Robustly ensure NLTK resources are available.
    Handles SSL certificate errors which are common in NLTK downloads.
    """
    # Workaround for SSL certificate verify failed errors
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"), # Added for newer NLTK versions
        ("vader_lexicon", "sentiment/vader_lexicon"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ]

    print("Checking NLTK resources...")
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading NLTK resource: {name}")
                nltk.download(name, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource '{name}': {e}")

_ensure_nltk()

# --- Constants ---

AI_COMMON_PHRASES = [
    "as an ai", "as a language model", "as a model", "i'm an ai", "i am an ai",
    "cutting-edge", "synergy", "leverage", "paradigm shift", "state-of-the-art",
    "robust framework", "seamless integration", "holistic approach",
    "innovative solution", "scalable architecture", "streamlined process",
    "leveraging advanced", "in today's fast-paced", "results-driven",
    "impactful", "domain expertise", "dynamic environment",
    "deliverables", "stakeholders", "best-in-class", "industry-leading",
    "forward-thinking", "thought leader", "strategic initiatives",
    "cross-functional", "value-add", "circle back", "touch base",
    "move the needle", "low-hanging fruit", "win-win", "deep dive",
    "game-changer", "disruptive", "next-generation", "best practices",
    "core competencies", "value proposition", "synergistic", "mission-critical",
    "actionable insights", "key takeaways", "ecosystem", "end-to-end",
    "transformative", "world-class", "bleeding edge", "groundbreaking",
    "proven track record", "extensive experience", "diverse background",
    "wide range", "in-depth knowledge", "comprehensive understanding",
]

AI_PATTERNS = [
    r"\b(furthermore|moreover|additionally|consequently|nevertheless)\b",
    r"\b(utilize|implement|facilitate|optimize|enhance|leverage|streamline)\b",
    r"\b(various|numerous|several|multiple)\s+\w+\s+(including|such as)\b",
    r"\b(it is important to note|it should be noted|notably)\b",
    r"\b(in conclusion|to summarize|in summary|overall)\b",
    r"\b(can be|will be|has been)\s+(achieved|implemented|utilized)\b",
    r"\b(in order to|with regard to|in terms of)\b",
    r"\b(extensive|comprehensive|diverse|wide range of)\b",
    r"\b(proven|demonstrated|established|recognized)\b",
]

AI_TRANSITION_WORDS = {
    'furthermore', 'moreover', 'additionally', 'consequently', 'nevertheless',
    'nonetheless', 'therefore', 'thus', 'hence', 'accordingly', 'subsequently',
}

HUMAN_CASUAL_WORDS = {
    'really', 'very', 'pretty', 'quite', 'yeah', 'stuff', 'things', 'got',
    'gonna', 'wanna', 'lots', 'tons', 'super', 'awesome', 'cool', 'great',
}

# --- Helper Functions ---

def _fallback_sent_tokenize(text: str) -> List[str]:
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r"[.!?]+\s+", text) if s.strip()]

def _fallback_word_tokenize(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\b\w+\b", text.lower())

def _get_stopwords() -> set:
    try:
        return set(stopwords.words("english"))
    except Exception:
        return {"the", "is", "in", "at", "of", "on", "and", "a", "to", "for", "it", "with", "as", "that", "this", "by", "an"}

def lcs_length(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def avg_lcs_similarity(sentences: List[str]) -> float:
    if len(sentences) < 2:
        return 0.0
    sims = []
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i].strip(), sentences[i + 1].strip()
        if s1 and s2:
            lcs = lcs_length(s1, s2)
            denom = max(len(s1), len(s2))
            if denom > 0:
                sims.append(lcs / denom)
    return float(sum(sims) / len(sims)) if sims else 0.0

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return float(entropy)

def word_entropy(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if len(tokens) < 2:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return float(entropy)

def pseudo_perplexity_unigram(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    V = len(set(tokens))
    counts = Counter(tokens)
    N = len(tokens)
    log_sum = sum(math.log((counts[t] + 1) / (N + V) + 1e-12) for t in tokens)
    ppl = math.exp(-log_sum / max(N, 1))
    return float(min(max(ppl, 0.0), 1000.0))

def pseudo_perplexity_bigram(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if len(tokens) < 3:
        return 100.0
    bigram_list = list(ngrams(tokens, 2))
    if not bigram_list:
        return 100.0
    bigram_counts = Counter(bigram_list)
    unigram_counts = Counter(tokens)
    V = len(unigram_counts)
    log_prob = sum(
        math.log((bigram_counts[bg] + 1) / (unigram_counts[bg[0]] + V) + 1e-12)
        for bg in bigram_list
    )
    perplexity = math.exp(-log_prob / len(bigram_list))
    return float(min(max(perplexity, 0.0), 1000.0))

def count_ai_phrases(text: str) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(phrase) for phrase in AI_COMMON_PHRASES)

def sentiment_neutrality_score(text: str) -> float:
    try:
        sia = SentimentIntensityAnalyzer()
        comp = sia.polarity_scores(text).get("compound", 0.0)
        return float(max(0.0, min(1.0, 1.0 - abs(comp))))
    except Exception:
        return 0.5

def burstiness_score(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    lengths = [len(_fallback_word_tokenize(s)) for s in sentences if s.strip()]
    if not lengths:
        return 0.0
    mean = np.mean(lengths)
    std = np.std(lengths)
    return float(std / mean) if mean > 0 else 0.0

def bullet_usage_score(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bullet_regex = re.compile(r"^(\-|\*|•|\u2022|\d+[\)\.]\s+)")
    bullet_count = sum(1 for ln in lines if bullet_regex.match(ln))
    return float(bullet_count / len(lines))

def stopword_ratio(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    sw = _get_stopwords()
    return float(sum(1 for t in tokens if t.lower() in sw) / len(tokens))

def grammar_error_count(text: str) -> int:
    try:
        if _GRAMMAR_TOOL is None:
            return 0
        snippet = text[:8000]
        matches = _GRAMMAR_TOOL.check(snippet)
        return len(matches)
    except Exception:
        return 0

def ai_pattern_count(text: str) -> int:
    return sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in AI_PATTERNS)

def transition_word_overuse(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t.lower() in AI_TRANSITION_WORDS)
    return float(count / len(tokens) * 100)

def casual_word_usage(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t.lower() in HUMAN_CASUAL_WORDS)
    return float(count / len(tokens) * 100)

def lexical_diversity(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if len(tokens) < 10:
        return 0.5
    return float(len(set(tokens)) / len(tokens))

def avg_word_length(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    return float(sum(len(t) for t in tokens) / len(tokens))

def passive_voice_ratio(text: str) -> float:
    try:
        words = word_tokenize(text)
        tags = pos_tag(words)
        passive_count = 0
        be_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        for i in range(len(tags) - 1):
            if tags[i][0].lower() in be_verbs:
                if tags[i + 1][1] in ['VBN', 'VBD']:
                    passive_count += 1
        sentences = _fallback_sent_tokenize(text)
        return float(passive_count / max(len(sentences), 1))
    except Exception:
        return 0.0

def sentence_start_diversity(sentences: List[str]) -> float:
    if len(sentences) < 3:
        return 0.5
    starts = [sent.strip().split()[0].lower() for sent in sentences if sent.strip().split()]
    if not starts:
        return 0.5
    return float(len(set(starts)) / len(starts))

def punctuation_variety(text: str) -> float:
    punct_marks = set('.,!?;:-"\'()[]{}')
    found = set(ch for ch in text if ch in punct_marks)
    return float(len(found) / len(punct_marks))

def contraction_usage(text: str) -> float:
    contractions = [
        "don't", "can't", "won't", "shouldn't", "wouldn't", "couldn't",
        "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
        "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    ]
    text_lower = text.lower()
    count = sum(text_lower.count(c) for c in contractions)
    words = len(_fallback_word_tokenize(text))
    return float(count / max(words, 1) * 100)

def personal_pronoun_ratio(text: str) -> float:
    tokens = _fallback_word_tokenize(text)
    if not tokens:
        return 0.0
    pronouns = {'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your'}
    count = sum(1 for t in tokens if t.lower() in pronouns)
    return float(count / len(tokens) * 100)

def sentence_similarity_variance(sentences: List[str]) -> float:
    if len(sentences) < 3:
        return 0.0
    similarities = []
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i].strip(), sentences[i + 1].strip()
        if s1 and s2:
            lcs = lcs_length(s1, s2)
            denom = max(len(s1), len(s2))
            if denom > 0:
                similarities.append(lcs / denom)
    return float(np.var(similarities)) if len(similarities) >= 2 else 0.0

# --- Feature Vector Class ---

@dataclass
class FeatureVector:
    readability: float
    ai_phrases: int
    sentiment_neutrality: float
    burstiness: float
    perplexity: float
    grammar_errors: int
    entropy_score: float
    bullet_score: float
    stopword_ratio: float
    lcs_similarity: float
    ai_patterns: int
    sent_similarity_var: float
    lexical_diversity: float
    avg_word_len: float
    passive_voice: float
    sent_start_diversity: float
    punct_variety: float
    contraction_rate: float
    word_entropy: float
    perplexity_bigram: float
    transition_overuse: float
    casual_words: float
    personal_pronouns: float

    def to_ordered_list(self) -> List[float]:
        return [
            self.readability,
            float(self.ai_phrases),
            self.sentiment_neutrality,
            self.burstiness,
            self.perplexity,
            float(self.grammar_errors),
            self.entropy_score,
            self.bullet_score,
            self.stopword_ratio,
            self.lcs_similarity,
            float(self.ai_patterns),
            self.sent_similarity_var,
            self.lexical_diversity,
            self.avg_word_len,
            self.passive_voice,
            self.sent_start_diversity,
            self.punct_variety,
            self.contraction_rate,
            self.word_entropy,
            self.perplexity_bigram,
            self.transition_overuse,
            self.casual_words,
            self.personal_pronouns,
        ]

# --- Main Detector Class ---

class AIDetector:
    def __init__(self, use_random_forest: bool = True) -> None:
        self._scaler = RobustScaler()
        self._use_rf = use_random_forest
        
        # Configure the model architecture
        if use_random_forest:
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            gb = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=12,
                min_samples_split=2,
                subsample=0.9,
                random_state=42
            )
            lr = LogisticRegression(
                max_iter=5000,
                C=0.1,
                random_state=42
            )
            self._model = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft',
                weights=[3, 3, 1]
            )
        else:
            self._model = GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.08,
                max_depth=15,
                random_state=42
            )
            
        self._feature_dim = 23
        self._train_synthetic_model()

    def _train_synthetic_model(self) -> None:
        """
        Trains a synthetic model based on statistical distributions typically
        observed in AI vs Human text. 
        """
        rng = np.random.default_rng(42)
        n = 1000
        
        # Features: [Readability, AI Phrases, Sentiment, Burstiness, Perplexity, Grammar, Entropy...]
        # Distributions approximated from analysis of generated vs human text
        human = np.column_stack([
            rng.normal(75, 20, n),     # Readability
            rng.poisson(0.1, n),       # AI Phrases
            rng.normal(0.40, 0.25, n), # Neutrality
            rng.normal(0.75, 0.30, n), # Burstiness
            rng.normal(250, 100, n),   # Perplexity
            rng.poisson(6.0, n),       # Grammar Errors
            rng.normal(4.6, 0.7, n),   # Entropy
            rng.beta(2.0, 5.0, n),     # Bullets
            rng.normal(0.38, 0.14, n), # Stopwords
            rng.normal(0.04, 0.06, n), # LCS Similarity
            rng.poisson(0.5, n),       # AI Patterns
            rng.normal(0.15, 0.08, n), # Sent Sim Var
            rng.normal(0.78, 0.12, n), # Lexical Div
            rng.normal(4.5, 0.8, n),   # Avg Word Len
            rng.normal(0.05, 0.04, n), # Passive Voice
            rng.normal(0.90, 0.08, n), # Sent Start Div
            rng.normal(0.72, 0.18, n), # Punct Variety
            rng.normal(5.0, 2.2, n),   # Contractions
            rng.normal(7.0, 1.5, n),   # Word Entropy
            rng.normal(250, 100, n),   # Bigram Ppl
            rng.normal(0.3, 0.3, n),   # Transition Overuse
            rng.normal(5.5, 2.5, n),   # Casual Words
            rng.normal(6.5, 3.0, n),   # Personal Pronouns
        ])
        
        ai = np.column_stack([
            rng.normal(35, 6, n),      # Readability
            rng.poisson(6.0, n),       # AI Phrases
            rng.normal(0.92, 0.04, n), # Neutrality
            rng.normal(0.12, 0.06, n), # Burstiness
            rng.normal(60, 18, n),     # Perplexity
            rng.poisson(0.1, n),       # Grammar Errors
            rng.normal(3.0, 0.25, n),  # Entropy
            rng.beta(4.0, 2.0, n),     # Bullets
            rng.normal(0.60, 0.05, n), # Stopwords
            rng.normal(0.45, 0.15, n), # LCS Similarity
            rng.poisson(9.0, n),       # AI Patterns
            rng.normal(0.02, 0.02, n), # Sent Sim Var
            rng.normal(0.50, 0.07, n), # Lexical Div
            rng.normal(6.2, 0.5, n),   # Avg Word Len
            rng.normal(0.25, 0.08, n), # Passive Voice
            rng.normal(0.55, 0.10, n), # Sent Start Div
            rng.normal(0.35, 0.12, n), # Punct Variety
            rng.normal(0.3, 0.3, n),   # Contractions
            rng.normal(4.5, 0.6, n),   # Word Entropy
            rng.normal(55, 15, n),     # Bigram Ppl
            rng.normal(4.5, 1.5, n),   # Transition Overuse
            rng.normal(0.2, 0.2, n),   # Casual Words
            rng.normal(0.8, 0.6, n),   # Personal Pronouns
        ])
        
        # Combine and Shuffle
        X = np.vstack([human, ai]).astype(float)
        y = np.array([0] * n + [1] * n) # 0 = Human, 1 = AI
        
        # Clipping for stability
        X[:, 0] = np.clip(X[:, 0], 0, 100)
        X[:, 2] = np.clip(X[:, 2], 0, 1)
        X[:, 3] = np.clip(X[:, 3], 0, 3)
        X[:, 4] = np.clip(X[:, 4], 0, 1000)
        X[:, 6] = np.clip(X[:, 6], 0, 8)
        X[:, 7] = np.clip(X[:, 7], 0, 1)
        X[:, 8] = np.clip(X[:, 8], 0, 1)
        X[:, 9] = np.clip(X[:, 9], 0, 1)
        X[:, 11] = np.clip(X[:, 11], 0, 1)
        X[:, 12] = np.clip(X[:, 12], 0, 1)
        X[:, 13] = np.clip(X[:, 13], 0, 15)
        X[:, 14] = np.clip(X[:, 14], 0, 1)
        X[:, 15] = np.clip(X[:, 15], 0, 1)
        X[:, 16] = np.clip(X[:, 16], 0, 1)
        X[:, 17] = np.clip(X[:, 17], 0, 20)
        X[:, 18] = np.clip(X[:, 18], 0, 12)
        X[:, 19] = np.clip(X[:, 19], 0, 1000)
        X[:, 20] = np.clip(X[:, 20], 0, 20)
        X[:, 21] = np.clip(X[:, 21], 0, 20)
        X[:, 22] = np.clip(X[:, 22], 0, 30)
        
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]
        
        self._scaler.fit(X)
        Xs = self._scaler.transform(X)
        self._model.fit(Xs, y)
        
        train_acc = self._model.score(Xs, y)
        logger.info(f"AIDetector trained (ULTRA SENSITIVE). Accuracy: {train_acc:.4f}")

    def _extract_features(self, text: str) -> FeatureVector:
        text = text or ""
        sentences = _fallback_sent_tokenize(text)
        
        return FeatureVector(
            readability=float(max(0.0, min(100.0, textstat.flesch_reading_ease(text) if text else 50.0))),
            ai_phrases=count_ai_phrases(text),
            sentiment_neutrality=sentiment_neutrality_score(text),
            burstiness=burstiness_score(sentences),
            perplexity=pseudo_perplexity_unigram(text),
            grammar_errors=grammar_error_count(text),
            entropy_score=shannon_entropy(text),
            bullet_score=bullet_usage_score(text),
            stopword_ratio=stopword_ratio(text),
            lcs_similarity=avg_lcs_similarity(sentences),
            ai_patterns=ai_pattern_count(text),
            sent_similarity_var=sentence_similarity_variance(sentences),
            lexical_diversity=lexical_diversity(text),
            avg_word_len=avg_word_length(text),
            passive_voice=passive_voice_ratio(text),
            sent_start_diversity=sentence_start_diversity(sentences),
            punct_variety=punctuation_variety(text),
            contraction_rate=contraction_usage(text),
            word_entropy=word_entropy(text),
            perplexity_bigram=pseudo_perplexity_bigram(text),
            transition_overuse=transition_word_overuse(text),
            casual_words=casual_word_usage(text),
            personal_pronouns=personal_pronoun_ratio(text),
        )

    def analyze_resume(self, text: str, candidate_name: str, resume_filename: str) -> Dict[str, Any]:
        feats = self._extract_features(text)
        X = np.array([feats.to_ordered_list()], dtype=float)
        Xs = self._scaler.transform(X)
        proba_ai = float(self._model.predict_proba(Xs)[0, 1])

        # Apply a curve to make the model more sensitive/aggressive
        if proba_ai < 0.5:
            boosted_prob = proba_ai ** 1.5
        else:
            boosted_prob = 1 - (1 - proba_ai) ** 1.5

        # Map probability to percentage with non-linear scaling
        if boosted_prob < 0.15:
            ai_percentage_base = 30.0 + (boosted_prob / 0.15) * 10.0
        elif boosted_prob < 0.35:
            ai_percentage_base = 40.0 + ((boosted_prob - 0.15) / 0.20) * 15.0
        elif boosted_prob < 0.50:
            ai_percentage_base = 55.0 + ((boosted_prob - 0.35) / 0.15) * 10.0
        elif boosted_prob < 0.65:
            ai_percentage_base = 65.0 + ((boosted_prob - 0.50) / 0.15) * 13.0
        elif boosted_prob < 0.85:
            ai_percentage_base = 78.0 + ((boosted_prob - 0.65) / 0.20) * 14.0
        else:
            ai_percentage_base = 92.0 + ((boosted_prob - 0.85) / 0.15) * 8.0

        ai_percentage_base = float(np.clip(ai_percentage_base, 30.0, 100.0))

        # Heuristic Rule-Based Boosting
        red_flags = []
        ai_score_boost = 0.0

        if feats.ai_phrases >= 5:
            red_flags.append(f"CRITICAL: Excessive AI buzzwords ({feats.ai_phrases})")
            ai_score_boost += 15.0
        elif feats.ai_phrases >= 3:
            red_flags.append(f"High AI phrase count ({feats.ai_phrases})")
            ai_score_boost += 8.0
        elif feats.ai_phrases >= 1:
            red_flags.append(f"AI phrases detected ({feats.ai_phrases})")
            ai_score_boost += 4.0

        if feats.ai_patterns >= 8:
            red_flags.append(f"CRITICAL: Heavy AI writing patterns ({feats.ai_patterns})")
            ai_score_boost += 12.0
        elif feats.ai_patterns >= 5:
            red_flags.append(f"AI writing patterns ({feats.ai_patterns})")
            ai_score_boost += 6.0

        if feats.sentiment_neutrality > 0.90:
            red_flags.append("CRITICAL: Extremely neutral sentiment")
            ai_score_boost += 10.0
        elif feats.sentiment_neutrality > 0.85:
            red_flags.append("Very neutral sentiment")
            ai_score_boost += 5.0

        if feats.burstiness < 0.15:
            red_flags.append("CRITICAL: Uniform sentence structure")
            ai_score_boost += 12.0
        elif feats.burstiness < 0.25:
            red_flags.append("Low sentence variation")
            ai_score_boost += 6.0

        if feats.lcs_similarity > 0.40:
            red_flags.append("CRITICAL: Highly repetitive patterns")
            ai_score_boost += 10.0
        elif feats.lcs_similarity > 0.30:
            red_flags.append("Repetitive sentence patterns")
            ai_score_boost += 5.0

        if feats.lexical_diversity < 0.55:
            red_flags.append("CRITICAL: Very low vocabulary diversity")
            ai_score_boost += 10.0
        elif feats.lexical_diversity < 0.60:
            red_flags.append("Low vocabulary diversity")
            ai_score_boost += 5.0

        if feats.passive_voice > 0.20:
            red_flags.append("CRITICAL: Excessive passive voice")
            ai_score_boost += 8.0
        elif feats.passive_voice > 0.15:
            red_flags.append("High passive voice usage")
            ai_score_boost += 4.0

        if feats.contraction_rate < 0.5:
            red_flags.append("Almost no contractions (overly formal)")
            ai_score_boost += 6.0
        elif feats.contraction_rate < 1.0:
            red_flags.append("Very few contractions")
            ai_score_boost += 3.0

        if feats.transition_overuse > 4.0:
            red_flags.append("CRITICAL: Overuse of formal transitions")
            ai_score_boost += 8.0
        elif feats.transition_overuse > 2.5:
            red_flags.append("Frequent formal transitions")
            ai_score_boost += 4.0

        if feats.casual_words < 0.3:
            red_flags.append("CRITICAL: No casual language")
            ai_score_boost += 8.0
        elif feats.casual_words < 0.5:
            red_flags.append("Lack of casual language")
            ai_score_boost += 4.0

        if feats.personal_pronouns < 1.0:
            red_flags.append("Lack of personal pronouns")
            ai_score_boost += 5.0

        # Calculate final percentage
        ai_percentage = min(100.0, ai_percentage_base + ai_score_boost)
        ai_percentage = round(ai_percentage, 1)
        human_percentage = round(100.0 - ai_percentage, 1)

        # AI Confidence is model certainty (how sure it is of its prediction)
        ai_confidence = round(max(proba_ai, 1 - proba_ai) * 100, 1)
        ai_score_0_to_10 = round(ai_percentage / 10.0, 2)

        # Debug Output
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG - {candidate_name}")
        print(f"{'='*60}")
        print(f"Raw Probability (AI): {proba_ai:.4f}")
        print(f"Raw Probability (Human): {1-proba_ai:.4f}")
        print(f"Model Confidence: {max(proba_ai, 1-proba_ai):.4f}")
        print(f"AI Percentage (with red flags): {ai_percentage}%")
        print(f"AI Confidence (model certainty): {ai_confidence}%")
        print(f"Different? {ai_percentage != ai_confidence}")
        print(f"{'='*60}\n")

        return {
            "candidate_name": candidate_name or "Unknown",
            "resume_filename": resume_filename or "unknown",
            "ai_percentage": ai_percentage,
            "human_percentage": human_percentage,
            "ai_confidence": ai_confidence,
            "ai_score_0_to_10": ai_score_0_to_10,
            "raw_probability": round(proba_ai, 4),
            "boosted_probability": round(boosted_prob, 4),
            "red_flag_boost": round(ai_score_boost, 1),
            "classification": self._get_classification(ai_percentage),
            "confidence_level": self._get_confidence_level(ai_confidence),
            "features": {
                "readability": round(feats.readability, 2),
                "ai_phrases": feats.ai_phrases,
                "sentiment_neutrality": round(feats.sentiment_neutrality, 3),
                "burstiness": round(feats.burstiness, 3),
                "perplexity": round(feats.perplexity, 2),
                "grammar_errors": feats.grammar_errors,
                "entropy_score": round(feats.entropy_score, 3),
                "bullet_score": round(feats.bullet_score, 3),
                "stopword_ratio": round(feats.stopword_ratio, 3),
                "lcs_similarity": round(feats.lcs_similarity, 3),
                "ai_patterns": feats.ai_patterns,
                "lexical_diversity": round(feats.lexical_diversity, 3),
                "passive_voice": round(feats.passive_voice, 3),
                "contraction_rate": round(feats.contraction_rate, 2),
                "transition_overuse": round(feats.transition_overuse, 2),
                "casual_words": round(feats.casual_words, 2),
                "personal_pronouns": round(feats.personal_pronouns, 2),
            },
            "is_ai_generated": ai_percentage >= 65.0,
            "red_flags": red_flags,
        }

    def _get_classification(self, ai_percentage: float) -> str:
        if ai_percentage >= 92:
            return "Almost certainly AI-generated"
        elif ai_percentage >= 78:
            return "Highly likely AI-generated"
        elif ai_percentage >= 65:
            return "Likely AI-generated"
        elif ai_percentage >= 55:
            return "Possibly AI-assisted"
        elif ai_percentage >= 40:
            return "Likely human-written"
        else:
            return "Very likely human-written"

    def _get_confidence_level(self, ai_confidence: float) -> str:
        if ai_confidence >= 90 or ai_confidence <= 10:
            return "Very High Confidence"
        elif ai_confidence >= 75 or ai_confidence <= 25:
            return "High Confidence"
        elif ai_confidence >= 60 or ai_confidence <= 40:
            return "Moderate Confidence"
        else:
            return "Low Confidence"
