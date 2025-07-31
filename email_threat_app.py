import logging
import streamlit as st
import joblib
import pandas as pd
import re
import plotly.express as px
import json
from datetime import datetime
import tldextract
from urllib.parse import urlparse
import string
from textstat import flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import sqlite3
import email
from email import policy

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def init_db():
    conn = sqlite3.connect('analysis_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            email_snippet TEXT,
            url_count INTEGER,
            prediction TEXT,
            confidence REAL,
            threat_score INTEGER,
            risk_level TEXT,
            file_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_analysis_result(email_snippet, url_count, prediction, confidence, threat_score, risk_level, file_name=None):
    conn = sqlite3.connect('analysis_results.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analysis_results (timestamp, email_snippet, url_count, prediction, confidence, threat_score, risk_level, file_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        email_snippet,
        url_count,
        prediction,
        confidence,
        threat_score,
        risk_level,
        file_name
    ))
    conn.commit()
    conn.close()

def get_analysis_history():
    conn = sqlite3.connect('analysis_results.db')
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, email_snippet, url_count, prediction, confidence, threat_score, risk_level, file_name FROM analysis_results ORDER BY timestamp DESC LIMIT 10')
    columns = ['timestamp', 'email_snippet', 'url_count', 'prediction', 'confidence', 'threat_score', 'risk_level', 'file_name']
    history = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return history

class FeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        # Modified: Replaced banking keywords with health-related keywords
        self.health_keywords = ['patient', 'medical', 'healthcare', 'hospital', 'clinic', 'appointment', 'prescription', 'record', 'diagnosis', 'treatment', 'billing', 'health']
        self.phishing_keywords = ['urgent', 'verify', 'suspended', 'click', 'update', 'confirm', 'expire', 'limited', 'act', 'immediate', 'security', 'alert', 'unauthorized', 'blocked', 'compromised', 'suspicious']
        self.spam_keywords = ['free', 'win', 'winner', 'prize', 'congratulations', 'offer', 'discount', 'limited', 'time', 'call', 'now', 'cash', 'money']

    def extract_url_features(self, text):
        features = {}
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        features['url_count'] = len(urls)
        features['has_url'] = 1 if urls else 0
        
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        shortened_domains = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly']
        features['suspicious_tld'] = 0
        features['shortened_url'] = 0
        features['ip_address'] = 0
        
        for url in urls:
            extracted = tldextract.extract(url)
            if f'.{extracted.suffix}' in suspicious_tlds:
                features['suspicious_tld'] = 1
            if any(domain in url for domain in shortened_domains):
                features['shortened_url'] = 1
            if re.match(r'https?://\d+\.\d+\.\d+\.\d+', url):
                features['ip_address'] = 1
        return features

    def extract_text_features(self, text):
        features = {}
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        text_lower = text.lower()
        features['health_keywords'] = sum(1 for keyword in self.health_keywords if keyword in text_lower)
        features['phishing_keywords'] = sum(1 for keyword in self.phishing_keywords if keyword in text_lower)
        features['spam_keywords'] = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        
        try:
            features['readability_score'] = flesch_kincaid_grade(text)
        except:
            features['readability_score'] = 0
            
        urgency_words = ['urgent', 'immediate', 'asap', 'now', 'quickly', 'hurry']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        features['currency_symbols'] = len(re.findall(r'[$€£¥₹]', text))
        features['numbers'] = len(re.findall(r'\d+', text))
        return features

    def extract_email_header_features(self, email_text):
        features = {}
        features['suspicious_sender'] = 1 if '@' in email_text and any(domain in email_text for domain in ['gmail.com', 'yahoo.com', 'hotmail.com']) else 0
        features['multiple_recipients'] = 1 if email_text.count('@') > 2 else 0
        features['reply_to_different'] = 0
        return features

    def extract_all_features(self, text):
        features = {}
        features.update(self.extract_url_features(text))
        features.update(self.extract_text_features(text))
        features.update(self.extract_email_header_features(text))
        return features

class MalwareDetector:
    def __init__(self):
        self.suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.cpl', '.dll', '.jar', '.js', '.jse', '.vbs', '.vbe', '.ws', '.wsf', '.wsh', '.ps1', '.psm1', '.psd1', '.zip', '.rar', '.7z', '.tar.gz']
        self.malicious_domains = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'fake-hospital.com', 'phishing-clinic.com', 'bogus-healthcare.com']
        self.malware_signatures = [r'eval\s*\(', r'document\.write\s*\(', r'<script[^>]*>.*?</script>', r'javascript:', r'vbscript:', r'activex', r'shell\.application', r'wscript\.shell']

    def analyze_attachments(self, email_text):
        warnings = []
        attachment_patterns = [
            r'attachment:\s*([^\s]+)', 
            r'filename[=:]\s*([^\s;]+)', 
            r'content-disposition:.*filename[=:]\s*([^\s;]+)'
        ]
        for pattern in attachment_patterns:
            matches = re.findall(pattern, email_text, re.IGNORECASE)
            for match in matches:
                filename = match.strip('\"\'')
                if any(filename.lower().endswith(ext) for ext in self.suspicious_extensions):
                    warnings.append(f"Suspicious attachment detected: {filename}")
        return warnings

    def analyze_urls(self, email_text):
        warnings = []
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if any(mal_domain in domain for mal_domain in self.malicious_domains):
                warnings.append(f"Known malicious domain detected: {domain}")
            if re.match(r'https?://\d+\.\d+\.\d+\.\d+', url):
                warnings.append(f"IP address URL detected: {url}")
            if any(shortener in domain for shortener in ['bit.ly', 'tinyurl', 't.co', 'goo.gl']):
                warnings.append(f"URL shortener detected: {domain}")
            if any(pattern in url.lower() for pattern in ['login', 'verify', 'update', 'secure']):
                warnings.append(f"Suspicious URL path detected: {url}")
        return warnings

    def analyze_content(self, email_text):
        warnings = []
        for signature in self.malware_signatures:
            if re.search(signature, email_text, re.IGNORECASE):
                warnings.append(f"Malware signature detected: {signature}")
        return warnings

    def calculate_threat_score(self, email_text):
        score = 0
        attachment_warnings = self.analyze_attachments(email_text)
        score += len(attachment_warnings) * 3
        
        url_warnings = self.analyze_urls(email_text)
        score += len(url_warnings) * 2
        
        content_warnings = self.analyze_content(email_text)
        score += len(content_warnings) * 4
        
        urgency_words = ['urgent', 'immediate', 'expires', 'suspended', 'limited time']
        urgency_count = sum(1 for word in urgency_words if word.lower() in email_text.lower())
        score += urgency_count
        
        return min(score, 10)

    def analyze_email(self, email_text):
        warnings = []
        warnings.extend(self.analyze_attachments(email_text))
        warnings.extend(self.analyze_urls(email_text))
        warnings.extend(self.analyze_content(email_text))
        
        threat_score = self.calculate_threat_score(email_text)
        risk_level = 'High' if threat_score >= 7 else 'Medium' if threat_score >= 4 else 'Low'
        
        return {
            'warnings': warnings, 
            'threat_score': threat_score, 
            'risk_level': risk_level
        }

class AlertSystem:
    def __init__(self):
        self.alert_config = {
            'email_alerts': False, 
            'webhook_alerts': False, 
            'log_alerts': True, 
            'real_time_alerts': True
        }
        self.severity_levels = {
            'phishing': 'HIGH', 
            'spam': 'MEDIUM', 
            'malware': 'CRITICAL', 
            'legitimate': 'LOW'
        }
        logging.basicConfig(
            filename='security_alerts.log', 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_threat(self, threat_type, email_content, confidence, additional_info=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'threat_type': threat_type,
            'confidence': confidence,
            'severity': self.severity_levels.get(threat_type, 'UNKNOWN'),
            'email_length': len(email_content),
            'additional_info': additional_info or {}
        }
        self.logger.info(f"Threat logged: {json.dumps(log_entry)}")

    def generate_real_time_alert(self, threat_type, confidence, email_preview):
        alert = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': threat_type,
            'confidence': confidence,
            'severity': self.severity_levels.get(threat_type, 'UNKNOWN'),
            'preview': email_preview[:50] + '...' if len(email_preview) > 50 else email_preview,
            'action_required': threat_type in ['phishing', 'malware']
        }
        return alert

    def process_threat_alert(self, threat_type, email_content, confidence, alert_config=None):
        config = alert_config or self.alert_config
        if config.get('log_alerts', True):
            self.log_threat(threat_type, email_content, confidence)
        if config.get('real_time_alerts', True):
            return self.generate_real_time_alert(threat_type, confidence, email_content)
        return None

def get_cybersecurity_tips(prediction):
    tips = {
        'phishing': "Beware of emails requesting patient information or containing suspicious links. Verify the sender's hospital or clinic domain before clicking.",
        'spam': "Avoid clicking on unsolicited medical offers or promotions. Use spam filters and report suspicious emails to your IT team.",
        'malware': "Do not open attachments or click links from unverified sources claiming to be from healthcare providers. Use antivirus software and report suspicious emails.",
        'legitimate': "Ensure emails are from trusted healthcare sources. Check for official hospital domains (e.g., @kathmanduhealth.org) and verify contact details."
    }
    return tips.get(prediction, "Always verify email authenticity before taking action.")

try:
    model = joblib.load('email_threat_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please train the models first by running email_threat_detection.py")
    st.stop()

malware_detector = MalwareDetector()
alert_system = AlertSystem()

st.set_page_config(page_title='Email Threat Detection System', layout='wide')
st.title('AI-Based Email Threat Detection for Health Sector')
st.markdown('Developed for healthcare institutions in Kathmandu Valley to detect phishing, spam, and malware in emails.')

col1, col2 = st.columns([2, 1])

with col1:
    st.header('Email Input')
    email_input = st.text_area('Enter email content:', height=200, placeholder="Paste email content here...")
    uploaded_file = st.file_uploader("Or upload a .txt or .eml file:", type=['txt', 'eml'])
    
    if st.button('Analyze Email', type='primary'):
        email_text = ""
        file_name = None
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            file_name = uploaded_file.name
            if uploaded_file.name.endswith('.eml'):
                msg = email.message_from_string(file_content, policy=policy.default)
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            email_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                else:
                    email_text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                email_text = file_content
        elif email_input:
            email_text = email_input
        else:
            st.warning("Please enter email content or upload a file to analyze")
            st.stop()

        if email_text:
            with st.spinner('Analyzing email...'):
                # Feature extraction
                feature_extractor = FeatureExtractor()
                features = feature_extractor.extract_all_features(email_text)
                
                # Text preprocessing
                processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', email_text.lower())
                processed_text = re.sub(r'\S+@\S+', '', processed_text)
                processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
                tokens = word_tokenize(processed_text)
                tokens = [PorterStemmer().stem(token) for token in tokens if token not in set(stopwords.words('english'))]
                processed_text = ' '.join(tokens)
                
                # Prediction
                email_vector = vectorizer.transform([processed_text])
                prediction = model.predict(email_vector)[0]
                probabilities = model.predict_proba(email_vector)[0]
                probability = probabilities.max()
                
                # Malware analysis
                malware_analysis = malware_detector.analyze_email(email_text)
                if malware_analysis['threat_score'] >= 4:
                    prediction = 'malware' if malware_analysis['risk_level'] in ['Medium', 'High'] else prediction
                
                # Generate alert
                alert = alert_system.process_threat_alert(prediction, email_text, probability)
                
                # Save to SQLite
                email_snippet = email_text[:100] + '...' if len(email_text) > 100 else email_text
                save_analysis_result(
                    email_snippet=email_snippet,
                    url_count=features['url_count'],
                    prediction=prediction,
                    confidence=probability,
                    threat_score=malware_analysis['threat_score'],
                    risk_level=malware_analysis['risk_level'],
                    file_name=file_name
                )
                
                # Display results
                st.subheader('Analysis Result')
                if prediction in ['phishing', 'spam', 'malware']:
                    st.error(f'🚨 Alert: This email is classified as **{prediction.upper()}** (Confidence: {probability:.2%})')
                else:
                    st.success(f'✅ This email is classified as **{prediction.upper()}** (Confidence: {probability:.2%})')
                
                if file_name:
                    st.info(f"Uploaded file: {file_name}")
                
                if malware_analysis['warnings']:
                    with st.expander("⚠️ Malware Warnings", expanded=True):
                        for warning in malware_analysis['warnings']:
                            st.warning(warning)
                
                st.subheader('Cybersecurity Tips')
                st.info(get_cybersecurity_tips(prediction))
                
                st.subheader('Confidence Scores')
                prob_df = pd.DataFrame({
                    'Category': model.classes_,
                    'Probability': probabilities
                })
                fig = px.bar(
                    prob_df, 
                    x='Category', 
                    y='Probability', 
                    color='Category',
                    color_discrete_map={
                        'phishing': '#FF4B4B',
                        'spam': '#FFA500',
                        'malware': '#FF0000',
                        'legitimate': '#00FF00'
                    },
                    title='Prediction Confidence Scores'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if alert:
                    with st.expander("🔔 Real-Time Alert Details", expanded=True):
                        st.json(alert)
                
                # Display history
                st.subheader('Analysis History')
                history = get_analysis_history()
                if history:
                    history_df = pd.DataFrame(history)
                    history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
                    st.table(history_df[['timestamp', 'email_snippet', 'url_count', 'prediction', 'confidence', 'threat_score', 'risk_level', 'file_name']])
                else:
                    st.info("No analysis history available.")

with col2:
    st.header('System Information')
    st.markdown('''
    ### Key Features:
    - Phishing detection
    - Spam identification  
    - Malware scanning
    - Legitimate email verification
    
    ### Technology Stack:
    - **Machine Learning**: Random Forest Classifier
    - **NLP**: TF-IDF with custom feature extraction
    - **Visualization**: Plotly interactive charts
    
    ### Data Sources:
    - SMS Spam Collection (Almeida & Hidalgo, 2012)
    - Synthetic phishing emails
    - Healthcare communication templates
    
    ### For IT Administrators:
    - Alerts logged to `security_alerts.log`
    - Model retraining available
    ''')
    
    st.divider()
    
    # Modified: Updated sample emails for health sector
    st.subheader('Sample Phishing Email')
    st.code('''URGENT: Your patient portal account will be suspended!
Click here to verify your details:
http://fake-hospital.com/verify?id=12345''', language='text')

    st.subheader('Sample Legitimate Email')
    st.code('''Dear Patient,
Your appointment on 2024-08-15 at Kathmandu Health Clinic is confirmed.
Please log in to your patient portal to view details.
Thank you for choosing our services.''', language='text')



