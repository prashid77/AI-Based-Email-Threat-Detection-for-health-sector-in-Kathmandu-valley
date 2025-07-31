import pandas as pd
import numpy as np
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tldextract
from urllib.parse import urlparse
from textstat import flesch_kincaid_grade
import string
import logging
from datetime import datetime
import json

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DataLoader:
    def __init__(self):
        self.data_sources = {
            'sms_spam': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip',
            'phishing_urls': 'https://data.mendeley.com/datasets/c2gw7fy2j4/3',
            'enron_emails': 'https://www.cs.cmu.edu/~enron/'
        }
        self.citations = {
            'sms_spam': 'Almeida, T. A., & Hidalgo, J. M. G. (2012). SMS spam collection data set. UCI Machine Learning Repository.',
            'phishing_urls': 'Vrbančič, G., Fister Jr, I., & Podgorelec, V. (2018). Datasets for phishing websites detection. Data in Brief, 20, 1560-1564.',
            'enron_emails': 'Klimt, B., & Yang, Y. (2004). The Enron corpus: A new dataset for email classification research. European Conference on Machine Learning, 217-226.'
        }

    def download_sms_spam_data(self):
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        try:
            df = pd.read_csv(url, encoding='latin-1')
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
            df['label'] = df['label'].map({'ham': 'legitimate', 'spam': 'spam'})
            return df
        except Exception as e:
            print(f"Error downloading SMS spam data: {e}")
            return None

    def create_phishing_data(self):
        phishing_emails = []
        # Modified: Updated templates for health sector
        templates = [
            "URGENT: Your {service} patient portal will be suspended in 24 hours. Click here to verify: {url}",
            "Security Alert: Suspicious activity detected on your {service} medical account. Update your password at: {url}",
            "Your {service} patient record has been compromised. Secure it immediately: {url}",
            "Action Required: Your {service} health account needs verification. Click: {url}",
            "Important: Update your {service} medical details or lose access. Visit: {url}",
            "Your {service} patient portal is locked. Unlock it now: {url}",
            "We detected unusual activity on your {service} health account. Verify at: {url}",
            "Your {service} medical bill payment failed. Update your information: {url}",
            "Confirm your {service} patient account to avoid suspension: {url}",
            "Your {service} appointment booking will expire soon. Renew at: {url}"
        ]
        services = ['hospital', 'clinic', 'patient portal', 'telemedicine', 'healthcare', 'medical center', 'pharmacy', 'lab']
        fake_urls = [f'http://fake-{service.lower().replace(" ", "-")}.com/verify' for service in services]
        for i, template in enumerate(templates):
            for j, service in enumerate(services):
                email_text = template.format(service=service, url=fake_urls[j])
                phishing_emails.append({'text': email_text, 'label': 'phishing'})
        return pd.DataFrame(phishing_emails)

    def create_legitimate_health_data(self):  # Modified: Renamed and updated for health sector
        legitimate_templates = [
            "Dear patient, your appointment at {hospital} on {date} is confirmed.",
            "Your recent medical bill of ${amount} at {hospital} has been processed successfully.",
            "Welcome to our patient portal. Your health account is now active.",
            "Your prescription request has been approved. Please visit {hospital} for collection.",
            "Thank you for choosing our healthcare services. Your medical report is available on {date}.",
            "Your health insurance claim for {hospital} will be processed by {date}.",
            "We have received your request for a medical certificate. It will be available within 5 business days.",
            "Your telehealth appointment for ${amount} has been scheduled successfully.",
            "Annual health check-up reminder for your account. Book now at {hospital}.",
            "Your medical record as of {date} is available. Please check your patient portal."
        ]
        legitimate_emails = []
        for i, template in enumerate(legitimate_templates):
            for j in range(5):
                email_text = template.format(
                    hospital=f"Kathmandu Health {j + 1}", 
                    amount=f"{100 + j * 10}", 
                    date=f"2024-{(j % 12) + 1:02d}-15"
                )
                legitimate_emails.append({'text': email_text, 'label': 'legitimate'})
        return pd.DataFrame(legitimate_emails)

    def load_all_data(self):
        print("Loading SMS Spam Collection dataset...")
        sms_data = self.download_sms_spam_data()
        print("Creating phishing email dataset...")
        phishing_data = self.create_phishing_data()
        print("Creating legitimate health email dataset...")
        legitimate_data = self.create_legitimate_health_data()
        
        all_data = []
        if sms_data is not None:
            all_data.append(sms_data)
        all_data.extend([phishing_data, legitimate_data])
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print("Balancing dataset...")
        combined_df = self.balance_dataset(combined_df)
        return combined_df

    def balance_dataset(self, df):
        min_count = df['label'].value_counts().min()
        balanced_data = []
        for label in df['label'].unique():
            label_data = df[df['label'] == label].sample(n=min_count, random_state=42)
            balanced_data.append(label_data)
        return pd.concat(balanced_data, ignore_index=True)

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

class ModelTrainer:
    def __init__(self):
        self.models = {'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)}
        self.vectorizer = None
        self.best_model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def train_models(self, df):
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        X_text = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2), 
            min_df=2, 
            max_df=0.95, 
            stop_words='english'
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print("Training Random Forest...")
        self.best_model = self.models['random_forest']
        self.best_model.fit(X_train_tfidf, y_train)
        
        y_pred = self.best_model.predict(X_test_tfidf)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.best_model, self.vectorizer

    def save_models(self, model_path='email_threat_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Models saved: {model_path}, {vectorizer_path}")

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

def main():
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Initializing components
    data_loader = DataLoader()
    feature_extractor = FeatureExtractor()
    model_trainer = ModelTrainer()
    
    # Loading and preparing data
    print("Loading and preparing data...")
    df = data_loader.load_all_data()
    df['features'] = df['text'].apply(feature_extractor.extract_all_features)
    
    # Training models
    print("Training models...")
    best_model, vectorizer = model_trainer.train_models(df)
    
    # Saving models
    model_trainer.save_models()
    
    print("Training completed and models saved!")

if __name__ == "__main__":
    main()
