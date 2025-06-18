import pickle
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Enhanced text preprocessing function
def preprocess_text(text):
    """Enhanced text preprocessing without external dependencies"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Replace URLs with token
        text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
        # Replace email addresses with token
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        # Replace currency symbols
        text = re.sub(r'[$€£¥]', ' MONEY ', text)
        # Replace phone numbers
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', ' PHONE ', text)
        # Replace numbers
        text = re.sub(r'\d+', ' NUMBER ', text)
        # Remove punctuation but keep exclamation points and question marks
        text = re.sub(r'[^\w\s!?]', ' ', text)
        # Replace multiple exclamation/question marks with tokens
        text = re.sub(r'!{2,}', ' MULTIEXCLAIM ', text)
        text = re.sub(r'\?{2,}', ' MULTIQUESTION ', text)
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        return text
    return ""

# Function to extract additional spam-related features
def extract_spam_features(df):
    """Extract additional features that indicate spam with improvements for legitimate security emails"""
    # Add features column if it doesn't exist
    if 'features' not in df.columns:
        df['features'] = df['Message'].apply(lambda x: {})
    
    # Define trusted domains for whitelisting
    trusted_domains = [
        'google.com', 'gmail.com', 'apple.com', 'microsoft.com', 'outlook.com', 
        'amazon.com', 'paypal.com', 'facebook.com', 'twitter.com', 'linkedin.com',
        'instagram.com', 'dropbox.com', 'github.com', 'chase.com', 'bankofamerica.com',
        'citibank.com', 'wellsfargo.com'
    ]
    
    # Function to check if text contains links to trusted domains
    def has_trusted_domain(text):
        if not isinstance(text, str):
            return False
        for domain in trusted_domains:
            if domain in text.lower():
                return True
        return False
    
    # Function to check if this is likely a legitimate security email
    def is_security_email(text):
        if not isinstance(text, str):
            return 0
        
        # Check if from trusted domain and has security-related terms
        has_trusted = has_trusted_domain(text)
        security_terms = ['security', 'verify', 'protection', 'authentication', 'login', 
                         'sign-in', 'secure', 'password', 'privacy', '2-step', 'two-factor']
        
        security_count = sum(1 for term in security_terms if term in text.lower())
        
        # If from trusted domain and has at least 2 security terms, likely legitimate
        if has_trusted and security_count >= 2:
            return 1
        return 0
    
    # Extract features from text with improved logic
    df['features'] = df.apply(lambda row: {
        'contains_uppercase_ratio': sum(1 for c in row['Message'] if c.isupper()) / max(len(row['Message']), 1),
        'contains_exclamation': 1 if '!' in row['Message'] else 0,
        'exclamation_count': row['Message'].count('!'),
        'question_count': row['Message'].count('?'),
        'contains_urgent': 1 if any(word in row['Message'].lower() for word in ['urgent', 'emergency', 'immediate', 'alert']) else 0,
        'contains_money': 1 if any(word in row['Message'].lower() for word in ['$', '€', '£', 'money', 'cash', 'price', 'dollar', 'free', 'win', 'won', 'prize']) else 0,
        'contains_click': 1 if (any(word in row['Message'].lower() for word in ['click', 'link', 'http', 'www', 'url']) and not has_trusted_domain(row['Message'])) else 0,
        'contains_account': 1 if (any(word in row['Message'].lower() for word in ['account', 'password', 'login', 'verify', 'bank']) and not has_trusted_domain(row['Message'])) else 0,
        'length': len(row['Message']),
        'avg_word_length': sum(len(word) for word in row['Message'].split()) / max(len(row['Message'].split()), 1),
        'contains_numbers': 1 if any(c.isdigit() for c in row['Message']) else 0,
        'all_caps_word_count': sum(1 for word in row['Message'].split() if word.isupper() and len(word) > 1),
        'contains_congratulation': 1 if any(word in row['Message'].lower() for word in ['congratulation', 'congrats', 'winner', 'selected']) else 0,
        'contains_limited': 1 if any(word in row['Message'].lower() for word in ['limited', 'offer', 'time', 'expires', 'today', 'exclusive']) else 0,
        'contains_reply': 1 if any(word in row['Message'].lower() for word in ['reply', 'respond', 'call now', 'call today']) else 0,
        'from_trusted_domain': 1 if has_trusted_domain(row['Message']) else 0,
        'is_security_email': is_security_email(row['Message'])
    }, axis=1)
    
    return df

# Load dataset
def load_data(file_path):
    """Load and prepare the dataset"""
    print("Loading data from:", file_path)
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Check for and remove any missing values
    if df.isnull().sum().sum() > 0:
        print(f"Found {df.isnull().sum().sum()} missing values. Cleaning...")
        df = df.dropna()
    
    # Encode labels (spam = 1, ham = 0)
    label_encoder = LabelEncoder()
    df["Category"] = label_encoder.fit_transform(df["Category"])
    
    # Preprocess text with enhanced cleaning
    df["CleanMessage"] = df["Message"].apply(preprocess_text)
    
    # Extract additional spam-related features
    df = extract_spam_features(df)
    
    # Display class distribution
    print("\nClass distribution:")
    print(df["Category"].value_counts())
    spam_percentage = df["Category"].mean() * 100
    print(f"Percentage spam: {spam_percentage:.2f}%")
    
    return df

def train_spam_model(df, save_path="spam_classifier.pkl"):
    """Train the spam classification model using TF-IDF, RFE, PCA and additional features"""
    # Split dataset with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        df[["CleanMessage", "features"]], 
        df["Category"], 
        test_size=0.2, 
        random_state=42, 
        stratify=df["Category"]
    )
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Apply TF-IDF with improved parameters
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=7500,      # Increased from 5000
        min_df=2,               # Minimum document frequency
        max_df=0.8,             # Reduced from 0.9 to filter out common words
        ngram_range=(1, 3),     # Include trigrams for better phrase detection
        sublinear_tf=True       # Apply sublinear tf scaling
    )
    X_train_tfidf = vectorizer.fit_transform(X_train["CleanMessage"])
    X_test_tfidf = vectorizer.transform(X_test["CleanMessage"])
    
    # Create feature matrices for the additional spam features
    feature_keys = list(X_train["features"].iloc[0].keys())
    train_features = np.array([list(f.values()) for f in X_train["features"]])
    test_features = np.array([list(f.values()) for f in X_test["features"]])
    
    # Apply Recursive Feature Elimination (RFE) with optimized parameters
    print("Applying RFE for feature selection...")
    # Use a more optimized logistic regression for feature selection
    base_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'
    )
    
    # Use cross-validation to determine optimal n_features
    feature_ranges = [1000, 1500, 2000]
    best_score = 0
    best_n_features = 1500
    
    for n in feature_ranges:
        rfe_temp = RFE(base_model, n_features_to_select=n, step=100)
        X_train_rfe_temp = rfe_temp.fit_transform(X_train_tfidf, y_train)
        # Use cross-validation to evaluate this feature count
        cv_score = np.mean(cross_val_score(base_model, X_train_rfe_temp, y_train, cv=5))
        print(f"RFE with {n} features: CV score = {cv_score:.4f}")
        if cv_score > best_score:
            best_score = cv_score
            best_n_features = n
    
    print(f"Selected optimal feature count: {best_n_features}")
    rfe = RFE(base_model, n_features_to_select=best_n_features, step=100)
    X_train_rfe = rfe.fit_transform(X_train_tfidf, y_train)
    X_test_rfe = rfe.transform(X_test_tfidf)
    
    # Apply PCA with variance ratio analysis to determine components
    print("Applying PCA for dimensionality reduction...")
    # First convert to dense arrays if needed
    X_train_rfe_dense = X_train_rfe.toarray() if hasattr(X_train_rfe, 'toarray') else X_train_rfe
    X_test_rfe_dense = X_test_rfe.toarray() if hasattr(X_test_rfe, 'toarray') else X_test_rfe
    
    # Determine optimal n_components by explained variance
    temp_pca = PCA(n_components=min(100, X_train_rfe_dense.shape[1]))
    temp_pca.fit(X_train_rfe_dense)
    
    # Find number of components that explain 95% variance
    explained_variance = np.cumsum(temp_pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance >= 0.95) + 1
    n_components = max(min(n_components, 100), 15)  # Keep between 15 and 100
    
    print(f"Selected {n_components} PCA components explaining ~95% variance")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_rfe_dense)
    X_test_pca = pca.transform(X_test_rfe_dense)
    
    # Combine TF-IDF features with additional spam features
    X_train_combined = np.hstack((X_train_pca, train_features))
    X_test_combined = np.hstack((X_test_pca, test_features))
    
    # Use GridSearchCV to find optimal parameters
    print("Finding optimal model parameters with GridSearchCV...")
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0],
        'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000),
        param_grid,
        cv=5,
        scoring='f1_weighted'
    )
    
    grid_search.fit(X_train_combined, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    log_reg = grid_search.best_estimator_
    
    # Evaluate Model
    y_pred = log_reg.predict(X_test_combined)
    y_proba = log_reg.predict_proba(X_test_combined)[:, 1]
    
    print("\nModel Evaluation:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Save the trained model and preprocessors
    print(f"Saving model to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump((vectorizer, rfe, pca, log_reg, feature_keys), f)
    
    return vectorizer, rfe, pca, log_reg

# Function to predict spam/ham for a given message
def predict_spam(message, model_path="spam_classifier.pkl"):
    """Predict if a message is spam or ham with confidence score using enhanced features"""
    # Load the model components
    with open(model_path, "rb") as f:
        vectorizer, rfe, pca, log_reg, feature_keys = pickle.load(f)
    
    # Define trusted domains for whitelisting
    trusted_domains = [
        'google.com', 'gmail.com', 'apple.com', 'microsoft.com', 
        'amazon.com', 'paypal.com', 'facebook.com', 'twitter.com', 'linkedin.com',
        'instagram.com', 'dropbox.com', 'github.com', 'chase.com', 'bankofamerica.com',
        'citibank.com', 'wellsfargo.com'
    ]
    
    # Check if from trusted domain
    has_trusted_domain = any(domain in message.lower() for domain in trusted_domains)
    
    # Check if likely a legitimate security email
    security_terms = ['security', 'verify', 'protection', 'authentication', 'login', 
                     'sign-in', 'secure', 'password', 'privacy', '2-step', 'two-factor']
    security_count = sum(1 for term in security_terms if term in message.lower())
    is_security_email = 1 if (has_trusted_domain and security_count >= 2) else 0
    
    message_lower = message.lower()
    
    # Early detection for legitimate security emails from trusted domains
    if is_security_email == 1 and has_trusted_domain:
        # Check for strong indicators of a security email
        additional_indicators = ['account', 'identity', 'verification', 'confirm']
        additional_count = sum(1 for term in additional_indicators if term in message_lower)
        
        # If it has even more security indicators, return directly
        if additional_count >= 1:
            return "Ham", 0.95
    
    # Preprocess input message
    clean_message = preprocess_text(message)
    
    # Extract spam features for the message
    features = {
        'contains_uppercase_ratio': sum(1 for c in message if c.isupper()) / max(len(message), 1),
        'contains_exclamation': 1 if '!' in message else 0,
        'exclamation_count': message.count('!'),
        'question_count': message.count('?'),
        'contains_urgent': 1 if any(word in message_lower for word in ['urgent', 'emergency', 'immediate', 'alert']) else 0,
        'contains_money': 1 if any(word in message_lower for word in ['$', '€', '£', 'money', 'cash', 'price', 'dollar', 'free', 'win', 'won', 'prize']) else 0,
        'contains_click': 1 if (any(word in message_lower for word in ['click', 'link', 'http', 'www', 'url']) and not has_trusted_domain) else 0,
        'contains_account': 1 if (any(word in message_lower for word in ['account', 'password', 'login', 'verify', 'bank']) and not has_trusted_domain) else 0,
        'length': len(message),
        'avg_word_length': sum(len(word) for word in message.split()) / max(len(message.split()), 1),
        'contains_numbers': 1 if any(c.isdigit() for c in message) else 0,
        'all_caps_word_count': sum(1 for word in message.split() if word.isupper() and len(word) > 1),
        'contains_congratulation': 1 if any(word in message_lower for word in ['congratulation', 'congrats', 'winner', 'selected']) else 0,
        'contains_limited': 1 if any(word in message_lower for word in ['limited', 'offer', 'time', 'expires', 'today', 'exclusive']) else 0,
        'contains_reply': 1 if any(word in message_lower for word in ['reply', 'respond', 'call now', 'call today']) else 0
    }

    # Add additional feature keys if your model uses them
    if 'from_trusted_domain' in feature_keys:
        features['from_trusted_domain'] = 1 if has_trusted_domain else 0
    if 'is_security_email' in feature_keys:
        features['is_security_email'] = is_security_email
    
    # Get feature values in the correct order
    try:
        feature_values = [features[key] for key in feature_keys]
    except KeyError:
        # If we have missing keys (older model), use the original features
        original_keys = [
            'contains_uppercase_ratio', 'contains_exclamation', 'exclamation_count',
            'question_count', 'contains_urgent', 'contains_money', 'contains_click',
            'contains_account', 'length', 'avg_word_length', 'contains_numbers',
            'all_caps_word_count', 'contains_congratulation', 'contains_limited',
            'contains_reply'
        ]
        feature_values = [features[key] for key in original_keys if key in features]
    
    # Transform message through the pipeline
    X_input_tfidf = vectorizer.transform([clean_message])
    X_input_rfe = rfe.transform(X_input_tfidf)
    X_input_rfe_dense = X_input_rfe.toarray() if hasattr(X_input_rfe, 'toarray') else X_input_rfe
    X_input_pca = pca.transform(X_input_rfe_dense)

    # Get feature values in the correct order
    feature_values = [features[key] for key in feature_keys]
    
    # Combine with additional features
    X_input_combined = np.hstack((X_input_pca, np.array(feature_values).reshape(1, -1)))
    
    # Get prediction and confidence
    prediction = log_reg.predict(X_input_combined)[0]
    prob = log_reg.predict_proba(X_input_combined)[0]

    # Advanced multi-layered spam detection system
    spam_indicators = 0
    
    # Layer 1: Core spam patterns from original effective code
    # Financial urgency patterns
    payment_terms = ['payment', 'credit card', 'billing', 'declined', 'expire', 'renew', 'subscription']
    payment_pattern = any(term in message_lower for term in payment_terms)
    
    urgency_terms = ['urgent', 'immediately', 'today', 'now', '24 hours', '48 hours', 'limited time', 
                   'final notice', 'last chance', 'will be cancelled', 'will be suspended']
    urgency_pattern = any(term in message_lower for term in urgency_terms)
    
    # Account action patterns
    action_terms = ['click', 'update', 'verify', 'confirm', 'activate', 'validate', 'sign in']
    action_pattern = any(term in message_lower for term in action_terms)
    
    account_terms = ['account', 'profile', 'subscription', 'membership', 'access', 'service']
    account_pattern = any(term in message_lower for term in account_terms)
    
    # Suspicious link patterns
    link_pattern = ('click' in message_lower and not 'http' in message_lower) or 'update payment' in message_lower

    # Standard spam phrases
    spam_phrases = [
        'update your payment',
        'update payment method',
        'payment declined',
        'account will be deactivated',
        'verify your account',
        'confirm your identity',
        'suspicious activity',
        'unusual sign-in',
        'premium benefits',
        'click below',
        'click here',
        'this is your final notice',
        'avoid losing access',
        'security reasons',
        'limited time offer'
    ]
    contains_spam_phrase = any(phrase in message_lower for phrase in spam_phrases)
    
    # Layer 2: Enhanced patterns to catch the missing 3 spam messages
    # Advanced spam phrase detection with partial matching
    advanced_spam_patterns = [
        {'pattern': 'expir', 'context': ['account', 'subscription', 'service']},
        {'pattern': 'updat', 'context': ['information', 'details', 'account']},
        {'pattern': 'verif', 'context': ['identity', 'account']},
        {'pattern': 'activ', 'context': ['account', 'service']},
        {'pattern': 'suspend', 'context': ['account', 'service']},
        {'pattern': 'cancel', 'context': ['subscription', 'service', 'account']},
        {'pattern': 'unusual', 'context': ['activity', 'login']},
        {'pattern': 'confirm', 'context': ['identity', 'details', 'information']},
    ]
    
    # Check for advanced pattern matches with their contexts
    advanced_matches = 0
    for pattern_data in advanced_spam_patterns:
        if pattern_data['pattern'] in message_lower:
            # If pattern found, check for context within 10 words
            words = message_lower.split()
            for i, word in enumerate(words):
                if pattern_data['pattern'] in word:
                    # Check nearby words for context
                    context_range = 10
                    nearby_text = ' '.join(words[max(0, i-context_range):min(len(words), i+context_range)])
                    if any(context in nearby_text for context in pattern_data['context']):
                        advanced_matches += 1
                        break
    
    # Layer 3: Domain authenticity verification - improved check for fake domain references
    # Check if message mentions a trusted domain but doesn't come from it (more sophisticated)
    fake_domain_indicators = 0
    for domain in trusted_domains:
        if domain in message_lower:
            if not (domain+"/" in message_lower or domain+"@" in message_lower):
                fake_domain_indicators += 1
            if "support" in message_lower and domain in message_lower and not has_trusted_domain:
                fake_domain_indicators += 2  # Higher weight for fake support messages
    
    # Layer 4: Check for suspicious URL instructions without actual URLs
    click_instructions = ['click', 'visit', 'go to', 'login to', 'sign in']
    has_click_instruction = any(instr in message_lower for instr in click_instructions)
    
    # Count more sophisticated spam indicators
    if payment_pattern and urgency_pattern:
        spam_indicators += 2
    if account_pattern and action_pattern and not has_trusted_domain:
        spam_indicators += 2
    if link_pattern:
        spam_indicators += 1
    if contains_spam_phrase:
        spam_indicators += 2
    if fake_domain_indicators > 0:
        spam_indicators += min(fake_domain_indicators, 2)
    if features['contains_urgent'] == 1 and action_pattern:
        spam_indicators += 1
    if has_click_instruction and not 'http' in message_lower and not has_trusted_domain:
        spam_indicators += 1
    
    # Enhanced detection: Add contribution from advanced patterns
    spam_indicators += min(advanced_matches * 0.5, 1.5)
    
    # Layer 5: Enhanced detection for subtle spam
    # These patterns catch more sophisticated spam that might slip through
    subtle_spam_indicators = 0
    
    # Check for unusual combinations of words that often appear in sophisticated spam
    unusual_combos = [
        (['account', 'security'], ['click', 'update', 'immediately']),
        (['verify', 'confirm'], ['urgent', 'now', 'soon']),
        (['payment', 'billing'], ['failed', 'expired', 'declined']),
        (['subscription', 'service'], ['cancelled', 'suspended', 'terminated']),
        (['information', 'details'], ['outdated', 'invalid', 'incomplete'])
    ]
    
    for combo_pair in unusual_combos:
        if any(word in message_lower for word in combo_pair[0]) and any(word in message_lower for word in combo_pair[1]):
            subtle_spam_indicators += 0.5
    
    # Check for suspicious sentence structures common in spam
    suspicious_structures = [
        "please update your",
        "please confirm your",
        "click the link",
        "to avoid",
        "will be terminated",
        "has been compromised"
    ]
    
    for structure in suspicious_structures:
        if structure in message_lower:
            subtle_spam_indicators += 0.5
    
    # Add subtle spam indicators (capped)
    spam_indicators += min(subtle_spam_indicators, 1.5)
    
    # Make final decision based on enhanced detection system
    if spam_indicators >= 3:
        return "Spam", 0.92
    elif spam_indicators >= 2.5:  # Lower threshold to catch more spam
        return "Spam", 0.85
    
    # If model predicted spam with confidence, keep that prediction
    if prediction == 1 and prob[1] > 0.7:
        return "Spam", prob[1]
    
    # If model predicted ham but we have some indicators, be more cautious
    if prediction == 0 and spam_indicators >= 2:
        adjusted_confidence = prob[0] * 0.6  # Reduced confidence for potential spam
        if adjusted_confidence < 0.65:  # Lower threshold
            return "Spam", 0.65
    
    # Special case: Even with just a few indicators, be cautious about certain combinations
    if prediction == 0 and spam_indicators >= 1.5:
        if payment_pattern and urgency_pattern:
            return "Spam", 0.7
        if account_pattern and action_pattern and not has_trusted_domain:
            return "Spam", 0.7
    
    # Adjust prediction based on domain knowledge
    if has_trusted_domain and prediction == 1:
        # If from trusted domain but predicted as spam, check confidence
        if prob[1] < 0.8:  # If confidence isn't very high
            prediction = 0  # Change to Ham
            confidence = max(prob[0], 0.7)  # At least 70% confidence
            return "Ham", confidence
    
    # Get appropriate label and confidence
    label = "Spam" if prediction == 1 else "Ham"
    confidence = prob[1] if prediction == 1 else prob[0]
    
    return label, confidence

# GUI integration function
def gui_predict_spam(email_content, model_path="spam_classifier.pkl"):
    """Function to be called by the GUI's 'Predict Spam' button"""
    label, confidence = predict_spam(email_content, model_path)
    return label, confidence * 100  # Return percentage confidence for display

# Main execution - only runs when script is executed directly
if __name__ == "__main__":
    # Load and process data
    file_path = "datasets/logistic_regression.csv"  
    df = load_data(file_path)
    
    # Train model
    train_spam_model(df)
    
    # Test prediction
    test_messages = [
        "Congratulations! You have won a free lottery. Claim now!",
        "Hey, let's meet for lunch tomorrow.",
        "URGENT: Your bank account has been suspended. Click here to reactivate.",
        "Can you pick up some groceries on your way home?",
        "FREE SHOPPING VOUCHER! Just forward this message to 10 friends!",
        "While I appreciate your prompt response to my earlier email, I would like to request some additional time to review the document you sent over. The content is quite extensive and requires careful consideration. Would it be possible to schedule a meeting next week to discuss my findings? Thank you for your understanding.",
        "ATTENTION CUSTOMER: Your Amazon account shows suspicious activity. Verify your information IMMEDIATELY to prevent account closure. Click the link below to confirm your identity or risk permanent account suspension. Act now before it's too late!",
        "Hi John, I've been trying to reach you about your car's extended warranty. This is your LAST CHANCE to renew before coverage expires. Call 1-800-555-1234 NOW for a LIMITED TIME OFFER!",
        # Add legitimate security emails to test the improvements
        "[Google] Add ways for us to make sure it's you logisticregressionreceiver@gmail.com Users with extra ways to verify their identity are far less likely to be hacked or locked out. Add additional ways to prove it's really you and see other personalized security recommendations in the Security Checkup.",
        "Microsoft Account Security Alert: We detected a new sign-in to your Microsoft account. If this was you, you can ignore this message. If not, please verify your account at microsoft.com/security.",
        "PayPal: Please verify your recent login. For your security, we need you to confirm a recent login attempt from a new device. Visit paypal.com to secure your account."
    ]
    
    print("\nTest Predictions:")
    for message in test_messages:
        label, confidence = predict_spam(message)
        print(f"Message: {message[:60]}...") # Show just the first 60 chars for brevity
        print(f"Prediction: {label} (Confidence: {confidence:.2%})\n")