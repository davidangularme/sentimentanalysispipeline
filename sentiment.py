import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
import random
from collections import Counter

def create_improved_simulated_lexicons():
    # Expanded word list
    words = ['good', 'bad', 'great', 'terrible', 'awesome', 'awful', 'excellent', 'poor', 'wonderful', 'horrible',
             'happy', 'sad', 'angry', 'calm', 'excited', 'bored', 'interesting', 'dull', 'important', 'trivial',
             'easy', 'difficult', 'simple', 'complex', 'clear', 'confusing', 'innovative', 'traditional', 'efficient', 'inefficient']
    
    # AFINN lexicon
    afinn = pd.DataFrame({
        'word': words,
        'score': [random.randint(-5, 5) for _ in words]
    })
    
    # Bing lexicon
    bing = pd.DataFrame({
        'word': words,
        'sentiment': [random.choice(['positive', 'negative']) for _ in words]
    })
    
    # NRC lexicon
    emotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust']
    nrc = pd.DataFrame({
        'word': words,
        **{emotion: [random.choice([0, 1]) for _ in words] for emotion in emotions}
    })
    
    # Loughran-McDonald lexicon
    categories = ['positive', 'negative', 'uncertainty', 'litigious', 'constraining', 'superfluous']
    loughran = pd.DataFrame({
        'word': words,
        **{category: [random.choice([0, 1]) for _ in words] for category in categories}
    })
    
    return afinn, bing, nrc, loughran

def generate_improved_sample_abstract(source):
    common_words = ['the', 'of', 'and', 'a', 'in', 'to', 'is', 'was', 'for', 'that', 'on', 'with', 'by', 'as', 'are']
    scientific_words = ['study', 'research', 'analysis', 'results', 'data', 'method', 'experiment', 'hypothesis',
                        'theory', 'observation', 'conclusion', 'evidence', 'significant', 'correlation', 'variable']
    domain_words = ['plant', 'growth', 'species', 'environmental', 'genetic', 'factors', 'photosynthesis', 'root',
                    'leaf', 'soil', 'climate', 'adaptation', 'ecosystem', 'biodiversity', 'molecular', 'cellular', 'physiological']
    
    if source == 'human':
        word_distribution = {
            'common': 0.5,
            'scientific': 0.3,
            'domain': 0.2
        }
        sentence_length_range = (15, 25)
        paragraph_length_range = (3, 5)
    else:  # chatgpt
        word_distribution = {
            'common': 0.4,
            'scientific': 0.35,
            'domain': 0.25
        }
        sentence_length_range = (10, 20)
        paragraph_length_range = (4, 6)
    
    paragraphs = []
    for _ in range(random.randint(*paragraph_length_range)):
        sentences = []
        for _ in range(random.randint(*sentence_length_range)):
            sentence_words = []
            for _ in range(random.randint(*sentence_length_range)):
                r = random.random()
                if r < word_distribution['common']:
                    sentence_words.append(random.choice(common_words))
                elif r < word_distribution['common'] + word_distribution['scientific']:
                    sentence_words.append(random.choice(scientific_words))
                else:
                    sentence_words.append(random.choice(domain_words))
            sentences.append(' '.join(sentence_words).capitalize() + '.')
        paragraphs.append(' '.join(sentences))
    
    abstract = '\n\n'.join(paragraphs)
    
    if source == 'human':
        abstract = f"In this study, we investigated {abstract.lower()}"
    else:  # chatgpt
        abstract = f"This research explores {abstract.lower()}"
    
    return abstract

def create_improved_sample_dataset(n_samples=145):
    data = []
    for i in range(n_samples):
        source = 'human' if i < n_samples // 2 else 'chatgpt'
        abstract = generate_improved_sample_abstract(source)
        data.append({'abstract': abstract, 'source': source})
    return pd.DataFrame(data)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text.split()

def extract_sentiment_features(tokens, afinn, bing, nrc, loughran):
    features = {}
    
    # Bing lexicon
    bing_sentiments = bing[bing['word'].isin(tokens)]['sentiment'].value_counts()
    features['bing_positive'] = bing_sentiments.get('positive', 0)
    features['bing_negative'] = bing_sentiments.get('negative', 0)
    
    # Afinn lexicon
    afinn_scores = afinn[afinn['word'].isin(tokens)]['score']
    features['afinn_avg'] = afinn_scores.mean() if not afinn_scores.empty else 0
    features['afinn_std'] = afinn_scores.std() if not afinn_scores.empty else 0
    
    # NRC lexicon
    for emotion in nrc.columns[1:]:
        features[f'nrc_{emotion}'] = nrc[nrc['word'].isin(tokens)][emotion].sum()
    
    # Loughran-McDonald lexicon
    for category in loughran.columns[1:]:
        features[f'loughran_{category}'] = loughran[loughran['word'].isin(tokens)][category].sum()
    
    # Additional linguistic features
    features['text_length'] = len(tokens)
    features['unique_words'] = len(set(tokens))
    features['avg_word_length'] = sum(len(word) for word in tokens) / len(tokens) if tokens else 0
    
    return features

def main():
    afinn, bing, nrc, loughran = create_improved_simulated_lexicons()
    data = create_improved_sample_dataset()
    
    # Extract sentiment and linguistic features
    features = []
    for text in data['abstract']:
        tokens = preprocess_text(text)
        features.append(extract_sentiment_features(tokens, afinn, bing, nrc, loughran))
    
    X = pd.DataFrame(features)
    y = data['source']
    
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), X.columns)
        ])
    
    # Create a pipeline with preprocessor and random forest classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_impurity_decrease=1e-3, random_state=42))
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=10)
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train and evaluate on a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
