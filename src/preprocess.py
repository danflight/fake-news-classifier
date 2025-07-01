import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text, stopwords_set=STOPWORDS, lemmatize=True, expand_contractions=True):
    """
    Cleans the input text by performing the following operations:
    - Converts text to lowercase
    - Removes URLs
    - Removes mentions and hashtags
    - Removes non-alphabetic characters
    - Expands contractions (optional)
    - Removes English stopwords (customizable)
    - Lemmatizes words (optional)
    - Removes extra whitespace
    - Handles non-string and empty input gracefully

    Args:
        text (str): The input text to clean.
        stopwords_set (set, optional): Set of stopwords to remove. Defaults to English stopwords.
        lemmatize (bool, optional): Whether to lemmatize words. Defaults to True.
        expand_contractions (bool, optional): Whether to expand contractions. Defaults to True.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text) 
    text = re.sub(r"[^a-z\s]", '', text)
    if expand_contractions:
        contractions = {"don't": "do not", "can't": "cannot", "won't": "will not", "it's": "it is", "i'm": "i am", "you're": "you are", "they're": "they are", "we're": "we are", "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "doesn't": "does not", "didn't": "did not", "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not", "mustn't": "must not"}
        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)
    words = text.split()
    words = [word for word in words if word and word not in stopwords_set]
    if lemmatize:
        words = [LEMMATIZER.lemmatize(word) for word in words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def vectorize(texts, vectorizer=None):
    """
    Vectorizes a list of texts using TF-IDF. Can use a pre-fitted vectorizer for consistency.

    Args:
        texts (list of str): The input texts to vectorize.
        vectorizer (TfidfVectorizer, optional): Pre-fitted vectorizer. If None, a new one is fitted.

    Returns:
        tuple: (features, vectorizer)
            features: TF-IDF feature matrix (scipy.sparse matrix)
            vectorizer: The fitted or provided TfidfVectorizer instance
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
    else:
        features = vectorizer.transform(texts)
    return features, vectorizer
