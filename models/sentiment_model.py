from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline
import nltk

nltk.download("vader_lexicon")

# VADER Analyzer (lightweight, good for finance headlines)
vader_analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    return vader_analyzer.polarity_scores(text)["compound"]

# TextBlob (alternative sentiment scoring)
def textblob_score(text):
    return TextBlob(text).sentiment.polarity

# Optional: Transformer-based sentiment model (FinBERT or general)
try:
    transformer_pipeline = pipeline("sentiment-analysis")
except Exception:
    transformer_pipeline = None

def transformer_score(text):
    if transformer_pipeline is None:
        return 0.0
    result = transformer_pipeline(text)[0]
    return 1.0 if result["label"].lower() == "positive" else -1.0 if result["label"].lower() == "negative" else 0.0
