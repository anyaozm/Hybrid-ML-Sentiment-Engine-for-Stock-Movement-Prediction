import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def add_sentiment_column(df: pd.DataFrame, headlines: list) -> pd.DataFrame:
    if len(headlines) != len(df):
        raise ValueError("Headlines and DataFrame length mismatch")
    sentiments = [sia.polarity_scores(text)["compound"] for text in headlines]
    df["sentiment"] = sentiments
    return df
