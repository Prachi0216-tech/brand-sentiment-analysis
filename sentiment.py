import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def load_data():
    df = pd.read_csv("data/tweets.csv")
    return df

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))
    compound = score['compound']
    
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment(df):
    df["predicted_sentiment"] = df["text"].apply(get_sentiment)
    return df
from textblob import TextBlob

def predict_review(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']

    text_lower = text.lower()

    # Business rule override (important for real-world cases)
    negative_keywords = ["delay", "delayed", "cancel", "cancelled", 
                         "refund", "lost", "rude", "late", "poor", "bad"]

    for word in negative_keywords:
        if word in text_lower:
            return "Negative"

    # VADER decision
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"
