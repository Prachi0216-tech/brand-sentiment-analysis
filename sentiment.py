import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Single global analyzer instance — created once, reused everywhere.
# Creating it inside each function wastes memory on every call.
analyzer = SentimentIntensityAnalyzer()


def load_data(filepath="data/tweets.csv"):
    """
    Load tweet data from a CSV file.

    Args:
        filepath (str): Path to the CSV file. Defaults to 'data/tweets.csv'.

    Returns:
        pd.DataFrame: Loaded dataframe, or empty dataframe if file not found.
    """
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()
    return pd.read_csv(filepath)


def get_sentiment_label(compound_score):
    """
    Convert a VADER compound score into a sentiment label.

    Args:
        compound_score (float): VADER compound score in range [-1, 1].

    Returns:
        str: 'Positive', 'Negative', or 'Neutral'.
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_sentiment(text):
    """
    Analyze sentiment of a single piece of text using VADER.

    Args:
        text (str): Input text (tweet or review).

    Returns:
        str: Sentiment label — 'Positive', 'Negative', or 'Neutral'.
    """
    score = analyzer.polarity_scores(str(text))
    return get_sentiment_label(score["compound"])


def analyze_sentiment(df):
    """
    Apply sentiment analysis to an entire DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column.

    Returns:
        pd.DataFrame: Same DataFrame with a new 'predicted_sentiment' column added.
    """
    df = df.copy()  # Avoid mutating the original dataframe (best practice)
    df["predicted_sentiment"] = df["text"].apply(get_sentiment)
    return df


# Negative keywords that override VADER for domain-specific cases.
# This is a business rule layer — important for real-world NLP pipelines.
NEGATIVE_KEYWORDS = [
    "delay", "delayed", "cancel", "cancelled",
    "refund", "lost", "rude", "late", "poor", "bad"
]


def predict_review(text):
    """
    Predict sentiment of a user-entered review using VADER + keyword rules.

    Business rule: If the text contains known complaint keywords, it is
    classified as Negative regardless of VADER score. This handles sarcasm
    and domain-specific language that VADER may miss.

    Args:
        text (str): User-entered review text.

    Returns:
        str: Sentiment label — 'Positive', 'Negative', or 'Neutral'.
    """
    text_lower = text.lower()

    # Business rule override — check for complaint keywords first
    for word in NEGATIVE_KEYWORDS:
        if word in text_lower:
            return "Negative"

    # Fall back to VADER score
    score = analyzer.polarity_scores(text)["compound"]
    return get_sentiment_label(score)
