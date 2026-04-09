"""
model.py — Train and evaluate a sentiment classification model.

Pipeline:
    raw text → TF-IDF vectorizer → Logistic Regression classifier
    Labels: positive / negative / neutral  (3-class)

Usage:
    Run directly to train and save the model:
        python model.py

    Import in app.py or sentiment.py:
        from model import load_model, predict_ml
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# File paths — all in one place so they're easy to change
# ---------------------------------------------------------------------------
DATA_PATH  = "data/tweets.csv"
MODEL_PATH = "model.pkl"        # saved trained model
VECTORIZER_PATH = "vectorizer.pkl"  # saved TF-IDF vectorizer


# ---------------------------------------------------------------------------
# Step 1 — Load and clean data
# ---------------------------------------------------------------------------
def load_training_data(filepath=DATA_PATH):
    """
    Load tweet data and return clean text + label columns.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.Series: X (tweet text)
        pd.Series: y (sentiment labels: positive/negative/neutral)
    """
    df = pd.read_csv(filepath)

    # Drop rows where text or label is missing
    df = df.dropna(subset=["text", "airline_sentiment"])

    # Keep only the columns we need
    X = df["text"]
    y = df["airline_sentiment"]  # positive / negative / neutral

    print(f"Dataset loaded: {len(df)} rows")
    print(f"Label distribution:\n{y.value_counts()}\n")

    return X, y


# ---------------------------------------------------------------------------
# Step 2 — Train the model
# ---------------------------------------------------------------------------
def train_model(X, y):
    """
    Train a TF-IDF + Logistic Regression pipeline.

    Why TF-IDF?
        Converts text into numerical features by weighing words that are
        important to a document but not too common across all documents.

    Why Logistic Regression?
        Fast, interpretable, and works very well for text classification.
        It is a strong baseline that often beats complex models on small datasets.

    Args:
        X (pd.Series): Tweet text.
        y (pd.Series): Sentiment labels.

    Returns:
        vectorizer: Fitted TF-IDF vectorizer.
        model: Trained Logistic Regression model.
        X_test_vec, y_test: Held-out test data for evaluation.
    """
    # Split: 80% train, 20% test — stratified so class balance is preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,        # fixed seed = reproducible results
        stratify=y              # keeps positive/negative/neutral ratio equal in both splits
    )

    # TF-IDF vectorizer
    # max_features=5000 → keep the 5000 most important words
    # ngram_range=(1,2) → include both single words and two-word phrases
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"    # remove common words like 'the', 'is', 'at'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)    # only transform, never fit on test data

    # Logistic Regression
    # max_iter=1000 → enough iterations for the solver to converge on text data
    # class_weight='balanced' → handles unequal class sizes automatically
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    print("Model training complete.\n")
    return vectorizer, model, X_test_vec, y_test


# ---------------------------------------------------------------------------
# Step 3 — Evaluate the model
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test_vec, y_test):
    """
    Print accuracy, classification report, and save confusion matrix plot.

    Args:
        model: Trained Logistic Regression model.
        X_test_vec: TF-IDF transformed test features.
        y_test (pd.Series): True labels for test set.
    """
    y_pred = model.predict(X_test_vec)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}\n")

    # Per-class precision, recall, F1
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix — saved as an image for README / report
    labels = ["positive", "negative", "neutral"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Confusion Matrix — Sentiment Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png\n")


# ---------------------------------------------------------------------------
# Step 4 — Save and load model
# ---------------------------------------------------------------------------
def save_model(vectorizer, model):
    """
    Save the trained vectorizer and model to disk using pickle.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        model: Trained Logistic Regression model.
    """
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}\n")


def load_model():
    """
    Load the saved vectorizer and model from disk.

    Returns:
        vectorizer: Fitted TF-IDF vectorizer.
        model: Trained Logistic Regression model.

    Raises:
        FileNotFoundError: If model files don't exist yet (run model.py first).
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run `python model.py` first to train the model."
        )

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


# ---------------------------------------------------------------------------
# Step 5 — Predict using the trained ML model
# ---------------------------------------------------------------------------
def predict_ml(text):
    """
    Predict sentiment of a single review using the trained ML model.

    Args:
        text (str): Input review text.

    Returns:
        str: Predicted label — 'positive', 'negative', or 'neutral'.
    """
    vectorizer, model = load_model()
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction.capitalize()  # returns 'Positive', 'Negative', or 'Neutral'


# ---------------------------------------------------------------------------
# Run this file directly to train, evaluate, and save the model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Training Brand Sentiment Classifier")
    print("=" * 50 + "\n")

    X, y = load_training_data()
    vectorizer, model, X_test_vec, y_test = train_model(X, y)
    evaluate_model(model, X_test_vec, y_test)
    save_model(vectorizer, model)

    print("Done! You can now use predict_ml() in app.py.")
