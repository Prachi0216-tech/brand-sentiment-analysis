# 💬 Brand Sentiment Analysis

A machine learning web application that analyzes airline customer reviews to classify sentiment and identify common complaint patterns.

Built with **Python**, **Scikit-learn**, **VADER**, and **Streamlit**.

---

## 🖥️ Live Demo

> Run locally using the steps below.

---

## 📌 Problem Statement

Airlines receive thousands of customer reviews daily. Manually reading each one is impossible. This project automates sentiment classification — identifying whether a review is **Positive**, **Negative**, or **Neutral** — and surfaces the most common issues like delays, cancellations, and refund complaints.

---

## ⚙️ Features

- 🔍 **Sentiment Classification** — Classifies reviews as Positive, Negative, or Neutral
- 🤖 **ML Model** — TF-IDF + Logistic Regression trained on real airline tweet data
- 📊 **Interactive Dashboard** — Airline-wise filtering, sentiment distribution chart, complaint metrics
- ⚡ **Live Prediction** — Type any review and get instant sentiment prediction
- 🧠 **Dual Approach** — ML model + VADER rule-based comparison

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| ML Model | Scikit-learn (TF-IDF + Logistic Regression) |
| NLP | VADER Sentiment Analyzer |
| Dashboard | Streamlit |
| Visualization | Plotly, Seaborn, Matplotlib |
| Data | Pandas |

---

## 📁 Project Structure

```
brand-sentiment-analysis/
│
├── data/
│   └── tweets.csv           # Airline customer review dataset
│
├── app.py                   # Streamlit dashboard (main UI)
├── sentiment.py             # VADER-based sentiment functions
├── model.py                 # TF-IDF + Logistic Regression model
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Prachi0216-tech/brand-sentiment-analysis.git
cd brand-sentiment-analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML model
```bash
python model.py
```
This will train the classifier and save `model.pkl` and `vectorizer.pkl` locally.  
It will also generate a `confusion_matrix.png` showing model performance.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

The Logistic Regression classifier is trained on 80% of the data and evaluated on the remaining 20%.

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Features | TF-IDF (5000 features, unigrams + bigrams) |
| Classes | Positive / Negative / Neutral |
| Train/Test Split | 80% / 20% (stratified) |

> Run `python model.py` to see the full classification report and accuracy on your machine.

---

## 🧠 Approach

```
Raw Tweet Text
      ↓
Text Preprocessing (lowercasing, stopword removal)
      ↓
TF-IDF Vectorization (5000 features, bigrams)
      ↓
Logistic Regression Classifier
      ↓
Sentiment Label: Positive / Negative / Neutral
```

**Why Logistic Regression?**
Fast, interpretable, and performs strongly on text classification tasks.
It serves as a solid baseline before experimenting with deep learning approaches.

**Why TF-IDF?**
Weighs words by how important they are to a specific document relative to the entire dataset — more meaningful than simple word counts.

---

## 📂 Dataset

- **Source:** [Twitter US Airline Sentiment — Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Size:** ~14,000 tweets
- **Airlines:** United, Delta, American, Southwest, US Airways, Virgin America
- **Labels:** positive, negative, neutral

---

## 🔮 Future Improvements

- [ ] Add BERT / transformer-based model for higher accuracy
- [ ] Deploy on Streamlit Cloud for public access
- [ ] Add word cloud visualization per sentiment
- [ ] Expand to other industries beyond airlines

---

## 👩‍💻 Author

**Prachi Singh**  
B.Tech CSE (Data Science) — Dr. A.P.J. Abdul Kalam Technical University  
[LinkedIn](https://linkedin.com/in/prachisinghshishodiya) · [GitHub](https://github.com/Prachi0216-tech)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
