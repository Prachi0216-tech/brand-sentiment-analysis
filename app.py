import streamlit as st
import pandas as pd
import plotly.express as px

from sentiment import load_data, analyze_sentiment, predict_review

# Page settings
st.set_page_config(
    page_title="Brand Sentiment Analysis",
    layout="wide"
)

# Title
st.title("Brand Sentiment Analysis")
st.write(
    "This application analyzes customer reviews to understand sentiment "
    "and identify common problems faced by customers."
)
st.markdown("---")

# Load data
df = load_data()

# Airline selection
airlines = df["airline"].unique()
selected_airline = st.selectbox("Select Airline", airlines)
df = df[df["airline"] == selected_airline]

# Sentiment analysis
df = analyze_sentiment(df)

# Overall insight
st.subheader("Overall Insight")

negative_count = (df["predicted_sentiment"] == "Negative").sum()
total_reviews = len(df)

st.write("Total reviews:", total_reviews)
st.write("Negative reviews:", negative_count)

if negative_count > total_reviews * 0.3:
    st.write("Observation: A noticeable number of customers are dissatisfied.")
else:
    st.write("Observation: Most customers are satisfied or neutral.")

# Common problems
st.subheader("Common Problems Found")

df["text_lower"] = df["text"].str.lower()

delay = df["text_lower"].str.contains("delay|delayed").sum()
cancel = df["text_lower"].str.contains("cancel|cancelled").sum()
service = df["text_lower"].str.contains("service|staff|support").sum()
refund = df["text_lower"].str.contains("refund").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Delay Issues", delay)
c2.metric("Cancellation Issues", cancel)
c3.metric("Service Complaints", service)
c4.metric("Refund Issues", refund)

# Sentiment distribution
st.subheader("Sentiment Distribution")

sentiment_counts = df["predicted_sentiment"].value_counts()

fig = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index
)
st.plotly_chart(fig, use_container_width=True)

# Review data
with st.expander("View Review Data"):
    st.dataframe(df[["airline", "text", "predicted_sentiment"]])

# Live review prediction
st.markdown("---")
st.subheader("Live Review Prediction")

user_input = st.text_area("Enter a customer review")

if st.button("Analyze"):
    if user_input.strip():
        result = predict_review(user_input)
        st.write("Predicted sentiment:", result)

        text_lower = user_input.lower()

        if result == "Negative":
            if "delay" in text_lower:
                st.write("Possible issue: Delay")
            elif "cancel" in text_lower:
                st.write("Possible issue: Cancellation")
            elif "service" in text_lower:
                st.write("Possible issue: Service quality")
            elif "refund" in text_lower:
                st.write("Possible issue: Refund related problem")
            else:
                st.write("Possible issue: General dissatisfaction")
    else:
        st.write("Please enter some text.")