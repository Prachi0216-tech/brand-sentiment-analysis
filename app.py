import streamlit as st
import pandas as pd
import plotly.express as px
from sentiment import load_data, analyze_sentiment, predict_review

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brand Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Title & description
# ---------------------------------------------------------------------------
st.title("💬 Brand Sentiment Analysis")
st.write(
    "Analyze airline customer reviews to understand sentiment trends "
    "and identify the most common complaints."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load data — with a safety check so the app never crashes on a missing file
# ---------------------------------------------------------------------------
df_raw = load_data("data/tweets.csv")

if df_raw.empty:
    st.error("⚠️ Data file not found or is empty. Please check `data/tweets.csv`.")
    st.stop()  # Stops execution cleanly — nothing below runs

# ---------------------------------------------------------------------------
# Sidebar filters — cleaner than inline selectbox for dashboards
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
airlines = sorted(df_raw["airline"].unique())
selected_airline = st.sidebar.selectbox("Select Airline", airlines)

# ---------------------------------------------------------------------------
# Filter & analyze — use separate variable names to keep raw data intact
# ---------------------------------------------------------------------------
df_filtered = df_raw[df_raw["airline"] == selected_airline].copy()

with st.spinner("Analyzing sentiment..."):
    df_analyzed = analyze_sentiment(df_filtered)

# ---------------------------------------------------------------------------
# Overall insight — key metrics at the top
# ---------------------------------------------------------------------------
st.subheader(f"Overall Insight — {selected_airline}")

total_reviews = len(df_analyzed)
negative_count = (df_analyzed["predicted_sentiment"] == "Negative").sum()
positive_count = (df_analyzed["predicted_sentiment"] == "Positive").sum()
neutral_count  = (df_analyzed["predicted_sentiment"] == "Neutral").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews",    total_reviews)
col2.metric("Positive",         positive_count)
col3.metric("Neutral",          neutral_count)
col4.metric("Negative",         negative_count)

if negative_count > total_reviews * 0.3:
    st.warning("⚠️ Observation: A noticeable number of customers are dissatisfied.")
else:
    st.success("✅ Observation: Most customers are satisfied or neutral.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Common problems — count keyword occurrences in a temporary series
# (not added as a column so it never leaks into the displayed table)
# ---------------------------------------------------------------------------
st.subheader("Common Problems Found")

text_lower = df_analyzed["text"].str.lower()  # temporary series, not a column

delay_count   = text_lower.str.contains("delay|delayed",           regex=True).sum()
cancel_count  = text_lower.str.contains("cancel|cancelled",        regex=True).sum()
service_count = text_lower.str.contains("service|staff|support",   regex=True).sum()
refund_count  = text_lower.str.contains("refund",                  regex=True).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Delay Issues",        delay_count)
c2.metric("Cancellation Issues", cancel_count)
c3.metric("Service Complaints",  service_count)
c4.metric("Refund Issues",       refund_count)

st.markdown("---")

# ---------------------------------------------------------------------------
# Sentiment distribution chart — with meaningful colors and a title
# ---------------------------------------------------------------------------
st.subheader("Sentiment Distribution")

sentiment_counts = df_analyzed["predicted_sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

# Map sentiments to intuitive colors
color_map = {
    "Positive": "#2ecc71",  # green
    "Negative": "#e74c3c",  # red
    "Neutral":  "#95a5a6",  # grey
}

fig = px.pie(
    sentiment_counts,
    values="Count",
    names="Sentiment",
    title=f"Sentiment Breakdown — {selected_airline}",
    color="Sentiment",
    color_discrete_map=color_map,
    hole=0.3,  # donut style — looks more modern than a full pie
)
fig.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Review data table — only show relevant columns, no temp columns
# ---------------------------------------------------------------------------
with st.expander("📋 View Review Data"):
    st.dataframe(
        df_analyzed[["airline", "text", "predicted_sentiment"]],
        use_container_width=True
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Live review prediction — with color-coded result
# ---------------------------------------------------------------------------
st.subheader("🔍 Live Review Prediction")
st.write("Type any customer review below and get an instant sentiment prediction.")

user_input = st.text_area("Enter a customer review", placeholder="e.g. The flight was delayed for 3 hours and no one helped us...")

if st.button("Analyze Review"):
    if user_input.strip():
        result = predict_review(user_input)

        # Color-coded sentiment result
        color_map_result = {
            "Positive": "green",
            "Negative": "red",
            "Neutral":  "grey"
        }
        color = color_map_result.get(result, "grey")
        st.markdown(f"**Predicted Sentiment:** :{color}[{result}]")

        # Issue detection for negative reviews
        if result == "Negative":
            text_lower_input = user_input.lower()
            if "delay" in text_lower_input:
                issue = "Flight delay"
            elif "cancel" in text_lower_input:
                issue = "Cancellation"
            elif "service" in text_lower_input or "staff" in text_lower_input:
                issue = "Service quality"
            elif "refund" in text_lower_input:
                issue = "Refund related problem"
            else:
                issue = "General dissatisfaction"
            st.markdown(f"**Possible Issue:** {issue}")
    else:
        st.warning("Please enter some text before clicking Analyze.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Built by Prachi Singh · Brand Sentiment Analysis · Python & Streamlit")
