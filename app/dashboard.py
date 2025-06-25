import streamlit as st
import pandas as pd
from utils.fetch_data import fetch_stock_data
from utils.preprocess import add_sentiment_column
from models.price_predictor import prepare_features, train_model

# Sample headlines
mock_headlines = [
    "Apple beats earnings expectations",
    "iPhone sales growth slows down",
    "Apple announces new product line",
    "Regulatory concerns affect Apple stock",
    "Apple reports strong services revenue",
    "Tim Cook announces AI integration in iOS",
    "Apple faces supply chain issues in China",
    "New iPhone 15 hits record preorders",
    "Analysts raise Apple price target",
    "Apple faces antitrust lawsuit in EU"
]

st.set_page_config(page_title="MarketMood Dashboard", layout="wide")
st.title("📈 MarketMood: Sentiment + Price Prediction Engine")

# Sidebar
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-06-01"))

# Fetch and process data
with st.spinner("Fetching stock data..."):
    df = fetch_stock_data(ticker, start=str(start_date), end=str(end_date))
    headlines = mock_headlines * (len(df) // len(mock_headlines) + 1)
    headlines = headlines[:len(df)]
    df = add_sentiment_column(df, headlines)

# Train and predict
X, y = prepare_features(df)
model = train_model(X, y)
df["Prediction"] = model.predict(X)

# Display data
st.subheader("Sentiment vs Price")
st.line_chart(df[["Close", "sentiment"]])

st.subheader("Prediction Outcomes")
st.write(df[["Close", "sentiment", "Prediction"]].tail(10))

st.success("Dashboard Ready!")
