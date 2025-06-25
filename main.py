from utils.fetch_data import fetch_stock_data
from utils.preprocess import add_sentiment_column
from models.price_predictor import prepare_features, train_model, evaluate_model

# Example headlines (replace with scraped data in production)
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

def main():
    # Fetch stock data
    df = fetch_stock_data("AAPL", start="2022-01-01", end="2022-06-01")
    headlines = mock_headlines * (len(df) // len(mock_headlines) + 1)
    headlines = headlines[:len(df)]  # Trim to length

    # Add sentiment
    df = add_sentiment_column(df, headlines)

    # Prepare features and train
    X, y = prepare_features(df)
    X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
    y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
