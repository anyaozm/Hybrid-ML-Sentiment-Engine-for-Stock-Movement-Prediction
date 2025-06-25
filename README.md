# 🧠 Hybrid ML Sentiment Engine for Stock Movement Prediction

This project is a hybrid machine learning model that predicts short-term stock price movement using a combination of:
- **Technical analysis**
- **Financial news sentiment**
- **Historical price trends**

The goal is to provide a lightweight, explainable pipeline for real-time or batch prediction of stock movement using public data.

---

## 📦 Features

- 📈 Fetches historical stock data from Yahoo Finance (`yfinance`)
- 📰 Analyzes real-time financial news headlines via sentiment analysis (`nltk`, `VADER`)
- 🤖 Uses ensemble ML models (e.g. SVM, Logistic Regression, Random Forest) for binary price movement prediction
- 📊 Visualizes sentiment and prediction results using `matplotlib` and `streamlit`
- 💡 Designed to be modular and extensible for future deep learning, crypto data, or Reddit sentiment

---

## 🛠️ Installation

```bash
git clone https://github.com/redkoai/Hybrid-ML-Sentiment-Engine-for-Stock-Movement-Prediction.git
cd Hybrid-ML-Sentiment-Engine-for-Stock-Movement-Prediction

# (Recommended) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## 🚀 Usage
bash
Copy
Edit
# Run full pipeline in terminal
python3 main.py

# OR launch web interface
streamlit run streamlit_app.py
🧪 Example Output
Predicted: 📈 Price Up Tomorrow

Accuracy: ~78% on backtested 30-day window

Sentiment: 🟢 Positive (from news + Twitter)

Confidence Score: 0.81

## 📂 Project Structure
Hybrid-ML-Sentiment-Engine/
├── main.py                       # Main pipeline
├── streamlit_app.py             # Web UI
├── utils/
│   ├── fetch_data.py            # Pulls stock price history
│   ├── sentiment_analysis.py    # VADER-based NLP
│   ├── feature_engineering.py   # Adds technical indicators
│   └── model.py                 # Trains ML models
└── requirements.txt

## 🧠 Model Roadmap
 Classical ML: SVM, RF, Logistic Regression

 Deep Learning: LSTM / Transformer

 Reddit / X / News scraping layer

 Real-time WebSocket stock stream

 Backtesting framework with UI

## 👤 Author
Built by @anyaozmen
Open to collaborators, PRs, and new ideas.

