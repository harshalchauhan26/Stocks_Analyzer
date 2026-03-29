# NIFTY 50 Stock Analyzer

# IMPORTANT
Link:https://stocksanalyzer-arum2p623mpbtxkd2facx9.streamlit.app/

To see model training and code kindly download stock.ipynb as a raw file and then view.

## Overview

NIFTY 50 Stock Analyzer is a machine learning powered web application that analyzes stock data and predicts the next closing price based on historical patterns. The application provides a clean, dark-themed fintech dashboard with live market data integration and user-driven inputs.

This project demonstrates the end-to-end pipeline of data preprocessing, model training, prediction, and deployment using Streamlit.

---

## Features

* Real-time stock data integration using Yahoo Finance
* LSTM-based deep learning model for price prediction
* Interactive dashboard with clean dark UI
* Company selection from NIFTY 50 stocks
* Manual input for Open, High, Low, Close, and Volume
* Dynamic prediction based on recent market volatility
* Styled data tables matching fintech dashboards

---

## Tech Stack

### Frontend

* Streamlit

### Backend / ML

* PyTorch (LSTM Model)
* NumPy
* Pandas

### Data Source

* yfinance (Yahoo Finance API)

### Visualization & UI

* Streamlit Components
* Custom CSS for styling

---

## Libraries Used

* streamlit
* pandas
* numpy
* torch
* scikit-learn
* yfinance
* pickle

---

## Project Structure

```
project/
│
├── app.py
├── model.pth
├── scalers.pkl
├── nifty_10yrs.csv
├── requirements.txt
└── README.md
```

---

## How It Works

1. Historical stock data is used to train an LSTM model.
2. Data is normalized using MinMaxScaler for each company.
3. The model learns patterns from sequences of past 60 days.
4. Live data is fetched using yfinance.
5. User inputs are combined with recent data to make predictions.
6. Output is adjusted using a volatility-based constraint for realistic results.

---

## Installation

Clone the repository:

```
git clone <your-repo-link>
cd <project-folder>
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

```
streamlit run app.py
```

---

## Usage

1. Select a company from the sidebar.
2. View recent market data.
3. Modify input values if needed.
4. Click on "Predict" to get the next closing price.

---

## Model Details

* Model Type: LSTM (Long Short-Term Memory)
* Input Features: Open, High, Low, Close, Volume
* Sequence Length: 60 days
* Output: Next day closing price

---

## Limitations

* The model is trained on historical data and does not account for real-world events.
* Predictions are based on patterns, not financial advice.
* Market volatility and external factors are not fully captured.

---

## Future Improvements

* Integration with real-time NSE APIs
* Candlestick chart visualization
* Multi-day forecasting
* Model improvement using returns instead of raw prices
* Deployment on cloud platforms

---

## Disclaimer

This project is for educational and research purposes only. It should not be used for financial decision-making.

---
