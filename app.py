import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import yfinance as yf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NIFTY 50 Analyzer", layout="wide")

# ---------------- IMPROVED UI CSS ----------------
st.markdown("""
<style>

/* Background */
body, .stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

/* Metric text */
[data-testid="stMetricLabel"] {
    color: #cbd5f5 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] {
    font-weight: 600;
}

/* Input labels */
label {
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}

/* Table container */
[data-testid="stDataFrame"] {
    background-color: #111827;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("NIFTY 50 Stock Analyzer")

# ---------------- SAFE FLOAT ----------------
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

# ---------------- MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ---------------- LOAD ----------------
df = pd.read_csv("nifty_10yrs.csv")
df['Date'] = pd.to_datetime(df['Date'])

with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

model = LSTMModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

companies = sorted(df['Company'].unique())
company = st.sidebar.selectbox("Select Company", companies)

# ---------------- LIVE DATA ----------------
symbol = company + ".NS"

try:
    live_data = yf.download(symbol, period="10d", interval="1d")
    live_data = live_data.dropna()
    latest = live_data.iloc[-1]
except:
    st.error("Unable to fetch live data")
    st.stop()

# ---------------- VALUES ----------------
close_val = safe_float(latest["Close"])
open_val = safe_float(latest["Open"])
high_val = safe_float(latest["High"])
low_val = safe_float(latest["Low"])
volume_val = safe_float(latest["Volume"])

prev_close = safe_float(live_data['Close'].iloc[-2]) if len(live_data) > 1 else close_val

change = close_val - prev_close
percent = (change / prev_close * 100) if prev_close != 0 else 0

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

delta_color = "normal" if change >= 0 else "inverse"

col1.metric("Current Price", f"{close_val:.2f}")
col2.metric("Change", f"{change:.2f}", f"{percent:.2f}%", delta_color=delta_color)
col3.metric("Volume", f"{int(volume_val):,}")

# ---------------- DARK TABLE ----------------
st.subheader("Recent Market Data")

table_df = live_data.tail(5).reset_index()
table_df = table_df[['Date','Open','High','Low','Close','Volume']]

# format values
table_df['Open'] = table_df['Open'].map(lambda x: f"{x:.2f}")
table_df['High'] = table_df['High'].map(lambda x: f"{x:.2f}")
table_df['Low'] = table_df['Low'].map(lambda x: f"{x:.2f}")
table_df['Close'] = table_df['Close'].map(lambda x: f"{x:.2f}")
table_df['Volume'] = table_df['Volume'].map(lambda x: f"{int(x):,}")

# style table
styled_table = table_df.style \
    .set_properties(**{
        'background-color': '#111827',
        'color': '#e5e7eb',
        'border-color': '#1f2937',
        'font-size': '14px'
    }) \
    .set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#020617'),
                ('color', '#60a5fa'),
                ('font-weight', '600'),
                ('border', '1px solid #1f2937')
            ]
        },
        {
            'selector': 'tbody tr:hover',
            'props': [('background-color', '#1e293b')]
        }
    ])

st.dataframe(styled_table, use_container_width=True)

# ---------------- INPUT ----------------
st.subheader("Manual Input")

col1, col2, col3, col4, col5 = st.columns(5)

open_p = col1.number_input("Open", value=open_val)
high_p = col2.number_input("High", value=high_val)
low_p = col3.number_input("Low", value=low_val)
close_p = col4.number_input("Close", value=close_val)
volume = col5.number_input("Volume", value=volume_val)

# ---------------- PREPARE ----------------
def prepare_input(df, scaler, company, today_input):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    company_df = df[df['Company'] == company].sort_values(by='Date')
    data = company_df[features].values

    data_scaled = scaler.transform(data)
    last_60 = data_scaled[-60:]

    today_scaled = scaler.transform(today_input)

    final_input = np.vstack([last_60, today_scaled])
    final_input = final_input[-60:]

    X_input = np.expand_dims(final_input, axis=0)
    return torch.tensor(X_input, dtype=torch.float32)

# ---------------- PREDICT ----------------
if st.button("Predict"):

    scaler = scalers[company]
    today = np.array([[open_p, high_p, low_p, close_p, volume]])

    X_input = prepare_input(df, scaler, company, today)

    with torch.no_grad():
        pred = model(X_input).item()

    dummy = np.zeros((1,5))
    dummy[0,3] = pred
    raw_price = scaler.inverse_transform(dummy)[0,3]

    # dynamic clamp
    volatility = abs(high_p - low_p)
    lower = close_p - volatility
    upper = close_p + volatility

    final_price = max(min(raw_price, upper), lower)
    change = ((final_price - close_p) / close_p) * 100 if close_p != 0 else 0

    st.subheader("Prediction")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Price", f"{final_price:.2f}")
    col2.metric("Expected Change %", f"{change:.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Model trained on historical data. Not financial advice.")