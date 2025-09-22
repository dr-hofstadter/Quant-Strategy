# app/phase1_moving_average.py
import streamlit as st
import yfinance as yf
import vectorbt as vbt
import pandas as pd

st.set_page_config(page_title="Phase 1 - Moving Average Strategy", layout="wide")

st.title("📈 Phase 1: Moving Average Crossover Backtest")

# Sidebar parameters
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2015-01-01"))
init_cash = st.sidebar.number_input("Initial Cash", value=100000, step=1000)
fast_window = st.sidebar.slider("Fast MA window", 5, 100, 20)
slow_window = st.sidebar.slider("Slow MA window", 10, 200, 50)
fees = st.sidebar.number_input("Fees (e.g. 0.0005 = 0.05%)", value=0.0005)
slippage = st.sidebar.number_input("Slippage (e.g. 0.0002 = 0.02%)", value=0.0002)

# Download data
st.write(f"Downloading {ticker} data…")
price = yf.download(ticker, start=start_date)['Close']

st.line_chart(price, height=200)

# Strategy
fast_ma = price.rolling(fast_window).mean()
slow_ma = price.rolling(slow_window).mean()
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma

# Backtest
portfolio = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=init_cash,
    fees=fees,
    slippage=slippage,
    freq='1D'
)

# Results
stats = portfolio.stats()
st.subheader("Performance Stats")
st.dataframe(stats.to_frame("Value"))

st.subheader("Equity Curve & Drawdowns")
st.plotly_chart(portfolio.plot(), use_container_width=True)

st.subheader("Rolling Sharpe (1Y)")
st.line_chart(portfolio.sharpe_ratio_rolling(window=252))
