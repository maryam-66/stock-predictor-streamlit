import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="ربات تحلیل پیشرفته سهام", page_icon="📈", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 ربات تحلیل تکنیکال و پیش‌بینی قیمت سهام</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ تنظیمات")
    model_choice = st.selectbox("مدل پیش‌بینی:", ["Linear Regression", "Random Forest"])
    symbol = st.selectbox("📌 نماد سهام:", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    period = st.selectbox("⏳ دوره زمانی:", ["3mo", "6mo", "1y", "2y"])
    show_ema = st.checkbox("📈 نمایش EMA", True)
    show_rsi = st.checkbox("📉 نمایش RSI", True)
    show_macd = st.checkbox("📊 نمایش MACD", True)
    show_boll = st.checkbox("📏 نمایش Bollinger Bands", True)
    date_filter = st.checkbox("🗓️ فعال‌سازی فیلتر تاریخ")
    if date_filter:
        start_date = st.date_input("تاریخ شروع")
        end_date = st.date_input("تاریخ پایان")

if st.button("🚀 شروع تحلیل"):
    df = yf.download(symbol, period=period)

    if df.empty:
        st.error("❌ خطا در دریافت داده‌ها")
    else:
        df['MA5'] = df['Close'].rolling(5).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['PriceChange'] = df['Close'].pct_change()
        df['NextDayPrice'] = df['Close'].shift(-1)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = sma_20 + (2 * std_20)
        df['Bollinger_Lower'] = sma_20 - (2 * std_20)

        df.dropna(inplace=True)

        if date_filter:
            df = df.loc[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

        # Model
        X = df[['Close', 'MA5', 'PriceChange']]
        y = df['NextDayPrice']
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Chart: Price + Technicals
        st.subheader(f"📈 نمودار قیمت و شاخص‌ها برای {symbol}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_test, name="قیمت واقعی", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions, name="پیش‌بینی", line=dict(color="red", dash="dash")))

        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA 20", line=dict(color="orange", width=1)))

        if show_boll:
            fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Upper'], name="Bollinger Upper", line=dict(color="green", width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Lower'], name="Bollinger Lower", line=dict(color="green", width=1, dash="dot")))

        fig.update_layout(template="plotly_white", xaxis_title="تاریخ", yaxis_title="قیمت ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Chart: RSI
        if show_rsi:
            st.subheader("📉 نمودار RSI")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="purple")))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(template="plotly_white", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

        # Chart: MACD
        if show_macd:
            st.subheader("📊 MACD Histogram")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="black")))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color="red", dash="dot")))
            fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color="lightblue"))
            fig_macd.update_layout(template="plotly_white", yaxis_title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

        # Metrics
        direction_true = (y_test > y_test.shift(1)).astype(int)[1:]
        direction_pred = (predictions[1:] > y_test[:-1].values).astype(int)
        accuracy = (direction_true == direction_pred).mean() * 100
        latest_price = y_test.iloc[-1]
        tomorrow_pred = predictions[-1]
        change = ((tomorrow_pred - latest_price) / latest_price) * 100

        st.markdown("### 🎯 دقت و پیش‌بینی:")
        col1, col2, col3 = st.columns(3)
        col1.metric("دقت پیش‌بینی", f"{accuracy:.1f}٪")
        col2.metric("آخرین قیمت", f"${latest_price:.2f}")
        col3.metric("پیش‌بینی فردا", f"${tomorrow_pred:.2f}", f"{change:+.1f}%")
