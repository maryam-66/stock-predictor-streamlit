import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="ربات پیش‌بینی سهام", page_icon="📈")

st.title("🤖 ربات هوش مصنوعی پیش‌بینی سهام")
st.markdown("### یک ابزار ساده برای تمرین یادگیری ماشین در بازار سهام")

# انتخاب سهام
col1, col2 = st.columns(2)

with col1:
    symbol = st.selectbox(
        "📌 نماد سهام را انتخاب کنید:",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    )

with col2:
    period = st.selectbox(
        "⏳ مدت داده‌ها:",
        ["3mo", "6mo", "1y"]
    )

if st.button("🚀 شروع پیش‌بینی"):
    with st.spinner("📥 در حال دریافت داده‌ها..."):
        df = yf.download(symbol, period=period)
        
        if df.empty:
            st.error("❌ خطا در دریافت داده‌ها")
        else:
            df['MA5'] = df['Close'].rolling(5).mean()
            df['PriceChange'] = df['Close'].pct_change()
            df['NextDayPrice'] = df['Close'].shift(-1)
            df.dropna(inplace=True)
            
            X = df[['Close', 'MA5', 'PriceChange']]
            y = df['NextDayPrice']
            
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            st.success("✅ پیش‌بینی با موفقیت انجام شد!")
            
            # نمودار قیمت
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_test, name="قیمت واقعی", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions, name="پیش‌بینی", line=dict(color="red", dash="dash")))
            fig.update_layout(title=f"📈 پیش‌بینی قیمت {symbol}", xaxis_title="تاریخ", yaxis_title="قیمت ($)")
            st.plotly_chart(fig, use_container_width=True)

            # محاسبه دقت جهت قیمت
            dir_true = (y_test > y_test.shift(1)).astype(int)[1:]
            dir_pred = (predictions[1:] > y_test[:-1].values).astype(int)
            accuracy = (dir_true == dir_pred).mean() * 100
            
            # آمار
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 دقت پیش‌بینی", f"{accuracy:.1f}%")
            col2.metric("💰 آخرین قیمت", f"${y_test.iloc[-1]:.2f}")
            pred_tomorrow = predictions[-1]
            change = ((pred_tomorrow - y_test.iloc[-1]) / y_test.iloc[-1]) * 100
            col3.metric("🔮 پیش‌بینی فردا", f"${pred_tomorrow:.2f}", f"{change:+.1f}%")

            # تشویقی
            if accuracy > 60:
                st.success("🏆 عالی! دقت بالای ۶۰٪ گرفتی!")
                st.balloons()
            elif accuracy > 55:
                st.info("🥈 دقت خوبه، ادامه بده!")
            elif accuracy > 50:
                st.warning("🥉 شروع خوبی داشتی، بهتر هم می‌تونی!")
            else:
                st.error("💪 نگران نباش! مدلت رو بهتر کن!")

# راهنما
st.markdown("---")
st.markdown("### 📘 راهنمای استفاده:")
st.markdown("""
1. سهام موردنظر را انتخاب کنید  
2. مدت زمان را مشخص کنید  
3. دکمه «شروع پیش‌بینی» را بزنید  
4. دقت، نمودار و تحلیل را مشاهده کنید  

> ⚠️ توجه: این فقط یک تمرین آموزشی است. برای سرمایه‌گذاری واقعی قابل اتکا نیست.
""")
