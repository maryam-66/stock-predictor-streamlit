import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="ربات پیش‌بینی سهام", page_icon="📈", layout="wide")

st.markdown("<h1 style='text-align: center;'>🤖 ربات پیش‌بینی قیمت سهام با هوش مصنوعی</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar with filters
with st.sidebar:
    st.title("⚙️ تنظیمات")
    model_choice = st.selectbox("مدل پیش‌بینی:", ["Linear Regression", "Random Forest"])
    symbol = st.selectbox("📌 نماد سهام:", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    period = st.selectbox("⏳ مدت داده‌ها:", ["3mo", "6mo", "1y"])
    show_volume = st.checkbox("📊 نمایش حجم معامله", value=False)

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

            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Charts
            st.subheader(f"📈 نمودار قیمت واقعی و پیش‌بینی‌شده ({symbol})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_test, name="قیمت واقعی", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions, name="پیش‌بینی", line=dict(color="red", dash="dash")))
            if show_volume:
                fig.add_trace(go.Bar(x=df.index[-len(y_test):], y=df['Volume'][-len(y_test):], name="حجم", yaxis="y2", marker_color='rgba(128,128,128,0.3)'))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='حجم'), barmode='overlay')
            fig.update_layout(xaxis_title="تاریخ", yaxis_title="قیمت ($)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            direction_true = (y_test > y_test.shift(1)).astype(int)[1:]
            direction_pred = (predictions[1:] > y_test[:-1].values).astype(int)
            accuracy = (direction_true == direction_pred).mean() * 100

            latest_price = y_test.iloc[-1]
            tomorrow_pred = predictions[-1]
            change = ((tomorrow_pred - latest_price) / latest_price) * 100

            st.markdown("### 🎯 شاخص‌های عملکرد:")
            col1, col2, col3 = st.columns(3)
            col1.metric("دقت پیش‌بینی", f"{accuracy:.1f}٪")
            col2.metric("آخرین قیمت", f"${latest_price:.2f}")
            col3.metric("پیش‌بینی فردا", f"${tomorrow_pred:.2f}", f"{change:+.1f}%")

            # جدول گزارش کامل
            st.markdown("### 📋 جدول گزارش کامل:")
            report_df = pd.DataFrame({
                "تاریخ": y_test.index,
                "قیمت واقعی": y_test.values,
                "قیمت پیش‌بینی‌شده": predictions,
                "خطای پیش‌بینی ($)": (y_test - predictions).values,
                "جهت واقعی": (y_test > y_test.shift(1)).astype(int).values,
                "جهت پیش‌بینی": (predictions > y_test.shift(1)).astype(int)
            }).dropna()

            st.dataframe(report_df, use_container_width=True)

            # دکمه‌های دانلود
            csv = report_df.to_csv(index=False).encode('utf-8')
            excel_file = BytesIO()
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, index=False, sheet_name="Report")
                writer.save()
            excel_data = excel_file.getvalue()

            st.download_button("📥 دانلود CSV", csv, file_name="stock_report.csv", mime="text/csv")
            st.download_button("📥 دانلود Excel", excel_data, file_name="stock_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if accuracy > 60:
                st.success("🏆 عالی! دقت مدل بالاست.")
            elif accuracy > 55:
                st.info("🥈 مدل قابل قبولی است.")
            elif accuracy > 50:
                st.warning("🥉 مدل متوسط، قابل بهبود است.")
            else:
                st.error("💡 نیاز به بهبود مدل یا ویژگی‌ها.")
