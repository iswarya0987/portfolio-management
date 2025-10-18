
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Portfolio Management App", layout="wide")

st.title("📊 Smart Portfolio Management App")
st.markdown("Upload your stock portfolio CSV file to view insights, metrics, and risk analysis.")

st.sidebar.header("Upload Your Portfolio Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # --- Smart column detection ---
    df.columns = [c.strip() for c in df.columns]
    date_col = [c for c in df.columns if 'date' in c.lower()]
    price_col = [c for c in df.columns if any(x in c.lower() for x in ['ltp', 'close', 'price', 'adj close'])]

    if date_col and price_col:
        df.rename(columns={date_col[0]: 'Date', price_col[0]: 'ltp'}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df = df.dropna(subset=["Date", "ltp"])
        df = df.sort_values("Date")
        df["Returns"] = df["ltp"].pct_change()

        st.write("### 📈 Descriptive Statistics")
        st.dataframe(df["Returns"].describe().to_frame().T)

        # Portfolio metrics
        mean_return = df["Returns"].mean()
        volatility = df["Returns"].std()
        risk_free_rate = 0.05 / 252  # daily risk-free rate
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility != 0 else np.nan
        beta = np.cov(df["Returns"].dropna(), df["Returns"].dropna())[0][1] / np.var(df["Returns"].dropna()) if np.var(df["Returns"].dropna()) != 0 else np.nan
        jensen_alpha = mean_return - (risk_free_rate + beta * (mean_return - risk_free_rate)) if not np.isnan(beta) else np.nan
        treynor_ratio = (mean_return - risk_free_rate) / beta if beta not in [0, np.nan] else np.nan

        st.write("### ⚙️ Portfolio Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Daily Return", f"{mean_return:.5f}")
        col2.metric("Volatility", f"{volatility:.5f}")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        col4, col5 = st.columns(2)
        col4.metric("Jensen's Alpha", f"{jensen_alpha:.5f}")
        col5.metric("Treynor Ratio", f"{treynor_ratio:.5f}")

        # Graphical Analysis
        st.write("### 📊 Portfolio Return Trend")
        st.line_chart(df.set_index("Date")["ltp"])

        st.write("### 📉 Rolling Mean (+3 Days)")
        df["Rolling_Mean"] = df["ltp"].rolling(window=3).mean()
        st.line_chart(df.set_index("Date")[["ltp", "Rolling_Mean"]])

        # Suggestions
        st.write("### 💡 Summary & Suggestions")
        if sharpe_ratio > 1:
            st.success("✅ Great! Your portfolio offers strong risk-adjusted returns.")
        elif sharpe_ratio > 0.5:
            st.info("⚖️ Decent performance. Consider diversification for better stability.")
        else:
            st.warning("⚠️ Portfolio risk-adjusted returns are weak. Optimize asset allocation.")
    else:
        st.error("❌ Could not detect 'Date' or 'LTP/Close' columns. Please check your file headers.")
else:
    st.info("👆 Upload a CSV file to begin analysis.")
