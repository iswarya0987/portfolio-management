import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Analysis App", layout="wide")

st.title("📊 Portfolio Management and Risk Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)

    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to detect Date and LTP columns
    date_col = next((col for col in df.columns if "date" in col), None)
    ltp_col = next((col for col in df.columns if "ltp" in col or "close" in col), None)

    if not date_col or not ltp_col:
        st.error("❌ 'Date' or 'LTP/CLOSE' columns not found in uploaded file. Please include them.")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)

        # Convert numeric columns safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        df = df.dropna(subset=[ltp_col])
        df[ltp_col] = pd.to_numeric(df[ltp_col], errors='coerce')

        st.success("✅ Data processed successfully!")
        st.write("### Descriptive Statistics")
        st.write(df.describe())

        # Calculate daily returns
        df['returns'] = df[ltp_col].pct_change()
        mean_return = df['returns'].mean()
        std_dev = df['returns'].std()
        sharpe_ratio = mean_return / std_dev if std_dev != 0 else np.nan
        risk_free_rate = 0.05 / 252
        excess_return = df['returns'] - risk_free_rate
        beta = np.cov(df['returns'].dropna(), excess_return.dropna())[0, 1] / np.var(excess_return.dropna())
        treynor_ratio = mean_return / beta if beta != 0 else np.nan
        jensen_alpha = mean_return - (risk_free_rate + beta * (mean_return - risk_free_rate))

        st.write("### 📈 Portfolio Metrics")
        st.write({
            "Mean Daily Return": round(mean_return, 6),
            "Standard Deviation": round(std_dev, 6),
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Treynor Ratio": round(treynor_ratio, 3),
            "Jensen’s Alpha": round(jensen_alpha, 3),
            "Portfolio Beta": round(beta, 3)
        })

        st.write("### 📊 Visual Analysis")
        fig, ax = plt.subplots()
        ax.plot(df[date_col], df[ltp_col], label="LTP", color="blue")
        ax.set_title("LTP Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("LTP")
        st.pyplot(fig)

        st.write("### Suggestions")
        if sharpe_ratio > 1:
            st.success("Good risk-adjusted returns. Portfolio is performing efficiently.")
        elif sharpe_ratio > 0.5:
            st.info("Moderate performance. Review diversification and rebalancing strategy.")
        else:
            st.warning("Low risk-adjusted return. Consider revising asset allocation.")

