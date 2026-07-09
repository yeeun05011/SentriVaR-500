import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from risk_engine import (
    fetch_prices, fetch_macro, calculate_risk_metrics,
    detect_regime, REGIME_LABELS, fetch_news, keyword_sentiment,
    correlation_risk, copula_risk_amplifier, calculate_combined_risk_score,
    get_idiosyncratic_risk, dynamic_allocation
)

st.set_page_config(page_title="SentriVaR-500", layout="wide")

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

st.title("SentriVaR-500")
st.caption("Adaptive multi-signal portfolio risk system with regime-aware weighting")

# ─────────────────────────
# Sidebar — user input
# ─────────────────────────
st.sidebar.header("Portfolio input")
tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL, JPM"
)
run_button = st.sidebar.button("Run analysis", type="primary")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not run_button:
    st.info("Enter tickers in the sidebar and click **Run analysis** to begin.")
    st.stop()

if len(tickers) < 2:
    st.error("Please enter at least 2 tickers for portfolio-level analysis.")
    st.stop()

# ─────────────────────────
# Step 1 — Fetch data
# ─────────────────────────
with st.spinner("Fetching price and macro data..."):
    prices = fetch_prices(tickers, start_date="2020-01-01")
    macro = fetch_macro(start_date="2020-01-01")
    returns = prices.pct_change().dropna()

st.success(f"Data loaded: {prices.shape[0]} trading days, {len(tickers)} tickers")

# ─────────────────────────
# Step 2 — Risk metrics per ticker
# ─────────────────────────
st.header("1. Risk metrics")

metrics_rows = []
for t in tickers:
    m = calculate_risk_metrics(returns[t])
    m["Ticker"] = t
    metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("Ticker")
st.dataframe(metrics_df, use_container_width=True)

# ─────────────────────────
# Step 3 — Regime detection
# ─────────────────────────
st.header("2. Market regime detection")

with st.spinner("Detecting market regime (HMM)..."):
    reference_ticker = tickers[0]
    regime_series = detect_regime(returns, macro, reference_ticker)
    current_regime = int(regime_series.iloc[-1])
    regime_label = REGIME_LABELS[current_regime]

regime_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
st.metric("Current regime", f"{regime_colors[current_regime]} {regime_label}")

regime_counts = regime_series.value_counts().sort_index()
regime_pct = (regime_counts / regime_counts.sum() * 100).round(1)
st.write("Regime distribution over the loaded period:")
st.bar_chart(regime_pct.rename(index=REGIME_LABELS))


# ─────────────────────────
# Step 4 — News sentiment (keyword-based)
# ─────────────────────────
st.header("3. News sentiment")

if not NEWSAPI_KEY:
    st.warning("NEWSAPI_KEY not found in .env — skipping live news sentiment.")
    sentiment_scores = {t: 0.0 for t in tickers}
else:
    sentiment_scores = {}
    with st.spinner("Fetching news headlines..."):
        for t in tickers:
            headlines = fetch_news(f"{t} stock", NEWSAPI_KEY, days_back=14)
            sentiment_scores[t] = keyword_sentiment(headlines)

    sentiment_df = pd.DataFrame.from_dict(
        sentiment_scores, orient="index", columns=["Sentiment score"]
    )
    st.dataframe(sentiment_df, use_container_width=True)

    # ─────────────────────────
# Step 5 — Combined risk score + Copula amplification
# ─────────────────────────
st.header("4. Combined risk score")

current_vix = macro["VIX"].iloc[-1]
current_spread = macro["Spread"].iloc[-1]
avg_sentiment = np.mean(list(sentiment_scores.values()))
port_vol = returns[tickers].mean(axis=1).iloc[-20:].std()

base_risk_score = calculate_combined_risk_score(
    avg_sentiment, current_vix, port_vol, current_spread, current_regime
)

corr_risk = correlation_risk(returns[tickers])
copula_score = copula_risk_amplifier(base_risk_score, current_regime, current_vix)

col1, col2, col3 = st.columns(3)
col1.metric("Base risk score", f"{base_risk_score:.3f}")
col2.metric("Copula-amplified score", f"{copula_score:.3f}",
            delta=f"{copula_score - base_risk_score:+.3f}")
col3.metric("Sector correlation", f"{corr_risk:.3f}")

if copula_score >= 0.5:
    st.error(f"High risk alert — regime: {regime_label}")
elif copula_score >= 0.3:
    st.warning(f"Caution — regime: {regime_label}")
else:
    st.success(f"Stable — regime: {regime_label}")


# ─────────────────────────
# Step 6 — Idiosyncratic risk + dynamic allocation
# ─────────────────────────
st.header("5. Dynamic asset allocation")

with st.spinner("Fetching idiosyncratic risk (analyst ratings, earnings, insider activity)..."):
    idiosyncratic_risk = {t: get_idiosyncratic_risk(t) for t in tickers}

combined_risk, weights = dynamic_allocation(copula_score, idiosyncratic_risk, current_regime)

risk_alloc_df = pd.DataFrame({
    "Idiosyncratic risk": idiosyncratic_risk,
    "Combined risk": combined_risk,
})
st.dataframe(risk_alloc_df, use_container_width=True)

st.subheader("Recommended allocation")
weights_df = pd.Series(weights, name="Weight").sort_values(ascending=False)
st.bar_chart(weights_df)

for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
    st.write(f"**{asset}**: {w*100:.1f}%")