# risk_engine.py
# Core calculation logic for SentriVaR-500, reusable across the Streamlit app

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from hmmlearn import hmm
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
import requests


# 1. Data collection

def fetch_prices(tickers, start_date="2020-01-01"):
    """Fetch adjusted close prices for the given tickers, skipping invalid ones."""
    valid_data = {}

    for t in tickers:
        try:
            raw = yf.download(t, start=start_date, auto_adjust=True, progress=False)

            if raw.empty:
                continue

            # Handle both flat and MultiIndex column structures
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][t] if t in raw["Close"].columns else raw["Close"].iloc[:, 0]
            else:
                close = raw["Close"]

            close = close.dropna()
            if len(close) > 0:
                valid_data[t] = close

        except Exception:
            continue

    if not valid_data:
        return pd.DataFrame()

    result = pd.concat(valid_data, axis=1)
    result.columns = list(valid_data.keys())
    return result.dropna()


def fetch_macro(start_date="2020-01-01"):
    """Fetch VIX and the 10Y-2Y Treasury spread from FRED."""
    vix = pdr.get_data_fred("VIXCLS", start=start_date)
    spread = pdr.get_data_fred("T10Y2Y", start=start_date)
    macro = pd.concat([vix, spread], axis=1, sort=False)
    macro.columns = ["VIX", "Spread"]
    return macro.dropna()


# 2. Risk metrics (VaR, CVaR, Sharpe, MDD)

def calculate_risk_metrics(return_series, confidence=0.95):
    """Calculate VaR, CVaR, Sharpe ratio, and Max Drawdown for a single asset."""
    VaR = return_series.quantile(1 - confidence)
    CVaR = return_series[return_series <= VaR].mean()
    sharpe = return_series.mean() / return_series.std() * np.sqrt(252)

    cumulative = (1 + return_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    mdd = drawdown.min()

    return {
        "VaR": round(VaR, 4),
        "CVaR": round(CVaR, 4),
        "Sharpe": round(sharpe, 4),
        "MDD": round(mdd, 4)
    }

# 3. HMM regime detection

REGIME_LABELS = {0: "Normal", 1: "Elevated", 2: "Crisis"}

def detect_regime(returns, macro, ticker, n_components=3, smooth_window=30):
    """
    Detect market regime (Normal / Elevated / Crisis) using a Gaussian HMM
    trained on returns, rolling volatility, and VIX + Treasury spread.
    """
    hmm_data = returns[[ticker]].copy()
    hmm_data["volatility"] = returns[ticker].rolling(20).std()
    hmm_data = hmm_data.join(macro[["VIX", "Spread"]]).dropna()
    dates = hmm_data.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hmm_data.values)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=300,
        random_state=42,
        init_params="mc"
    )
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([
        [0.95, 0.04, 0.01],
        [0.05, 0.90, 0.05],
        [0.02, 0.08, 0.90],
    ])
    model.fit(X_scaled)
    states = model.predict(X_scaled)

    smoothed = uniform_filter1d(states.astype(float), size=smooth_window)
    smoothed = np.round(smoothed).astype(int)
    state_series = pd.Series(smoothed, index=dates)

    # Auto-label regimes by mean VIX (lowest = Normal, highest = Crisis)
    vix_aligned = macro["VIX"].reindex(dates)
    state_vix_means = {s: vix_aligned[state_series == s].mean() for s in range(n_components)}
    sorted_states = sorted(state_vix_means, key=state_vix_means.get)
    label_map = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
    state_series = state_series.map(label_map)

    return state_series

# 4. News collection and keyword-based sentiment

POSITIVE_WORDS = [
    "beat", "beats", "surge", "surges", "rally", "record", "growth",
    "upgrade", "outperform", "strong", "bullish", "gain", "gains",
    "profit", "soar", "soars", "rebound", "optimism", "boost"
]

NEGATIVE_WORDS = [
    "miss", "misses", "plunge", "plunges", "crash", "downgrade",
    "underperform", "weak", "bearish", "loss", "losses", "recession",
    "slump", "warn", "warns", "fear", "fears", "sell-off", "selloff",
    "decline", "cut", "cuts", "tumble", "tumbles"
]

def fetch_news(query, api_key, days_back=14, page_size=30):
    """Fetch recent news headlines from NewsAPI for a given query."""
    from datetime import datetime, timedelta

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data.get("status") != "ok":
        return []

    return [article["title"] for article in data.get("articles", [])]


def keyword_sentiment(headlines):
    """
    Fast keyword-based sentiment score in [-1, 1].
    Positive words increase the score, negative words decrease it.
    """
    if not headlines:
        return 0.0

    scores = []
    for headline in headlines:
        text = headline.lower()
        pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
        neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)
        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            scores.append(0.0)
        else:
            scores.append((pos_hits - neg_hits) / total_hits)

    return round(np.mean(scores), 4)

# 5. Sector correlation risk

def correlation_risk(returns_df, window=30):
    """
    Rolling average pairwise correlation across assets.
    High values indicate assets moving together (contagion signal).
    """
    if len(returns_df) < window:
        return 0.5

    recent = returns_df.tail(window)
    corr_matrix = recent.corr()
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr_matrix.values[mask].mean()
    return round(float(avg_corr), 4)


# 6. Copula-inspired non-linear risk amplification

def copula_risk_amplifier(risk_score, regime, vix, vix_mean=20.0):
    """
    Amplify risk non-linearly depending on regime.
    Normal: unchanged. Elevated: mild amplification.
    Crisis: exponential tail-risk amplification.
    """
    if regime == 0:
        return risk_score
    elif regime == 1:
        vix_factor = max(1.0, vix / vix_mean)
        return risk_score * (1 + 0.3 * (vix_factor - 1))
    else:
        vix_factor = max(1.0, vix / vix_mean)
        tail_amplification = vix_factor ** 1.5
        return min(1.0, risk_score * tail_amplification)


# 7. Combined multi-signal risk score

def calculate_combined_risk_score(sentiment_score, vix, port_vol, spread, regime):
    """
    Combine sentiment, VIX, portfolio volatility, and spread into a single
    risk score, with regime-dependent weighting.
    """
    vix_norm = min(vix / 80, 1.0)
    sentiment_norm = (1 - sentiment_score) / 2
    spread_risk = 1 if spread < 0 else 0

    if regime == 0:
        weights = {"sentiment": 0.40, "vix": 0.25, "vol": 0.25, "spread": 0.10}
    elif regime == 1:
        weights = {"sentiment": 0.25, "vix": 0.35, "vol": 0.25, "spread": 0.15}
    else:
        weights = {"sentiment": 0.15, "vix": 0.45, "vol": 0.30, "spread": 0.10}

    score = (
        weights["sentiment"] * sentiment_norm +
        weights["vix"] * vix_norm +
        weights["vol"] * abs(port_vol) * 10 +
        weights["spread"] * spread_risk
    )
    return round(min(score, 1.0), 4)


# 8. Idiosyncratic risk (analyst ratings, earnings, insider activity)

def get_analyst_risk(ticker):
    """Analyst rating risk score based on aggregate buy/sell recommendations."""
    try:
        stock = yf.Ticker(ticker)
        rec = stock.recommendations
        if rec is None or rec.empty:
            return 0.5

        latest = rec[rec["period"] == "0m"].iloc[0]
        total = latest["strongBuy"] + latest["buy"] + latest["hold"] + latest["sell"] + latest["strongSell"]
        if total == 0:
            return 0.5

        weighted = (
            latest["strongBuy"] * 0.0 +
            latest["buy"] * 0.25 +
            latest["hold"] * 0.5 +
            latest["sell"] * 0.75 +
            latest["strongSell"] * 1.0
        ) / total
        return round(weighted, 4)
    except Exception:
        return 0.5


def get_earnings_risk(ticker):
    """Earnings surprise risk score: negative surprises raise risk."""
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            return 0.5

        recent = earnings.dropna(subset=["Surprise(%)"]).head(4)
        if recent.empty:
            return 0.5

        avg_surprise = recent["Surprise(%)"].mean()
        score = max(0, min(1, 0.5 - avg_surprise / 100))
        return round(score, 4)
    except Exception:
        return 0.5


def get_insider_risk(ticker):
    """Insider-selling acceleration risk: recent vs. older sell ratio."""
    try:
        stock = yf.Ticker(ticker)
        insider = stock.insider_transactions
        if insider is None or insider.empty:
            return 0.5

        insider["Start Date"] = pd.to_datetime(insider["Start Date"])
        insider["is_sale"] = insider["Text"].str.contains("Sale", case=False, na=False)

        cutoff = pd.Timestamp.today() - pd.DateOffset(months=3)
        recent = insider[insider["Start Date"] >= cutoff]
        older = insider[insider["Start Date"] < cutoff]

        recent_sale = recent[recent["is_sale"]]["Value"].fillna(0).sum()
        recent_total = recent["Value"].fillna(0).sum()
        older_sale = older[older["is_sale"]]["Value"].fillna(0).sum()
        older_total = older["Value"].fillna(0).sum()

        recent_ratio = recent_sale / recent_total if recent_total > 0 else 0.5
        older_ratio = older_sale / older_total if older_total > 0 else 0.5

        delta = recent_ratio - older_ratio
        score = max(0, min(1, 0.5 + delta * 0.5))
        return round(score, 4)
    except Exception:
        return 0.5


def get_idiosyncratic_risk(ticker):
    """Combine analyst, earnings, and insider signals into one score."""
    analyst = get_analyst_risk(ticker)
    earnings = get_earnings_risk(ticker)
    insider = get_insider_risk(ticker)
    combined = earnings * 0.40 + analyst * 0.30 + insider * 0.30
    return round(combined, 4)


# 9. Dynamic asset allocation

def dynamic_allocation(portfolio_risk, idiosyncratic_risk, regime):
    """
    Blend portfolio-level and idiosyncratic risk per ticker, then compute
    inverse-risk weights with a regime-dependent cash buffer.
    """
    tickers = list(idiosyncratic_risk.keys())

    combined_risk = {
        t: round(portfolio_risk * 0.5 + idiosyncratic_risk[t] * 0.5, 4)
        for t in tickers
    }

    inv_risk = {t: 1 / (r + 0.01) for t, r in combined_risk.items()}
    total_inv = sum(inv_risk.values())
    base_weights = {t: v / total_inv for t, v in inv_risk.items()}

    if regime == 0:
        cash_ratio = max(0, portfolio_risk * 0.2)
    elif regime == 1:
        cash_ratio = max(0.1, portfolio_risk * 0.4)
    else:
        cash_ratio = max(0.3, portfolio_risk * 0.6)

    equity_ratio = 1 - cash_ratio
    final_weights = {t: round(w * equity_ratio, 4) for t, w in base_weights.items()}
    final_weights["CASH"] = round(cash_ratio, 4)

    return combined_risk, final_weights

# 10. Risk score explainability (exact contribution decomposition)

def explain_risk_score(sentiment_score, vix, port_vol, spread, regime):
    """
    Decompose the combined risk score into per-signal contributions.
    Since calculate_combined_risk_score is a weighted linear combination,
    this decomposition is exact (equivalent to Shapley values for a
    linear/additive model) rather than an approximation.
    """
    vix_norm = min(vix / 80, 1.0)
    sentiment_norm = (1 - sentiment_score) / 2
    spread_risk = 1 if spread < 0 else 0
    vol_term = abs(port_vol) * 10

    if regime == 0:
        weights = {"Sentiment": 0.40, "VIX": 0.25, "Volatility": 0.25, "Spread": 0.10}
    elif regime == 1:
        weights = {"Sentiment": 0.25, "VIX": 0.35, "Volatility": 0.25, "Spread": 0.15}
    else:
        weights = {"Sentiment": 0.15, "VIX": 0.45, "Volatility": 0.30, "Spread": 0.10}

    raw_values = {
        "Sentiment": sentiment_norm,
        "VIX": vix_norm,
        "Volatility": vol_term,
        "Spread": spread_risk,
    }

    contributions = {
        signal: round(weights[signal] * raw_values[signal], 4)
        for signal in weights
    }

    return contributions