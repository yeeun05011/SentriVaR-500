# SentriVaR-500

**Adaptive multi-signal portfolio risk system with regime-aware weighting, idiosyncratic risk integration, and dynamic asset allocation.**

Traditional Value at Risk (VaR) is backward-looking: it measures risk from historical price data alone, which means it tends to *underestimate* danger right when markets are turning — the "tail risk" problem. SentriVaR-500 addresses this by fusing macro, market, and news signals into a single adaptive risk score whose weighting itself changes with the market regime, then validates the approach against three real market shocks from 2024–2026.

---

## What it does

1. Detects the current market regime (Normal / Elevated / Crisis) using a Hidden Markov Model trained on returns, rolling volatility, and VIX.
2. Splits real-time news into **systemic** (macro) and **idiosyncratic** (company-specific) sentiment using FinBERT, so a Fed headline doesn't get mistaken for an Apple-specific risk signal.
3. Combines VIX, sector correlation, and sentiment into a single risk score, with weights that shift depending on the detected regime (e.g. sentiment matters more in calm markets, VIX matters more in a crisis).
4. Applies a **Copula-inspired non-linear amplifier**: in a Crisis regime, signals compound each other instead of adding linearly — capturing the tail-risk behavior standard VaR misses.
5. Layers in **idiosyncratic, stock-specific risk** (analyst rating changes, earnings surprises, insider-selling acceleration) and blends it with the portfolio-level score to drive **dynamic position sizing**.
6. Backtests the whole pipeline against three real shocks — the Aug 2024 jobs report, the Apr 2025 tariff shock, and the Mar 2026 Iran conflict — measuring how many days earlier the system flagged danger versus a naive VaR baseline.

---

## Live demo

An interactive version of this system is deployed with Streamlit — enter any tickers (not limited to the ones used in the notebooks) and get a live risk analysis:

**[Live app →](https://sentrivar-500-7wjxew8z8yhhrad5ruidgw.streamlit.app/)**

The app (`app/app.py`, backed by `app/risk_engine.py`) re-runs the full pipeline — regime detection, sentiment, Copula amplification, idiosyncratic risk, and allocation — on whatever tickers the user enters, refitting the HMM and pulling fresh data each time. News sentiment in the live app uses a fast keyword-based scorer instead of FinBERT, trading a small amount of accuracy for near-instant response time.

---

## Why this design

A single VaR number can't tell you *why* risk is high or *what kind* of risk it is. This project treats risk as coming from three layers that behave differently:

- **Macro/systemic** — rates, inflation, geopolitics — moves every asset at once
- **Market structure** — sector correlation breakdown, volatility clustering — signals *contagion*, not just magnitude
- **Idiosyncratic** — earnings, analyst sentiment, insider activity — risk specific to one name

Most retail-level risk projects only model the first layer. Combining all three, with regime-dependent weighting instead of static weights, is what lets the system react faster and more specifically than a plain historical VaR model.

---

## Results

| Case | Shock date | Alert triggered | Lead time |
|---|---|---|---|
| 2024 Jobs Report Shock | 2024-08-05 | 2024-08-05 | 0 days |
| 2025 Tariff Shock | 2025-04-07 | 2025-04-04 | **3 days** |
| 2026 Iran Conflict | 2026-03-30 | 2026-03-27 | **3 days** |

In the Crisis regime, the Copula-amplified risk score saturates to its maximum (1.0) well before the naive linear score does — during the 2020 COVID crash, for example, the amplified score hits 1.0 while the linear score peaks around 0.94, several trading days earlier in the drawdown.

Portfolio composition (SOXX, a semiconductor ETF) showed distinctly different behavior across shocks — surging +61% during the 2026 Iran conflict window even as JPM and AAPL were roughly flat or negative, illustrating why sector-level idiosyncratic signals matter for allocation, not just headline VIX.

---

## Data sources

| Signal | Source |
|---|---|
| Prices (AAPL, MSFT, GOOGL, JPM, SOXX) | `yfinance` |
| VIX, 10Y–2Y Treasury spread, CPI | FRED (`pandas_datareader`) |
| News headlines | NewsAPI |
| Sentiment scoring | FinBERT (`ProsusAI/finbert`, HuggingFace) |
| Analyst ratings, earnings surprises, insider transactions | `yfinance` |

All data is free-tier / public. No paid terminals or licensed data required.

---

## Project structure

```
SentriVaR-500/
├── data/                        # Generated data + cached results (not committed)
├── notebooks/
│   ├── 01_data_pipeline.ipynb       # Price + macro data collection
│   ├── 02_risk_metrics.ipynb        # VaR, CVaR, Sharpe, Max Drawdown
│   ├── 03_sentiment.ipynb           # FinBERT + systemic/idiosyncratic news split
│   ├── 04_hmm_regime.ipynb          # HMM regime detection (Normal/Elevated/Crisis)
│   ├── 05_scoring.ipynb             # Regime-weighted multi-signal risk score
│   ├── 06_signals.ipynb             # Sector correlation risk
│   ├── 07_copula_risk.ipynb         # Non-linear tail-risk amplification
│   ├── 08_allocation.ipynb          # Idiosyncratic risk + dynamic allocation
│   └── 09_backtest.ipynb            # Case study validation (3 real shocks)
├── app/
│   ├── app.py                       # Streamlit dashboard (live, any tickers)
│   └── risk_engine.py               # Reusable calculation functions
├── requirements.txt
└── README.md
```

---

## Methodology notes

- **Regime detection**: `GaussianHMM` (3 components) on standardized returns, 20-day rolling volatility, and VIX, with a 30-day smoothing filter and regimes auto-labeled by mean VIX (lowest → Normal, highest → Crisis) rather than assumed a priori.
- **Copula amplification**: rather than a full copula fit, the project uses a simplified stand-in — a regime-conditional exponential amplification (`vix_factor ** 1.5` in Crisis) — that reproduces the tail-dependence *behavior* (signals compounding rather than adding) without the estimation overhead of a true copula model. This tradeoff is intentional and documented rather than disguised.
- **Idiosyncratic risk**: SOXX (an ETF) has no single-company earnings/insider data, so it's assigned a neutral 0.5 idiosyncratic score by design — reflecting that diversified vehicles genuinely carry less name-specific risk, not a data gap being papered over.

---

## Limitations / next steps

- Case studies use a small, fixed asset universe (4 large-caps + 1 sector ETF); results may not generalize to small-caps or less liquid names.
- SEC EDGAR full-text insider parsing was attempted but proved unreliable for structured extraction; the project falls back to `yfinance`'s `insider_transactions` with a sell-acceleration heuristic instead.
- A future iteration could add: SHAP-based attribution for *which* signal drove a given alert, a live Streamlit dashboard, and a proper Gaussian/Student-t copula fit in place of the current amplification heuristic.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
NEWSAPI_KEY=your_newsapi_key_here
```

Run notebooks in order, 01 through 09. Each notebook reads from and writes to `data/`, so later notebooks depend on earlier ones having been run at least once.