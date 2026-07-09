#  수익률과 리스크의 기본 감각 익히기
import yfinance as yf
import pandas as pd
import numpy as np

START_DATE = "2018-01-01"

# GS 빼고 일단 4개로
tickers = ["AAPL", "MSFT", "GOOGL", "JPM"]
# 데이터 받기
df = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)["Close"]

# NaN 행 제거
df = df.dropna()

if df.empty:
    raise SystemExit(
        "주가 데이터를 가져오지 못했습니다. 인터넷 연결을 확인한 뒤 다시 실행해주세요."
    )

# 일별 수익률
returns = df.pct_change().dropna()

print("평균 수익률:\n", returns.mean()) # 2018년도부터 하루에 평균 몇 프로 올랐는지
print("\n변동성:\n", returns.std())    # 수익률이 얼마나 들쭉날쭉 한지 높을 수록 리스크 큰 주식
print("\n상관관계:\n", returns.corr())  # 두 주식이 같이 오르고 떨어지는지(1에 가까울 수록 같이 움직이고 0에 가까울수록 독립적

# VaR 95프로 확률로 하루 손실이 이 숫자 이내
confidence = 0.95
VaR = returns["AAPL"].quantile(1 - confidence)
print(f"VaR (95%): {VaR: .4f}")
# VaR (95%): -0.0297 => 하루 손실이 2.97프로 이내

# CVaR 하위 5% 최악의 날들의 평균 손실
CVaR = returns["AAPL"][returns["AAPL"] <= VaR].mean()
print(f"CVaR (95%): {CVaR: .4f}")
# CVaR -0.0438 => 최악의 날 평균 4.28% 손실


#Sharpe 리스크 한 단위당 수익률이 얼마나 되는지
sharpe = returns["AAPL"].mean() / returns["AAPL"].std() * np.sqrt(252)  # 252 =  일년 거래일 292일
print(f"Sharpe Ratio: {sharpe: .4f}")
# Sharpe Ratio:  0.9394 => 1.0에 가까운 수준 시장 평균이랑 비슷한 효율 낫배드 (1.0 이상이면 좋은 편이고, 2.0 이상이면 우수)

# MDD
cumulative = (1 + returns["AAPL"]).cumprod()   # cumprod() : 수익률 누적해서 실제 자산 가치로 변환
rolloing_max = cumulative.cummax()             # cummax(): 매일 그 시점까지의 최고점 기록
drawdown = (cumulative - rolloing_max) / rolloing_max   # drawdown: 현재 가치가 최고점 대비 얼마나 떨어졌는지
max_drawdown = drawdown.min()                  # min(): 그 중에 가장 많이 떨어진 순간 
print(f"MDD: {max_drawdown: .4f}")
# MDD: -0.3852 => 고점대비 최대 38.52프로 떨어진적이 있음

# Macro economy data
import pandas_datareader as pdr

# FRED에서 데이터 가져오기
try:
    vix = pdr.get_data_fred("VIXCLS", start=START_DATE)     #VIX: 시장 공포 지수(30이 넘으면 투자자들이 불안하다는 신호)
    spread = pdr.get_data_fred("T10Y2Y", start=START_DATE)     # 국채 스프레드, T10Y2Y: 10년 국채 금리에서 2년 국채 금리를 뺸 값(마이너스로 떨어지면 경기침체 선행 신호로 분석)
    cpi = pdr.get_data_fred("CPIAUCSL", start=START_DATE)     # 소비자 물가 지수 (인플레이션 측정 지표)
except Exception as exc:
    raise SystemExit(
        "FRED 데이터를 가져오지 못했습니다. 인터넷 연결을 확인한 뒤 다시 실행해주세요."
    ) from exc

# 주가 데이터랑 날짜 기준으로 합치기
macro = pd.concat([vix, spread, cpi], axis=1) # 날짜 기준으로 하나의 테이블로 합치기
macro.columns = ["VIX", "Spread", "CPI"]

# Run FinBERT
from transformers import pipeline

try:
    # model load
    nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # testing
    headlines = [
        "Apple reports record quarterly earnings",
        "Markets crash amid tariff fears",
        "Fed signals potential rate hikes"
    ]
    # "Apple reports record quarterly earnings"
    # → positive (72.9%)

    # "Markets crash amid tariff fears"
    # → negative (95.2%)

    # "Fed signals potential rate hikes"
    # → positive (40.5%) => FinBERT의 한계(금융 맥락을 완전히 이해 못하는 케이스): Fed 금리 인상 신호를 positive로 분류했는데, 실제로는 시장에 부정적인 뉴스임
    # 신뢰도도 40%로 낮아서 모델이 확신을 못 하고 있음

    for h in headlines:
        result = nlp(h)
        print(f"{h}\n→ {result}\n")
except Exception as exc:
    print(f"FinBERT 실행 건너뜀: {exc}")

# Run HMM
# 시장에는 숨겨진 "국면"이 있다 (Normal / Elevated / Crisis)
# HMM은 관찰 가능한 데이터 (수익률, VIX)로 그 숨겨진 국면을 추정하는 모델
from hmmlearn import hmm
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# 국면별 색상 및 라벨 정의
colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
labels = {0: "Normal", 1: "Elevated", 2: "Crisis"}

# 수익률 + 20일 변동성 + VIX 세 개를 입력으로 사용
hmm_data = returns[["AAPL"]].copy()
hmm_data["volatility"] = returns["AAPL"].rolling(20).std()  # 20일 롤링 변동성 추가
hmm_data = hmm_data.join(vix).dropna()
dates = hmm_data.index
X = hmm_data.values

# HMM 학습 및 국면 예측
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(X)
states = model.predict(X)

# Smoothing (20일 단위 평탄화)
smoothed_states = uniform_filter1d(states.astype(float), size=20)
smoothed_states = np.round(smoothed_states).astype(int)
state_series = pd.Series(smoothed_states, index=dates)

# 평균 VIX 기준으로 자동 라벨링 (낮음=Normal, 높음=Crisis)
vix_aligned = vix.reindex(dates)["VIXCLS"]
state_vix_means = {}
for s in range(3):
    mask = state_series == s
    state_vix_means[s] = vix_aligned[mask].mean()
    print(f"  State {s} 평균 VIX: {state_vix_means[s]:.1f}")

sorted_states = sorted(state_vix_means, key=state_vix_means.get)
label_map = {
    sorted_states[0]: 0,  # VIX 낮음 → Normal
    sorted_states[1]: 1,  # VIX 중간 → Elevated
    sorted_states[2]: 2,  # VIX 높음 → Crisis
}
state_series = state_series.map(label_map)

# 국면 비율 확인
print("\n국면 비율:")
for state, label in labels.items():
    ratio = (state_series == state).mean() * 100
    print(f"  {label}: {ratio:.1f}%")

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 주가 + 국면 배경색
aapl_price = df["AAPL"].reindex(dates)
ax1.plot(dates, aapl_price, color="black", linewidth=0.8, label="AAPL", zorder=3)

y_min = aapl_price.min() * 0.95
y_max = aapl_price.max() * 1.05

for state, color in colors.items():
    mask = state_series == state
    ax1.fill_between(dates, y_min, y_max,
                     where=mask.values, alpha=0.3,
                     color=color, label=labels[state])

ax1.set_ylim(y_min, y_max)
ax1.set_ylabel("AAPL Price")
ax1.set_title("SentriVaR-500: HMM Regime Detection")
ax1.legend(loc="upper left")

# VIX + 이벤트 수직선
ax2.plot(dates, vix.reindex(dates)["VIXCLS"],
         color="purple", linewidth=0.8, label="VIX")
ax2.axvline(pd.Timestamp("2020-03-16"), color="darkred",
            linestyle="--", linewidth=1.5, label="COVID Crash (Mar 2020)")
ax2.axvline(pd.Timestamp("2024-08-05"), color="orange",
            linestyle="--", linewidth=1.5, label="Jobs Report Shock (Aug 2024)")
ax2.axvline(pd.Timestamp("2025-04-07"), color="red",
            linestyle="--", linewidth=1.5, label="Tariff Shock (Apr 2025)")

ax2.set_ylabel("VIX")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("regime_detection_v3.png", dpi=150)
plt.show()
print("저장 완료: regime_detection_v3.png")