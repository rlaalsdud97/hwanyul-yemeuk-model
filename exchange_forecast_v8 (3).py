# ==============================================================================
# [FINAL MASTERPIECE V8] 실무형 퀀트 환율 예측 파이프라인
# (V7 대비 개선사항)
#   1) 시그널 로직 재설계  : 예측값 기준 밴드 → "최근 20일 실제 시세" 볼린저 밴드 기준
#   2) 크롤링 견고화       : 여러 셀렉터 순차 시도 + 실패 이유 명확 로깅
#   3) 예측률 지표 추가    : MAPE(정확도 %) + 방향성 정확도(Directional Accuracy %)
#   4) OOT align 버그 수정 : reindex+ffill 왜곡 방지 (index 교집합 기반 정렬)
# ==============================================================================
import requests
from bs4 import BeautifulSoup
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')
plt.rcdefaults()
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ============================================
# [0] 예측 기준 설정 (이 변수들만 수정)
# ============================================
target_date = "2026-09-28"     # 예측을 시작할 기준일 (과거 날짜 입력 시 백테스트 모드 가동)
forecast_days = 5              # 예측할 미래 영업일 수

print(f"\n🚀 [System] 파이프라인 구동 시작... (요청 기준일: {target_date}, 예측기간: {forecast_days}일)")

# ---------------------------------------------------
# [1] 상수 및 함수 정의
# ---------------------------------------------------
MAX_PROPHET_WEIGHT = 0.05
VOL_MULTIPLIER = 1.5
ANCHOR_RATE = 0.35  # 실시간 충격 완화 변수
BAND_SIGMA = 1.0    # 🔥 [신규] 시그널 밴드 폭 (최근 20일 표준편차의 몇 배를 밴드로 쓸지)

DECAY_VECTOR = np.linspace(1.0, 0.5, forecast_days)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def model_score(mae, rmse):
    return (mae + rmse) / 2

def mape_score(y_true, y_pred):
    """평균 절대 오차율(%) — 낮을수록 정확. 정확도(%) = 100 - MAPE"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def directional_accuracy(y_true_series, y_pred_series):
    """
    방향성 정확도(%): '전일 대비 오를지/내릴지' 방향을 맞춘 비율.
    트레이딩에서 절대 오차보다 실전적으로 중요한 지표.
    y_true_series, y_pred_series 는 동일 인덱스로 정렬된 '종가(level)' Series.
    """
    true_dir = np.sign(np.asarray(y_true_series, dtype=float)[1:] -
                       np.asarray(y_true_series, dtype=float)[:-1])
    pred_dir = np.sign(np.asarray(y_pred_series, dtype=float)[1:] -
                       np.asarray(y_true_series, dtype=float)[:-1])
    # 예측 '다음날 방향'을, 전일 실제값 대비 예측값의 방향으로 판단
    valid = true_dir != 0
    if valid.sum() == 0:
        return float('nan')
    return float(np.mean(true_dir[valid] == pred_dir[valid]) * 100)

# ---------------------------------------------------
# [2] 데이터 수집 및 무결점 Feature 생성 (VIX, KOSPI 포함)
# ---------------------------------------------------
print("1. 데이터를 수집하고 동적 Feature를 생성합니다 (VIX, KOSPI 지표 융합)...")
tickers = {
    "KRW": "KRW=X",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "VIX": "^VIX",
    "KOSPI": "^KS11"
}

end_fetch_date = (pd.to_datetime(target_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
raw = yf.download(list(tickers.values()), start="2022-01-01", end=end_fetch_date)

idx = pd.IndexSlice
ml = pd.DataFrame({
    "Close": raw.loc[:, idx["Close", "KRW=X"]],
    "High": raw.loc[:, idx["High", "KRW=X"]],
    "Low": raw.loc[:, idx["Low", "KRW=X"]],
    "DXY": raw.loc[:, idx["Close", "DX-Y.NYB"]],
    "US10Y": raw.loc[:, idx["Close", "^TNX"]],
    "VIX": raw.loc[:, idx["Close", "^VIX"]],
    "KOSPI": raw.loc[:, idx["Close", "^KS11"]]
})

ml = ml.ffill().dropna()

if target_date not in ml.index.strftime("%Y-%m-%d"):
    valid_dates = ml.index[ml.index <= pd.to_datetime(target_date)]
    if not valid_dates.empty:
        target_date = valid_dates[-1].strftime("%Y-%m-%d")
        print(f"⚠️ 휴장일 감지: 가장 가까운 이전 영업일({target_date})을 기준일로 자동 조정합니다.")

ml = ml.loc[:target_date]

# 기술적 지표 생성
ml["MA3"] = ml["Close"].rolling(3).mean()
ml["MA5"] = ml["Close"].rolling(5).mean()
ml["MA10"] = ml["Close"].rolling(10).mean()
ml["MA20"] = ml["Close"].rolling(20).mean()
ml["MA60"] = ml["Close"].rolling(60).mean()
ml["STD5"] = ml["Close"].rolling(5).std()
ml["Spread"] = ml["High"] - ml["Low"]
ml["Return"] = ml["Close"].pct_change()

for lag in [1, 2, 3, 5, 10]:
    ml[f"Lag_{lag}"] = ml["Close"].shift(lag)

ml["Momentum3"] = ml["Close"] - ml["Close"].shift(3)
ml["Momentum5"] = ml["Close"] - ml["Close"].shift(5)

delta = ml["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-9)
ml["RSI14"] = 100 - (100 / (1 + rs))

exp12 = ml["Close"].ewm(span=12, adjust=False).mean()
exp26 = ml["Close"].ewm(span=26, adjust=False).mean()
ml["MACD"] = exp12 - exp26
ml["MACD_Signal"] = ml["MACD"].ewm(span=9, adjust=False).mean()

# --- 거시경제 지표 가공 (과적합 방지 및 Lag 처리) ---
ml["DXY_delta"] = ml["DXY"].diff()
ml["US10Y_delta"] = ml["US10Y"].diff()
ml["VIX_delta"] = ml["VIX"].diff()
ml["KOSPI_return"] = ml["KOSPI"].pct_change() * 100

ml["DXY_lag1"] = ml["DXY"].shift(1)
ml["US10Y_lag1"] = ml["US10Y"].shift(1)
ml["DXY_delta_lag1"] = ml["DXY_delta"].shift(1)
ml["US10Y_delta_lag1"] = ml["US10Y_delta"].shift(1)
ml["VIX_delta_lag1"] = ml["VIX_delta"].shift(1)
ml["KOSPI_return_lag1"] = ml["KOSPI_return"].shift(1)

# 🔥 달력 파생 변수 (월말, 분기말, 요일 효과)
ml["Weekday"] = ml.index.dayofweek                  # 월(0) ~ 금(4)
ml["Is_Month_End"] = ml.index.is_month_end.astype(int)  # 월말 여부 (0 또는 1)
ml["Is_Quarter_End"] = ml.index.is_quarter_end.astype(int) # 분기말 여부 (0 또는 1)

features = [
    "Lag_1", "Lag_2", "Lag_3", "Lag_5", "Lag_10",
    "MA3", "MA5", "MA10", "MA20", "MA60", "STD5",
    "Spread", "Return", "Momentum3", "Momentum5", "RSI14", "MACD", "MACD_Signal",
    "DXY_lag1", "DXY_delta_lag1", "US10Y_lag1", "US10Y_delta_lag1",
    "VIX_delta_lag1", "KOSPI_return_lag1",
    "Weekday", "Is_Month_End", "Is_Quarter_End"
]

ml = ml.dropna()

ml_step1 = ml.copy()
ml_step1["Target_Step1"] = ml_step1["Close"].shift(-1) - ml_step1["Close"]
ml_step1 = ml_step1.dropna(subset=["Target_Step1"])

# ---------------------------------------------------
# [3] 동적 OOT(블라인드 테스트) 날짜 계산 및 모델 평가
# ---------------------------------------------------
split_dt = ml_step1.index[-1] - pd.DateOffset(months=1)
split_date = split_dt.strftime("%Y-%m-%d")
test_start_date = (split_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

train_oot = ml_step1.loc[:split_date]
test_oot = ml_step1.loc[test_start_date:]
X_train_oot, y_train_oot = train_oot[features], train_oot["Target_Step1"]
X_test_oot, y_test_oot = test_oot[features], test_oot["Target_Step1"]

print(f"2. OOT 동적 검증 수행 (Train ➔ ~{split_date} | Test ➔ {test_start_date} ~)")

best_params_ = {
    'subsample': 0.7,
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.01,
    'colsample_bytree': 0.8
}

oot_model = XGBRegressor(**best_params_, random_state=42).fit(X_train_oot, y_train_oot)
xgb_preds_oot_delta = oot_model.predict(X_test_oot)

pred_close_xgb = test_oot["Close"] + xgb_preds_oot_delta
true_close_xgb = test_oot["Close"] + y_test_oot
xgb_mae, xgb_rmse = evaluate_model(true_close_xgb, pred_close_xgb)

arima_train = ml.loc[:split_date, "Close"].asfreq('B').ffill()
arima_test = ml.loc[test_start_date:, "Close"].asfreq('B').ffill()
arima_eval_model = ARIMA(arima_train, order=(0, 1, 1)).fit()
arima_eval_preds = arima_eval_model.forecast(steps=len(arima_test))
arima_mae, arima_rmse = evaluate_model(arima_test, arima_eval_preds)

prophet_train = ml.loc[:split_date].reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
prophet_train['ds'] = prophet_train['ds'].dt.tz_localize(None)
prophet_test = ml.loc[test_start_date:].reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
prophet_test['ds'] = prophet_test['ds'].dt.tz_localize(None)

prophet_eval_model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.01).fit(prophet_train)
prophet_eval_preds = prophet_eval_model.predict(prophet_test[['ds']])['yhat'].values
prophet_mae, prophet_rmse = evaluate_model(prophet_test['y'], prophet_eval_preds)

# ---------------------------------------------------
# [4] 목표 기간 다중 스텝 직접 예측 (Direct Multi-step)
# ---------------------------------------------------
print(f"3. 확정된 기준일({ml.index[-1].strftime('%Y-%m-%d')}) 바탕으로 미래 {forecast_days}영업일 예측 생성...")

automated_today_price = float(ml["Close"].iloc[-1])
latest_features = ml[features].iloc[[-1]].copy()
forecast_results = []

# 최종 앙상블용 전체 데이터 학습 모델 객체 보관용
final_xgb_models = []

for step in range(1, forecast_days + 1):
    target_name = f"Target_Delta_Step{step}"
    ml[target_name] = ml["Close"].shift(-step) - ml["Close"]
    train_df = ml.dropna(subset=features + [target_name])
    X_train, y_train = train_df[features], train_df[target_name]

    model_step = XGBRegressor(**best_params_, random_state=42).fit(X_train, y_train)
    final_xgb_models.append(model_step)

    pred_cumulative_delta = float(model_step.predict(latest_features)[0])
    forecast_results.append(automated_today_price + pred_cumulative_delta)

future_bdays = pd.bdate_range(start=ml.index[-1] + pd.offsets.BDay(1), periods=forecast_days)
forecast_dates = future_bdays.strftime("%Y-%m-%d").tolist()
forecast_rounded = [round(p, 2) for p in forecast_results]

# ---------------------------------------------------
# [5] 예측 수행 및 앙상블 가중치 할당
# ---------------------------------------------------
arima_series = ml["Close"].asfreq('B').ffill()
arima_fit = ARIMA(arima_series, order=(0, 1, 1)).fit()
arima_forecast = arima_fit.forecast(steps=forecast_days).values

prophet_df = ml.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.01).fit(prophet_df)

future_dates_df = pd.DataFrame({'ds': pd.to_datetime(forecast_dates)})
prophet_forecast = prophet_model.predict(future_dates_df)['yhat'].values

last_close = float(ml["Close"].iloc[-1])
volatility = ml["Spread"].tail(20).mean()
dynamic_limit = volatility * VOL_MULTIPLIER
prophet_forecast = np.clip(prophet_forecast, last_close - dynamic_limit, last_close + dynamic_limit)

xgb_forecast = np.array(forecast_rounded)

score_xgb = model_score(xgb_mae, xgb_rmse)
score_arima = model_score(arima_mae, arima_rmse)
score_prophet = model_score(prophet_mae, prophet_rmse)

scores = np.array([score_xgb, score_arima, score_prophet])
inverse_scores = 1 / scores
weights = inverse_scores / inverse_scores.sum()

if weights[2] > MAX_PROPHET_WEIGHT:
    extra = weights[2] - MAX_PROPHET_WEIGHT
    weights[2] = MAX_PROPHET_WEIGHT
    ratio_xgb = weights[0] / (weights[0] + weights[1])
    weights[0] += extra * ratio_xgb
    weights[1] += extra * (1 - ratio_xgb)

weight_xgb, weight_arima, weight_prophet = weights
ensemble_forecast = (xgb_forecast * weight_xgb) + (arima_forecast * weight_arima) + (prophet_forecast * weight_prophet)

# ---------------------------------------------------
# [6] 실시간 Re-anchoring (백테스트 자동 인식) + 크롤링 견고화
# ---------------------------------------------------
global_model_baseline = last_close
days_diff = (pd.Timestamp.today().normalize() - pd.to_datetime(target_date)).days


def fetch_naver_usdkrw():
    """
    실시간 USD/KRW 환율 수집.
    [핵심] 네이버 marketindex 페이지는 이제 값을 JavaScript로 나중에 채우므로
    requests+BeautifulSoup으로는 숫자를 못 읽는다. 그래서 네이버가 내부적으로 쓰는
    JSON API를 1순위로 호출하고, 실패 시에만 구식 HTML 파싱을 시도한다.
    반환: (rate: float|None, message: str)
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0 Safari/537.36"),
        "Referer": "https://finance.naver.com/marketindex/",
    }

    def _to_float(v):
        """'1,363.30' 같은 콤마 포함 문자열 → 1363.3 float. 실패 시 None."""
        if v is None:
            return None
        try:
            return float(str(v).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    # ── 1순위: 네이버 증권 JSON API (가장 안정적) ──────────────────
    # 응답 구조 확인 결과:
    #  1) .../exchange/FX_USDKRW           → data["exchangeInfo"]["closePrice"]
    #  2) front-api/marketIndex/prices     → data["result"][0]["closePrice"]
    api_urls = [
        "https://api.stock.naver.com/marketindex/exchange/FX_USDKRW",
        "https://m.stock.naver.com/front-api/marketIndex/prices?category=exchange&reutersCode=FX_USDKRW&page=1",
    ]
    for api in api_urls:
        try:
            r = requests.get(api, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()

            # 여러 응답 구조를 방어적으로 탐색
            candidates = []
            if isinstance(data.get("exchangeInfo"), dict):
                candidates.append(data["exchangeInfo"].get("closePrice"))
            if isinstance(data.get("result"), list) and data["result"]:
                candidates.append(data["result"][0].get("closePrice"))
            if isinstance(data.get("result"), dict):
                candidates.append(data["result"].get("closePrice"))
            candidates.append(data.get("closePrice"))

            for c in candidates:
                rate = _to_float(c)
                if rate is not None:
                    return rate, f"네이버 JSON API 성공: {api.split('?')[0]}"
        except (requests.RequestException, ValueError, KeyError):
            continue

    # ── 2순위: 구식 HTML 파싱 (혹시 서버 렌더링될 때 대비) ──────────
    try:
        response = requests.get("https://finance.naver.com/marketindex/",
                                headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        selectors = [
            "#exchangeList > li.on > a.head.usd > div > span.value",
            "#exchangeList > li:nth-child(1) span.value",
            "div.market1 span.value",
            "span.value",
        ]
        for sel in selectors:
            node = soup.select_one(sel)
            if node is not None and node.text.strip():
                try:
                    return float(node.text.replace(",", "").strip()), f"HTML 셀렉터 성공: '{sel}'"
                except ValueError:
                    continue
    except requests.RequestException as e:
        return None, f"JSON API·HTML 모두 실패 (네트워크 오류: {e})"

    return None, "JSON API·HTML 파싱 모두 값을 찾지 못함 (네이버 구조 변경 의심)"


if days_diff > 3:
    print(f"\n🕰️ [백테스트 모드] 과거 날짜({target_date}) 검증이므로 네이버 실시간 보정을 차단합니다.")
    current_rate = global_model_baseline
    crawl_ok = False
else:
    print("\n🌐 [자동화] 네이버 금융 실시간 환율 수집 중...")
    rate, msg = fetch_naver_usdkrw()
    if rate is not None:
        current_rate = rate
        crawl_ok = True
        print(f"✅ 실시간 환율 수집 성공: {current_rate}원 ({msg})")
    else:
        current_rate = global_model_baseline
        crawl_ok = False
        print(f"⚠️ 크롤링 실패 → 모델 종가({global_model_baseline:.2f}원)로 대체합니다.")
        print(f"   ↳ 실패 원인: {msg}")

market_basis_gap = (current_rate - global_model_baseline) * ANCHOR_RATE

xgb_forecast_adj = xgb_forecast + (market_basis_gap * DECAY_VECTOR)
arima_forecast_adj = arima_forecast + (market_basis_gap * DECAY_VECTOR)
prophet_forecast_adj = prophet_forecast + (market_basis_gap * DECAY_VECTOR)
ensemble_forecast_adj = ensemble_forecast + (market_basis_gap * DECAY_VECTOR)

# ---------------------------------------------------
# [7] 시그널 산출 (🔥 재설계: 최근 20일 실제 시세 볼린저 밴드 기준)
# ---------------------------------------------------
# [변경 이유]
#   기존 로직은 밴드 중심이 '예측값'이라, 크롤링 실패로 current_rate가 모델 종가로
#   대체되면 밴드 중심과 현재가가 붙어버려 사실상 항상 WAIT만 출력되는 버그가 있었음.
#   → 밴드 중심을 '최근 20일 실제 종가 이동평균'으로 바꿔, "지금 환율이 최근 흐름 대비
#     싼가/비싼가"를 판단하도록 재설계. 크롤링 성공/실패와 무관하게 논리가 일관됨.
recent20 = ml["Close"].tail(20)
band_center = float(recent20.mean())   # 최근 20일 실제 평균 (밴드 중심)
band_std = float(recent20.std())       # 최근 20일 실제 표준편차

lower_band = band_center - BAND_SIGMA * band_std   # 저가 구간(매수 유리)
upper_band = band_center + BAND_SIGMA * band_std   # 고가 구간(매수 불리)

# 현재가가 밴드 내 어디에 위치하는지 (%B: 0=하단, 1=상단)
band_width = (upper_band - lower_band)
percent_b = (current_rate - lower_band) / band_width if band_width > 0 else 0.5


def make_signal(rate, low, up):
    """현재가가 최근 20일 밴드에서 어디에 있는지로 시그널 판정."""
    width = up - low
    if width <= 0:
        return "🟡 HOLD"
    pos = (rate - low) / width   # 0(하단)~1(상단)
    if pos <= 0.33:
        return "🟢 BUY"    # 최근 흐름 대비 싼 구간 → 분할 매수 유리
    elif pos <= 0.66:
        return "🟡 HOLD"   # 중립 구간
    else:
        return "🔴 WAIT"   # 최근 흐름 대비 비싼 구간 → 조정 대기


# 예측 5일치 각각에 대해서도, 해당 예측 종가가 밴드 어디에 있는지로 시그널 산출
signals = [make_signal(p, lower_band, upper_band) for p in ensemble_forecast_adj]

# 표에 표시할 밴드(고정된 최근 20일 밴드)
buy_zone_arr = np.full(forecast_days, lower_band)
sell_zone_arr = np.full(forecast_days, upper_band)

ensemble_df = pd.DataFrame({
    "Date": forecast_dates,
    "XGBoost": np.round(xgb_forecast_adj, 2),
    "ARIMA": np.round(arima_forecast_adj, 2),
    "Prophet": np.round(prophet_forecast_adj, 2),
    "🔥 Final": np.round(ensemble_forecast_adj, 2),
    "👇 Buy Band": np.round(buy_zone_arr, 2),
    "☝️ Sell Band": np.round(sell_zone_arr, 2),
    "🎯 Signal": signals
})

print("\n" + "=" * 95)
print(f" [FINAL V8] KOSPI/VIX 결합 자동화 앙상블 파이프라인 (기준일: {target_date} | {forecast_days}일 예측)")
print("=" * 95)

model_names = ["XGBoost", "ARIMA", "Prophet"]
best_model_idx = np.argmin(scores)

print(f" • Best Model : {model_names[best_model_idx]} (Score: {scores[best_model_idx]:.2f})")
print(f" • Final Ensemble Weight : {weight_xgb:.1%} / {weight_arima:.1%} / {weight_prophet:.1%}")
print(f" • 최근20일 밴드 : 중심 {band_center:.2f} / 하단 {lower_band:.2f} / 상단 {upper_band:.2f}")
print(f" • 현재가 밴드 위치(%B) : {percent_b*100:.1f}%  (0%=하단/저가, 100%=상단/고가)")
print("=" * 95)

try:
    from IPython.display import display
    display(ensemble_df)
except ImportError:
    print(ensemble_df.to_string(index=False))

print("=" * 95)

# 오늘 시그널 = 현재가(spot) 기준 밴드 위치로 판정 (가장 실전적)
today_signal = make_signal(current_rate, lower_band, upper_band)
print("\n📊 [Executive Summary] 핵심 요약")
print("-" * 60)
print(f"📌 기준가 (Spot) : {current_rate:.2f} KRW"
      + ("" if crawl_ok else "  (⚠️ 실시간 수집 실패 → 모델 종가 대체)"))
print(f"📌 밴드 위치(%B) : {percent_b*100:.1f}%")
print(f"📌 트레이딩 시그널 : {today_signal}")

if today_signal == "🟢 BUY":
    print("→ 현재 환율이 최근 20일 흐름 대비 '저가 구간(밴드 하단)'에 있습니다.")
    print("→ 상대적으로 유리한 레벨이므로 분할 매수 결제를 고려할 만합니다.")
elif today_signal == "🟡 HOLD":
    print("→ 현재 환율은 최근 흐름의 중립 구간에 있습니다.")
    print("→ 급격한 진입보다 시장 추이를 하루 더 관찰하는 것을 권장합니다.")
else:
    print("→ 현재 환율이 최근 20일 흐름 대비 '고가 구간(밴드 상단)'에 있습니다.")
    print("→ 달러 매수를 보류하고 조정을 기다리는 보수적 전략을 권장합니다.")
print("-" * 60 + "\n")

import sys
sys.stdout.flush()   # Colab에서 display()와 print() 출력 순서가 섞이는 것 방지

# ---------------------------------------------------
# [8] 시각화 (Model Comparison)
# ---------------------------------------------------
recent_history = ml.tail(30)
forecast_idx = pd.to_datetime(forecast_dates)
connect_idx = [recent_history.index[-1]] + list(forecast_idx)

connect_xgb = [current_rate] + list(xgb_forecast_adj)
connect_arima = [current_rate] + list(arima_forecast_adj)
connect_prophet = [current_rate] + list(prophet_forecast_adj)
connect_ensemble = [current_rate] + list(ensemble_forecast_adj)

plt.figure(figsize=(10, 5))
plt.plot(recent_history.index, recent_history["Close"], label="Actual (Past 30 Days)", color="#2c3e50", linewidth=2.5)
plt.plot(recent_history.index[-1], current_rate, marker='o', markersize=8, color="#2c3e50", label=f"Current Spot ({current_rate:.2f} KRW)")

plt.plot(connect_idx, connect_xgb, label=f"XGBoost ({weight_xgb:.1%})", color="#27ae60", linestyle="--", alpha=0.7)
plt.plot(connect_idx, connect_arima, label=f"ARIMA ({weight_arima:.1%})", color="#2980b9", linestyle="-.", alpha=0.7)
plt.plot(connect_idx, connect_prophet, label=f"Prophet ({weight_prophet:.1%} Cap)", color="#f39c12", linestyle=":", alpha=0.7)
plt.plot(connect_idx, connect_ensemble, label=" Final Ensemble", color="#c0392b", linewidth=3.5, marker='s')

# 🔥 시그널 밴드(최근 20일 실제 밴드)를 수평 밴드로 표시
plt.axhline(band_center, color="#8e44ad", linestyle="-", alpha=0.4, label=f"20D Mean ({band_center:.1f})")
plt.axhspan(lower_band, upper_band, color="#8e44ad", alpha=0.10, label=f"Signal Band (±{BAND_SIGMA}σ)")

plt.title(f"USDKRW {forecast_days}-Day Forecast: Ensemble Model & Trading Band", fontsize=15, fontweight="bold", pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Exchange Rate (KRW/USD)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# [9] 최종 앙상블 모델 OOT 구간 정확도 (🔧 align 버그 수정 + 예측률 지표 추가)
# ---------------------------------------------------
# [변경 이유]
#   기존: reindex(val_idx).ffill() → 인덱스가 미세하게 안 맞으면 대량 NaN이 ffill로
#         메꿔지며 정확도가 왜곡될 수 있었음.
#   수정: '실제 겹치는 날짜(교집합)'로만 정렬해 왜곡을 방지. ffill 최소화.
prophet_series = pd.Series(prophet_eval_preds, index=pd.to_datetime(prophet_test['ds'].values))
arima_series_eval = pd.Series(np.asarray(arima_eval_preds), index=pd.to_datetime(arima_test.index))

# XGB 예측 인덱스를 tz 제거해 통일
xgb_series = pred_close_xgb.copy()
xgb_series.index = pd.to_datetime(xgb_series.index).tz_localize(None)
true_series = true_close_xgb.copy()
true_series.index = pd.to_datetime(true_series.index).tz_localize(None)
prophet_series.index = prophet_series.index.tz_localize(None) if prophet_series.index.tz is not None else prophet_series.index
arima_series_eval.index = arima_series_eval.index.tz_localize(None) if arima_series_eval.index.tz is not None else arima_series_eval.index

# 세 모델과 실제값이 '모두 존재하는' 공통 날짜만 사용
common_idx = xgb_series.index.intersection(arima_series_eval.index).intersection(prophet_series.index)
common_idx = common_idx.intersection(true_series.index)

if len(common_idx) == 0:
    print("⚠️ OOT 공통 인덱스가 없어 앙상블 검증을 건너뜁니다.")
else:
    aligned_true = true_series.loc[common_idx]
    aligned_xgb = xgb_series.loc[common_idx]
    aligned_arima = arima_series_eval.loc[common_idx]
    aligned_prophet = prophet_series.loc[common_idx]

    ensemble_preds_oot = (aligned_xgb * weight_xgb) + (aligned_arima * weight_arima) + (aligned_prophet * weight_prophet)

    ensemble_mae, ensemble_rmse = evaluate_model(aligned_true, ensemble_preds_oot)
    ensemble_mape = mape_score(aligned_true, ensemble_preds_oot)
    ensemble_acc = 100 - ensemble_mape                       # 정확도(%) = 100 - MAPE
    ensemble_da = directional_accuracy(aligned_true, ensemble_preds_oot)  # 방향성 정확도(%)

    print("=" * 60)
    print("  📈 [최종 앙상블 모델 OOT 검증 결과]")
    print("-" * 60)
    print(f"  • 검증 표본 수     : {len(common_idx)}일")
    print(f"  • MAE (평균오차)   : {ensemble_mae:.2f} 원")
    print(f"  • RMSE            : {ensemble_rmse:.2f} 원")
    print(f"  • MAPE (오차율)    : {ensemble_mape:.2f} %")
    print(f"  • ✅ 예측 정확도   : {ensemble_acc:.2f} %   (100 - MAPE)")
    print(f"  • 🎯 방향성 정확도 : {ensemble_da:.1f} %   (오를지/내릴지 방향 적중률)")
    print("=" * 60)
    print("  [해석 가이드]")
    print("   - 예측 정확도(%)는 '값'이 얼마나 가까운지 → 환율은 원래 하루 변동이 작아 대개 높게 나옴.")
    print("   - 방향성 정확도(%)가 실전 핵심: 50%면 동전던지기, 55~60%+면 유의미. ")
    print("=" * 60)
