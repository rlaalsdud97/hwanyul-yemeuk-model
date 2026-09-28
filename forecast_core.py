# ==============================================================================
# forecast_core.py
# 환율 예측 파이프라인의 '엔진' 모듈.
#   - exchange_forecast_v8.py 의 검증된 로직을 run_forecast() 하나로 묶었습니다.
#   - 대시보드(app.py)에서 import 해서 사용합니다.
#   - print/plt.show() 같은 출력 대신, 결과를 dict로 '반환'합니다.
# ==============================================================================
import requests
from bs4 import BeautifulSoup
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------
# 상수
# ---------------------------------------------------
MAX_PROPHET_WEIGHT = 0.05
VOL_MULTIPLIER = 1.5
ANCHOR_RATE = 0.35
BAND_SIGMA = 1.0
BEST_PARAMS = {
    "subsample": 0.7,
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.01,
    "colsample_bytree": 0.8,
}


# ---------------------------------------------------
# 지표 함수
# ---------------------------------------------------
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
    """방향성 정확도(%): '전일 대비 오를지/내릴지' 방향을 맞춘 비율."""
    t = np.asarray(y_true_series, dtype=float)
    p = np.asarray(y_pred_series, dtype=float)
    true_dir = np.sign(t[1:] - t[:-1])
    pred_dir = np.sign(p[1:] - t[:-1])
    valid = true_dir != 0
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(true_dir[valid] == pred_dir[valid]) * 100)


def make_signal(rate, low, up):
    """현재가가 최근 20일 밴드에서 어디에 있는지로 시그널 판정."""
    width = up - low
    if width <= 0:
        return "🟡 HOLD"
    pos = (rate - low) / width
    if pos <= 0.33:
        return "🟢 BUY"
    elif pos <= 0.66:
        return "🟡 HOLD"
    else:
        return "🔴 WAIT"


# ---------------------------------------------------
# 실시간 환율 크롤링 (네이버 JSON API)
# ---------------------------------------------------
def fetch_naver_usdkrw():
    """
    실시간 USD/KRW 환율 수집. 네이버 JSON API 1순위, HTML 파싱 2순위.
    반환: (rate: float|None, message: str)
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0 Safari/537.36"),
        "Referer": "https://finance.naver.com/marketindex/",
    }

    def _to_float(v):
        if v is None:
            return None
        try:
            return float(str(v).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    api_urls = [
        "https://api.stock.naver.com/marketindex/exchange/FX_USDKRW",
        "https://m.stock.naver.com/front-api/marketIndex/prices?category=exchange&reutersCode=FX_USDKRW&page=1",
    ]
    for api in api_urls:
        try:
            r = requests.get(api, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
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
                v = _to_float(node.text)
                if v is not None:
                    return v, f"HTML 셀렉터 성공: '{sel}'"
    except requests.RequestException as e:
        return None, f"JSON API·HTML 모두 실패 (네트워크 오류: {e})"

    return None, "JSON API·HTML 파싱 모두 값을 찾지 못함 (네이버 구조 변경 의심)"


# ---------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------
def run_forecast(target_date="2026-09-28", forecast_days=5, use_realtime=True, log=None):
    """
    환율 예측 파이프라인 실행.

    Parameters
    ----------
    target_date : str  예측 기준일 (과거 날짜면 백테스트 모드)
    forecast_days : int  예측할 미래 영업일 수
    use_realtime : bool  네이버 실시간 환율 보정 사용 여부
    log : callable|None  진행상황 로그 콜백 (예: st.write). None이면 무시.

    Returns
    -------
    dict  예측 결과 전체 (아래 키 참조)
    """
    def _log(msg):
        if log is not None:
            log(msg)

    decay_vector = np.linspace(1.0, 0.5, forecast_days)

    # [2] 데이터 수집 -------------------------------------------------
    _log("데이터 수집 및 Feature 생성 중 (VIX, KOSPI 융합)...")
    tickers = {
        "KRW": "KRW=X", "DXY": "DX-Y.NYB", "US10Y": "^TNX",
        "VIX": "^VIX", "KOSPI": "^KS11",
    }
    end_fetch_date = (pd.to_datetime(target_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers.values()), start="2022-01-01", end=end_fetch_date, progress=False)

    idx = pd.IndexSlice
    ml = pd.DataFrame({
        "Close": raw.loc[:, idx["Close", "KRW=X"]],
        "High": raw.loc[:, idx["High", "KRW=X"]],
        "Low": raw.loc[:, idx["Low", "KRW=X"]],
        "DXY": raw.loc[:, idx["Close", "DX-Y.NYB"]],
        "US10Y": raw.loc[:, idx["Close", "^TNX"]],
        "VIX": raw.loc[:, idx["Close", "^VIX"]],
        "KOSPI": raw.loc[:, idx["Close", "^KS11"]],
    })
    ml = ml.ffill().dropna()

    adjusted_note = None
    if target_date not in ml.index.strftime("%Y-%m-%d"):
        valid_dates = ml.index[ml.index <= pd.to_datetime(target_date)]
        if not valid_dates.empty:
            new_td = valid_dates[-1].strftime("%Y-%m-%d")
            adjusted_note = f"휴장일 감지: 기준일을 {new_td}로 자동 조정"
            _log("⚠️ " + adjusted_note)
            target_date = new_td
    ml = ml.loc[:target_date]

    # 기술적 지표
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

    ml["Weekday"] = ml.index.dayofweek
    ml["Is_Month_End"] = ml.index.is_month_end.astype(int)
    ml["Is_Quarter_End"] = ml.index.is_quarter_end.astype(int)

    features = [
        "Lag_1", "Lag_2", "Lag_3", "Lag_5", "Lag_10",
        "MA3", "MA5", "MA10", "MA20", "MA60", "STD5",
        "Spread", "Return", "Momentum3", "Momentum5", "RSI14", "MACD", "MACD_Signal",
        "DXY_lag1", "DXY_delta_lag1", "US10Y_lag1", "US10Y_delta_lag1",
        "VIX_delta_lag1", "KOSPI_return_lag1",
        "Weekday", "Is_Month_End", "Is_Quarter_End",
    ]
    ml = ml.dropna()

    ml_step1 = ml.copy()
    ml_step1["Target_Step1"] = ml_step1["Close"].shift(-1) - ml_step1["Close"]
    ml_step1 = ml_step1.dropna(subset=["Target_Step1"])

    # [3] OOT 검증 ----------------------------------------------------
    split_dt = ml_step1.index[-1] - pd.DateOffset(months=1)
    split_date = split_dt.strftime("%Y-%m-%d")
    test_start_date = (split_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _log(f"OOT 동적 검증 (Train ~{split_date} | Test {test_start_date}~)")

    train_oot = ml_step1.loc[:split_date]
    test_oot = ml_step1.loc[test_start_date:]
    X_train_oot, y_train_oot = train_oot[features], train_oot["Target_Step1"]
    X_test_oot, y_test_oot = test_oot[features], test_oot["Target_Step1"]

    oot_model = XGBRegressor(**BEST_PARAMS, random_state=42).fit(X_train_oot, y_train_oot)
    xgb_preds_oot_delta = oot_model.predict(X_test_oot)
    pred_close_xgb = test_oot["Close"] + xgb_preds_oot_delta
    true_close_xgb = test_oot["Close"] + y_test_oot
    xgb_mae, xgb_rmse = evaluate_model(true_close_xgb, pred_close_xgb)

    arima_train = ml.loc[:split_date, "Close"].asfreq("B").ffill()
    arima_test = ml.loc[test_start_date:, "Close"].asfreq("B").ffill()
    arima_eval_model = ARIMA(arima_train, order=(0, 1, 1)).fit()
    arima_eval_preds = arima_eval_model.forecast(steps=len(arima_test))
    arima_mae, arima_rmse = evaluate_model(arima_test, arima_eval_preds)

    prophet_train = ml.loc[:split_date].reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_train["ds"] = prophet_train["ds"].dt.tz_localize(None)
    prophet_test = ml.loc[test_start_date:].reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_test["ds"] = prophet_test["ds"].dt.tz_localize(None)
    prophet_eval_model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                                 daily_seasonality=False, changepoint_prior_scale=0.01).fit(prophet_train)
    prophet_eval_preds = prophet_eval_model.predict(prophet_test[["ds"]])["yhat"].values
    prophet_mae, prophet_rmse = evaluate_model(prophet_test["y"], prophet_eval_preds)

    # [4] 미래 예측 ---------------------------------------------------
    _log(f"미래 {forecast_days}영업일 예측 생성...")
    automated_today_price = float(ml["Close"].iloc[-1])
    latest_features = ml[features].iloc[[-1]].copy()
    forecast_results = []
    for step in range(1, forecast_days + 1):
        target_name = f"Target_Delta_Step{step}"
        ml[target_name] = ml["Close"].shift(-step) - ml["Close"]
        train_df = ml.dropna(subset=features + [target_name])
        X_train, y_train = train_df[features], train_df[target_name]
        model_step = XGBRegressor(**BEST_PARAMS, random_state=42).fit(X_train, y_train)
        pred_cumulative_delta = float(model_step.predict(latest_features)[0])
        forecast_results.append(automated_today_price + pred_cumulative_delta)

    future_bdays = pd.bdate_range(start=ml.index[-1] + pd.offsets.BDay(1), periods=forecast_days)
    forecast_dates = future_bdays.strftime("%Y-%m-%d").tolist()
    forecast_rounded = [round(p, 2) for p in forecast_results]

    # [5] 앙상블 ------------------------------------------------------
    arima_series = ml["Close"].asfreq("B").ffill()
    arima_fit = ARIMA(arima_series, order=(0, 1, 1)).fit()
    arima_forecast = arima_fit.forecast(steps=forecast_days).values

    prophet_df = ml.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                            daily_seasonality=False, changepoint_prior_scale=0.01).fit(prophet_df)
    future_dates_df = pd.DataFrame({"ds": pd.to_datetime(forecast_dates)})
    prophet_forecast = prophet_model.predict(future_dates_df)["yhat"].values

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

    # [6] 실시간 Re-anchoring -----------------------------------------
    global_model_baseline = last_close
    days_diff = (pd.Timestamp.today().normalize() - pd.to_datetime(target_date)).days
    crawl_msg = ""
    if (days_diff > 3) or (not use_realtime):
        reason = "백테스트 모드(과거 날짜)" if days_diff > 3 else "실시간 보정 비활성화"
        crawl_msg = f"{reason} → 모델 종가 사용"
        current_rate = global_model_baseline
        crawl_ok = False
    else:
        rate, msg = fetch_naver_usdkrw()
        if rate is not None:
            current_rate = rate
            crawl_ok = True
            crawl_msg = f"실시간 수집 성공: {current_rate}원 ({msg})"
        else:
            current_rate = global_model_baseline
            crawl_ok = False
            crawl_msg = f"실시간 수집 실패 → 모델 종가 대체 ({msg})"
    _log(crawl_msg)

    market_basis_gap = (current_rate - global_model_baseline) * ANCHOR_RATE
    xgb_forecast_adj = xgb_forecast + (market_basis_gap * decay_vector)
    arima_forecast_adj = arima_forecast + (market_basis_gap * decay_vector)
    prophet_forecast_adj = prophet_forecast + (market_basis_gap * decay_vector)
    ensemble_forecast_adj = ensemble_forecast + (market_basis_gap * decay_vector)

    # [7] 시그널 (최근 20일 실제 밴드) --------------------------------
    recent20 = ml["Close"].tail(20)
    band_center = float(recent20.mean())
    band_std = float(recent20.std())
    lower_band = band_center - BAND_SIGMA * band_std
    upper_band = band_center + BAND_SIGMA * band_std
    band_width = upper_band - lower_band
    percent_b = (current_rate - lower_band) / band_width if band_width > 0 else 0.5
    signals = [make_signal(p, lower_band, upper_band) for p in ensemble_forecast_adj]
    today_signal = make_signal(current_rate, lower_band, upper_band)

    forecast_table = pd.DataFrame({
        "Date": forecast_dates,
        "XGBoost": np.round(xgb_forecast_adj, 2),
        "ARIMA": np.round(arima_forecast_adj, 2),
        "Prophet": np.round(prophet_forecast_adj, 2),
        "Final": np.round(ensemble_forecast_adj, 2),
        "Buy_Band": round(lower_band, 2),
        "Sell_Band": round(upper_band, 2),
        "Signal": signals,
    })

    # [9] OOT 앙상블 검증 (교집합 정렬) -------------------------------
    prophet_series = pd.Series(prophet_eval_preds, index=pd.to_datetime(prophet_test["ds"].values))
    arima_series_eval = pd.Series(np.asarray(arima_eval_preds), index=pd.to_datetime(arima_test.index))
    xgb_series = pred_close_xgb.copy()
    xgb_series.index = pd.to_datetime(xgb_series.index).tz_localize(None)
    true_series = true_close_xgb.copy()
    true_series.index = pd.to_datetime(true_series.index).tz_localize(None)
    if prophet_series.index.tz is not None:
        prophet_series.index = prophet_series.index.tz_localize(None)
    if arima_series_eval.index.tz is not None:
        arima_series_eval.index = arima_series_eval.index.tz_localize(None)

    common_idx = xgb_series.index.intersection(arima_series_eval.index).intersection(prophet_series.index)
    common_idx = common_idx.intersection(true_series.index)

    metrics = {}
    if len(common_idx) > 0:
        aligned_true = true_series.loc[common_idx]
        aligned_xgb = xgb_series.loc[common_idx]
        aligned_arima = arima_series_eval.loc[common_idx]
        aligned_prophet = prophet_series.loc[common_idx]
        ensemble_preds_oot = (aligned_xgb * weight_xgb) + (aligned_arima * weight_arima) + (aligned_prophet * weight_prophet)
        e_mae, e_rmse = evaluate_model(aligned_true, ensemble_preds_oot)
        e_mape = mape_score(aligned_true, ensemble_preds_oot)
        metrics = {
            "sample_days": int(len(common_idx)),
            "mae": round(e_mae, 2),
            "rmse": round(e_rmse, 2),
            "mape": round(e_mape, 2),
            "accuracy": round(100 - e_mape, 2),
            "directional_accuracy": round(directional_accuracy(aligned_true, ensemble_preds_oot), 1),
        }

    model_names = ["XGBoost", "ARIMA", "Prophet"]
    best_model_idx = int(np.argmin(scores))

    return {
        "target_date": target_date,
        "forecast_days": forecast_days,
        "adjusted_note": adjusted_note,
        "current_rate": round(float(current_rate), 2),
        "crawl_ok": crawl_ok,
        "crawl_msg": crawl_msg,
        "today_signal": today_signal,
        "percent_b": round(percent_b * 100, 1),
        "band_center": round(band_center, 2),
        "lower_band": round(lower_band, 2),
        "upper_band": round(upper_band, 2),
        "best_model": model_names[best_model_idx],
        "best_score": round(float(scores[best_model_idx]), 2),
        "weights": {
            "XGBoost": round(float(weight_xgb) * 100, 1),
            "ARIMA": round(float(weight_arima) * 100, 1),
            "Prophet": round(float(weight_prophet) * 100, 1),
        },
        "forecast_table": forecast_table,
        "forecast_dates": forecast_dates,
        "xgb_forecast": np.round(xgb_forecast_adj, 2).tolist(),
        "arima_forecast": np.round(arima_forecast_adj, 2).tolist(),
        "prophet_forecast": np.round(prophet_forecast_adj, 2).tolist(),
        "ensemble_forecast": np.round(ensemble_forecast_adj, 2).tolist(),
        "history": ml["Close"].tail(30),   # 최근 30일 실제 종가 (그래프용)
        "metrics": metrics,
    }


# ---------------------------------------------------
# 단독 실행 시 콘솔 테스트
# ---------------------------------------------------
if __name__ == "__main__":
    result = run_forecast(target_date="2026-09-28", forecast_days=5, log=print)
    print("\n=== 예측 결과표 ===")
    print(result["forecast_table"].to_string(index=False))
    print("\n=== 성능 지표 ===")
    print(result["metrics"])
    print(f"\n오늘 시그널: {result['today_signal']} (%B={result['percent_b']}%)")
