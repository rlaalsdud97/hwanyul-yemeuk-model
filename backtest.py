# ==============================================================================
# backtest.py — 방향 예측 1년 검증 (walk-forward)
#   - 최근 N 영업일 각각에 대해 "그날까지의 데이터로만" 학습 → 다음 날 방향 예측 → 채점
#   - 여러 모델/피처 조합과 단순 기준선을 같은 잣대로 비교합니다.
#   - 실행:  python backtest.py              (내일 방향, 250일)
#            python backtest.py 500 5        (5영업일 뒤 방향, 500일)
#   - 결과: 콘솔 표 + data/backtest.json (h=1) / data/backtest_h5.json (h=5)
#   - h일 뒤 방향은 정답이 h일 뒤에야 확정되므로, 학습에는 그 시점에 정답이
#     확정된 행만 씁니다 (미래정보 누수 방지). 또 검증 표본이 서로 겹치므로
#     오차범위는 '유효 표본 = 일수 / h' 로 넓혀서 계산합니다.
# ==============================================================================
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from forecast_core import CLF_PARAMS, DIR_FEATURES, load_dataset

KST = timezone(timedelta(hours=9))
DATA_DIR = Path(__file__).parent / "data"
CONF = 0.10  # 확신 구간: 상승확률이 60% 이상 또는 40% 이하인 날만


def _xgb(params):
    def make(y):
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        return XGBClassifier(**params, random_state=42, scale_pos_weight=neg / pos,
                             eval_metric="logloss", n_jobs=2)
    return make


def _logit(y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))


SIMPLE_PARAMS = {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.03,
                 "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}


# ------------------------------------------------------------------
# 외부 수급·금리 데이터 (실험용)
#   EWY  : 미국 상장 한국 ETF → 외국인이 한국을 어떻게 보는지 (외국인 수급 대용)
#   SOX  : 미국 반도체 지수 → 한국 수출·외국인 매매의 큰 축
#   SPX  : S&P500 (위험선호)
#   CNY/JPY/TWD : 주변국 통화 (원화와 같이 움직이는 경우 많음)
#   US2Y : 미국 2년물 금리 (FRED) → 한미 금리차의 미국 쪽
#   ※ '당일' 값은 쓰지 않는다 (2026-11-19 확인):
#     야후 원달러(KRW=X) 일봉 종가는 대개 한국 시간 낮(UTC 00~09시)의 가격이라,
#     그날 밤 미국 장(EWY 등)은 환율 '종가 이후'의 일이다. 당일 EWY를 넣으면
#     적중률이 62.8%까지 오르지만, 이미 밤사이 반영된 움직임을 맞히는 것이라
#     실제 매수 결정에는 쓸 수 없다 (전일 값만 쓰면 46%). 그래서 G만 남김.
# ------------------------------------------------------------------
EXTRA_TICKERS = {"EWY": "EWY", "SOX": "^SOX", "SPX": "^GSPC",
                 "CNY": "CNY=X", "JPY": "JPY=X", "TWD": "TWD=X"}


def add_extra_features(ml, start="2019-01-01"):
    import yfinance as yf
    import requests, io
    idx = ml.index
    raw = yf.download(list(EXTRA_TICKERS.values()), start=start, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    same, lag = [], []
    for name, tk in EXTRA_TICKERS.items():
        if tk not in raw.columns or raw[tk].notna().sum() < 500:
            print(f"  (건너뜀: {name} 데이터 부족)")
            continue
        px = raw[tk].reindex(idx, method="ffill")
        ml[f"{name}_ret"] = px.pct_change() * 100
        ml[f"{name}_ret_lag1"] = ml[f"{name}_ret"].shift(1)
        same.append(f"{name}_ret"); lag.append(f"{name}_ret_lag1")
    # 원화가 주변 통화보다 더/덜 움직였는지 (상대 움직임)
    for peer in ["TWD", "CNY"]:
        if f"{peer}_ret" in ml:
            ml[f"KRW_vs_{peer}"] = ml["Ret1"] - ml[f"{peer}_ret"]
            ml[f"KRW_vs_{peer}_lag1"] = ml[f"KRW_vs_{peer}"].shift(1)
            same.append(f"KRW_vs_{peer}"); lag.append(f"KRW_vs_{peer}_lag1")
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2", timeout=30)
        f = pd.read_csv(io.StringIO(r.text))
        f.columns = ["date", "v"]
        f["v"] = pd.to_numeric(f["v"], errors="coerce")
        us2 = f.dropna().set_index(pd.to_datetime(f.dropna()["date"]))["v"]
        us2 = us2.reindex(idx, method="ffill")
        ml["US2Y_chg"] = us2.diff()
        ml["US2Y_chg_lag1"] = ml["US2Y_chg"].shift(1)
        ml["US2Y_10Y_slope"] = ml["US10Y"] - us2   # 장단기 금리차
        same += ["US2Y_chg", "US2Y_10Y_slope"]; lag += ["US2Y_chg_lag1"]
    except Exception as e:
        print(f"  (건너뜀: FRED 미국 2년물 — {e})")
    return same, lag


def walk_forward(df, feats, make_model, test_n, window=None, horizon=1):
    """각 날짜 i: 그 시점에 정답이 확정된 행(i-horizon 까지)으로 학습 → i 행 방향 확률 예측."""
    probas = []
    for i in range(len(df) - test_n, len(df)):
        end = i - horizon + 1
        lo = 0 if window is None else max(0, end - window)
        tr = df.iloc[lo:end]
        m = make_model(tr["Up"]).fit(tr[feats], tr["Up"])
        probas.append(float(m.predict_proba(df[feats].iloc[[i]])[0][1]))
    return np.array(probas)


def score(pred_up, true_up, proba=None, horizon=1):
    pred_up = np.asarray(pred_up, dtype=bool)
    true_up = np.asarray(true_up, dtype=bool)
    hit = pred_up == true_up
    n = len(hit)
    p = hit.mean()
    half = n // 2
    n_eff = max(n / horizon, 1)  # 겹치는 표본 보정
    out = {
        "n": int(n),
        "hit": round(float(p * 100), 1),
        "ci95": round(float(1.96 * np.sqrt(p * (1 - p) / n_eff) * 100), 1),
        "first_half": round(float(hit[:half].mean() * 100), 1),
        "second_half": round(float(hit[half:].mean() * 100), 1),
    }
    if proba is not None:
        conf = np.abs(proba - 0.5) >= CONF
        out["conf_cover"] = round(float(conf.mean() * 100), 1)
        out["conf_n"] = int(conf.sum())
        out["conf_hit"] = round(float(hit[conf].mean() * 100), 1) if conf.sum() else None
    return out


def main(test_n=250, horizon=1):
    t0 = time.time()
    today = datetime.now(KST).strftime("%Y-%m-%d")
    ml, features, target_date, _ = load_dataset(today, start="2019-01-01")
    extra_same, extra_lag = add_extra_features(ml)
    ml = ml.dropna(subset=extra_same + extra_lag)
    print(f"[backtest] 추가 피처 {len(extra_same)}+{len(extra_lag)}개: {extra_same}")

    nxt = ml["Close"].shift(-horizon)
    ml["Up"] = (nxt > ml["Close"]).astype(float).where(nxt.notna())
    df = ml[(nxt != ml["Close"])].dropna(subset=["Up"]).copy()  # 보합(변동 0) 날 제외
    df["Up"] = df["Up"].astype(int)
    test_n = min(test_n, len(df) - 300)
    test = df.iloc[-test_n:]
    y = test["Up"].values
    print(f"[backtest] {horizon}영업일 뒤 방향 · 데이터 {len(df)}일 · 검증 {test_n}일 "
          f"({test.index[0]:%Y-%m-%d} ~ {test.index[-1]:%Y-%m-%d})")

    results = {}

    # --- 기준선 -----------------------------------------------------------
    results["기준선: 항상 상승"] = score(np.ones(test_n), y, horizon=horizon)
    trend_col = "Ret1" if horizon == 1 else ("Ret5" if horizon <= 5 else "Ret10")
    prev_up = (df[trend_col] > 0).values[-test_n:]
    label = "오늘" if horizon == 1 else f"최근 {trend_col[3:]}일 추세"
    results[f"기준선: {label} 방향 그대로"] = score(prev_up, y, horizon=horizon)
    results[f"기준선: {label} 반대로"] = score(~prev_up, y, horizon=horizon)

    # --- 모델 -------------------------------------------------------------
    configs = {
        "A. 현재 모델 (수준 피처 + XGB)": (features, _xgb(CLF_PARAMS), None),
        "B. 변화량 피처 + XGB": (DIR_FEATURES, _xgb(CLF_PARAMS), None),
        "C. 변화량 피처 + 단순 XGB": (DIR_FEATURES, _xgb(SIMPLE_PARAMS), None),
        "D. 변화량 피처 + 로지스틱": (DIR_FEATURES, _logit, None),
        "E. 변화량 + 단순 XGB (최근 2년만 학습)": (DIR_FEATURES, _xgb(SIMPLE_PARAMS), 500),
        "G. 변화량 + 수급·금리(전일만) + 단순 XGB": (DIR_FEATURES + extra_lag, _xgb(SIMPLE_PARAMS), None),
    }
    probas = {}
    for name, (feats, maker, window) in configs.items():
        t = time.time()
        pr = walk_forward(df, feats, maker, test_n, window, horizon)
        probas[name] = pr
        results[name] = score(pr >= 0.5, y, pr, horizon)
        print(f"  - {name}: {results[name]['hit']}%  ({time.time() - t:.0f}초)")

    # --- 출력 -------------------------------------------------------------
    rows = []
    for name, r in results.items():
        rows.append({
            "모델": name, "적중률%": r["hit"], "±95%": r["ci95"],
            "전반기%": r["first_half"], "후반기%": r["second_half"],
            "확신일 적중%": r.get("conf_hit"), "확신일 비중%": r.get("conf_cover"),
        })
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + table.to_string(index=False))
    print(f"\n※ 검증 {test_n}일이면 우연만으로도 ±{results['기준선: 항상 상승']['ci95']}%p 정도 흔들립니다.")
    print(f"※ 확신일 = 상승확률 {50 + CONF * 100:.0f}% 이상 또는 {50 - CONF * 100:.0f}% 이하인 날")
    print(f"[backtest] 완료 ({time.time() - t0:.0f}초)")

    out_path = DATA_DIR / ("backtest.json" if horizon == 1 else f"backtest_h{horizon}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "horizon": int(horizon),
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "test_days": int(test_n),
        "period": [f"{test.index[0]:%Y-%m-%d}", f"{test.index[-1]:%Y-%m-%d}"],
        "conf_threshold": CONF,
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250,
         int(sys.argv[2]) if len(sys.argv) > 2 else 1)
