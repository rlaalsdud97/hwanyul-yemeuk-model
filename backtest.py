# ==============================================================================
# backtest.py — 방향 예측 1년 검증 (walk-forward)
#   - 최근 N 영업일 각각에 대해 "그날까지의 데이터로만" 학습 → 다음 날 방향 예측 → 채점
#   - 여러 모델/피처 조합과 단순 기준선을 같은 잣대로 비교합니다.
#   - 실행:  python backtest.py            (기본 250일)
#            python backtest.py 120         (검증 일수 지정)
#   - 결과: 콘솔 표 + data/backtest.json (대시보드 표시용)
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
OUT_PATH = Path(__file__).parent / "data" / "backtest.json"
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


def walk_forward(df, feats, make_model, test_n, window=None):
    """각 날짜 i: i 이전 행으로 학습 → i 행의 다음날 방향 확률 예측."""
    probas = []
    for i in range(len(df) - test_n, len(df)):
        lo = 0 if window is None else max(0, i - window)
        tr = df.iloc[lo:i]
        m = make_model(tr["Up"]).fit(tr[feats], tr["Up"])
        probas.append(float(m.predict_proba(df[feats].iloc[[i]])[0][1]))
    return np.array(probas)


def score(pred_up, true_up, proba=None):
    pred_up = np.asarray(pred_up, dtype=bool)
    true_up = np.asarray(true_up, dtype=bool)
    hit = pred_up == true_up
    n = len(hit)
    p = hit.mean()
    half = n // 2
    out = {
        "n": int(n),
        "hit": round(float(p * 100), 1),
        "ci95": round(float(1.96 * np.sqrt(p * (1 - p) / n) * 100), 1),
        "first_half": round(float(hit[:half].mean() * 100), 1),
        "second_half": round(float(hit[half:].mean() * 100), 1),
    }
    if proba is not None:
        conf = np.abs(proba - 0.5) >= CONF
        out["conf_cover"] = round(float(conf.mean() * 100), 1)
        out["conf_n"] = int(conf.sum())
        out["conf_hit"] = round(float(hit[conf].mean() * 100), 1) if conf.sum() else None
    return out


def main(test_n=250):
    t0 = time.time()
    today = datetime.now(KST).strftime("%Y-%m-%d")
    ml, features, target_date, _ = load_dataset(today, start="2019-01-01")

    nxt = ml["Close"].shift(-1)
    ml["Up"] = (nxt > ml["Close"]).astype(float).where(nxt.notna())
    df = ml[(nxt != ml["Close"])].dropna(subset=["Up"]).copy()  # 보합(변동 0) 날 제외
    df["Up"] = df["Up"].astype(int)
    test_n = min(test_n, len(df) - 300)
    test = df.iloc[-test_n:]
    y = test["Up"].values
    print(f"[backtest] 데이터 {len(df)}일 · 검증 {test_n}일 "
          f"({test.index[0]:%Y-%m-%d} ~ {test.index[-1]:%Y-%m-%d})")

    results = {}

    # --- 기준선 -----------------------------------------------------------
    results["기준선: 항상 상승"] = score(np.ones(test_n), y)
    prev_up = (df["Ret1"] > 0).values[-test_n:]
    results["기준선: 오늘 방향 그대로"] = score(prev_up, y)
    results["기준선: 오늘 반대로"] = score(~prev_up, y)

    # --- 모델 -------------------------------------------------------------
    configs = {
        "A. 현재 모델 (수준 피처 + XGB)": (features, _xgb(CLF_PARAMS), None),
        "B. 변화량 피처 + XGB": (DIR_FEATURES, _xgb(CLF_PARAMS), None),
        "C. 변화량 피처 + 단순 XGB": (DIR_FEATURES, _xgb(SIMPLE_PARAMS), None),
        "D. 변화량 피처 + 로지스틱": (DIR_FEATURES, _logit, None),
        "E. 변화량 + 단순 XGB (최근 2년만 학습)": (DIR_FEATURES, _xgb(SIMPLE_PARAMS), 500),
    }
    probas = {}
    for name, (feats, maker, window) in configs.items():
        t = time.time()
        pr = walk_forward(df, feats, maker, test_n, window)
        probas[name] = pr
        results[name] = score(pr >= 0.5, y, pr)
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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "test_days": int(test_n),
        "period": [f"{test.index[0]:%Y-%m-%d}", f"{test.index[-1]:%Y-%m-%d}"],
        "conf_threshold": CONF,
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
