# ==============================================================================
# fixed_time_check.py — "야후 종가"와 "매일 같은 시각 가격"으로 결과가 같은지 확인
#
#   배경: 야후 환율 일봉 종가는 날마다 다른 시각의 가격이다 (2026-11-19 확인).
#         이런 데이터는 실제로는 없는 '올랐다 내렸다(되돌림)' 패턴을 만들 수 있고,
#         모델이 그 가짜 패턴을 맞혀 적중률이 부풀려질 수 있다.
#   방법: 최근 약 2년 시간봉에서 매일 한국시간 15:00(UTC 06:00) 가격만 뽑아
#         '고정 시각 일봉'을 만들고, 같은 기간·같은 모델(A)로 두 데이터를 비교한다.
#   실행: python fixed_time_check.py [USD EUR]  → data/fixed_check(_eur).json
#   결과(2026-11-19 첫 실행): 야후 종가에선 일간수익률 자기상관이 USD -0.17, EUR -0.32로
#   강한 가짜 되돌림이 있었고, 고정 시각 가격에서는 거의 0. 모델 적중률도 EUR 56.8% → 48.8%.
#   → 대시보드의 방향 검증은 이 '고정 시각' 결과를 기준으로 판정한다.
# ==============================================================================
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt
import forecast_core as fc
from result_io import CURRENCIES, data_path

KST = timezone(timedelta(hours=9))
SNAP_UTC = 6          # 한국시간 15:00
TEST_N = 250

_orig_download = yf.download


def fixed_time_bars(ticker):
    """시간봉 → 매일 SNAP_UTC 시각 가격 (High/Low는 직전 24시간 최고/최저)."""
    h = _orig_download(ticker, period="730d", interval="1h", progress=False, threads=False)
    c, hi, lo = h["Close"][ticker], h["High"][ticker], h["Low"][ticker]
    for s in (c, hi, lo):
        s.index = s.index.tz_convert("UTC")
    # 05:00 시작 시간봉의 종가 = 06:00 시점 가격
    mask = c.index.hour == SNAP_UTC - 1
    out = pd.DataFrame({
        "Close": c[mask],
        "High": hi.rolling("24h").max()[mask],
        "Low": lo.rolling("24h").min()[mask],
    }).dropna()
    out.index = out.index.normalize().tz_localize(None)
    return out[out.index.dayofweek < 5]


def load(currency, fixed):
    """load_dataset을 그대로 쓰되, fixed=True면 환율 Close/High/Low를 고정 시각 값으로 교체."""
    fx = CURRENCIES[currency]["ticker"]
    bars = fixed_time_bars(fx) if fixed else None

    def patched(tickers, start, end=None, **k):
        raw = _orig_download(tickers, start=start, end=end, **k)
        if bars is not None:
            raw = raw.loc[raw.index.isin(bars.index)].copy()
            for col in ("Close", "High", "Low"):
                raw[(col, fx)] = bars[col].reindex(raw.index).values
        return raw

    fc.yf.download = patched
    try:
        today = datetime.now(KST).strftime("%Y-%m-%d")
        start = (datetime.now(KST) - timedelta(days=740)).strftime("%Y-%m-%d")
        ml, features, _, _ = fc.load_dataset(today, start=start, currency=currency)
    finally:
        fc.yf.download = _orig_download
    return ml, features


def label(ml):
    nxt = ml["Close"].shift(-1)
    ml = ml.copy()
    ml["Up"] = (nxt > ml["Close"]).astype(float).where(nxt.notna())
    df = ml[(nxt != ml["Close"])].dropna(subset=["Up"]).copy()
    df["Up"] = df["Up"].astype(int)
    return df


def evaluate(currency):
    print(f"\n==================== {CURRENCIES[currency]['pair']} ====================")
    ml_y, feats = load(currency, fixed=False)
    ml_f, _ = load(currency, fixed=True)
    common = ml_y.index.intersection(ml_f.index)         # 같은 날짜만 비교
    dfs = {"야후 종가": label(ml_y.loc[common]), "고정 15:00": label(ml_f.loc[common])}
    common_t = dfs["야후 종가"].index.intersection(dfs["고정 15:00"].index)
    test_n = min(TEST_N, len(common_t) - 150)
    print(f"공통 {len(common)}일 · 검증 {test_n}일 ({common_t[-test_n]:%Y-%m-%d} ~ {common_t[-1]:%Y-%m-%d})")

    rows, saved = [], {}
    for name, df in dfs.items():
        df = df.loc[df.index.isin(common_t)]
        r1 = df["Close"].pct_change()
        y = df["Up"].values[-test_n:]
        prev_up = (df["Ret1"] > 0).values[-test_n:]
        rev = bt.score(~prev_up, y)
        pr = bt.walk_forward(df, feats, bt._xgb(fc.CLF_PARAMS), test_n)
        a = bt.score(pr >= 0.5, y, pr)
        rows.append({
            "데이터": name,
            "일간수익률 자기상관": round(float(r1.autocorr(1)), 3),
            "오늘 반대로%": rev["hit"],
            "현재 모델 A%": a["hit"], "±95%": a["ci95"],
            "전반/후반": f"{a['first_half']} / {a['second_half']}",
        })
        saved[name] = {"autocorr": round(float(r1.autocorr(1)), 3),
                       "reversal_hit": rev["hit"], "model": a}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))
    path = data_path("fixed_check", currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "currency": currency, "snap_kst": f"{SNAP_UTC + 9:02d}:00", "test_days": int(test_n),
        "period": [f"{common_t[-test_n]:%Y-%m-%d}", f"{common_t[-1]:%Y-%m-%d}"],
        "results": saved,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    for cur in (sys.argv[1:] or ["USD", "EUR"]):
        evaluate(cur.upper())
    print("\n※ 자기상관이 크게 음수(-0.1 이하 등)면 '되돌림'이 데이터 착시일 가능성이 큼.")
    print("※ 고정 15:00에서도 A가 50%를 확실히 넘으면 진짜 신호일 가능성이 높아짐.")
