# 1) 야후 일봉 '종가'가 하루 중 몇 시(UTC) 가격인지 확인
# 2) F의 효과가 어떤 '당일' 피처에서 오는지 분리 실험
import sys, numpy as np, pandas as pd, yfinance as yf
pd.set_option("display.width", 200)
print("===== 일봉 종가 시각 확인 =====")
for tk in ["KRW=X", "TWD=X", "CNY=X", "EWY", "^GSPC"]:
    d = yf.download(tk, period="1mo", interval="1d", progress=False)["Close"][tk].dropna()
    h = yf.download(tk, period="1mo", interval="1h", progress=False)["Close"][tk].dropna()
    print(f"\n[{tk}] 일봉 index 예시: {d.index[-3:].tolist()}")
    print(f"[{tk}] 시간봉 tz: {h.index.tz}")
    hu = h.copy(); hu.index = hu.index.tz_convert("UTC")
    for day, val in list(d.items())[-6:-1]:
        dd = pd.Timestamp(day).date()
        same = hu[(hu.index.date >= dd) & (hu.index.date <= dd + pd.Timedelta(days=1))]
        if same.empty:
            print(f"  {dd}: 시간봉 없음"); continue
        best = (same - val).abs().idxmin()
        print(f"  {dd}: 일봉종가 {val:.4f} ≈ 시간봉 {best} (UTC) {same[best]:.4f} | 그날 마지막 시간봉 {hu[hu.index.date == dd].index.max()}")

print("\n===== 분리 실험 (내일 방향, 250일) =====")
import backtest as bt
from forecast_core import DIR_FEATURES, load_dataset
from datetime import datetime
ml, features, _, _ = load_dataset(datetime.now().strftime("%Y-%m-%d"), start="2019-01-01")
same, lag = bt.add_extra_features(ml)
ml = ml.dropna(subset=same + lag)
nxt = ml["Close"].shift(-1)
ml["Up"] = (nxt > ml["Close"]).astype(float).where(nxt.notna())
df = ml[(nxt != ml["Close"])].dropna(subset=["Up"]).copy(); df["Up"] = df["Up"].astype(int)
y = df["Up"].values[-250:]
us = [c for c in same if c.split("_")[0] in ("EWY", "SOX", "SPX", "US2Y")]
fx = [c for c in same if c not in us]
tests = {
    "F 전체": DIR_FEATURES + same + lag,
    "F - 미국당일(EWY·SOX·SPX·US2Y)만": DIR_FEATURES + us + lag,
    "F - 주변통화당일(CNY·JPY·TWD·KRW_vs)만": DIR_FEATURES + fx + lag,
}
for c in same:
    tests[f"단일: {c}"] = DIR_FEATURES + [c]
for name, feats in tests.items():
    pr = bt.walk_forward(df, feats, bt._xgb(bt.SIMPLE_PARAMS), 250)
    r = bt.score(pr >= 0.5, y, pr)
    print(f"{name:45s} {r['hit']:5.1f}%  (전반 {r['first_half']} / 후반 {r['second_half']})")
