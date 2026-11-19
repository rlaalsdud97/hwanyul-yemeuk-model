# ==============================================================================
# buy_backtest.py — "BUY 시그널대로 사면 더 싸게 살까?" 검증
#
#   가정: 매달 같은 금액의 달러를 반드시 사야 하는 회사.
#   사는 방식(전략)별로 달마다 평균 매입 환율을 계산해 '매일 나눠 사기'와 비교합니다.
#
#   전략 (하루 기본량 = 그 달 금액 ÷ 그 달 영업일 수)
#     1) 매일 나눠 사기 (기준)   : 매일 기본량
#     2) 월초 한 번에 사기       : 그 달 첫 영업일에 전부
#     3) 시그널 따라 조절        : 🟢 BUY 2배 · 🟡 HOLD 1배 · 🔴 WAIT 안 삼
#     4) WAIT 날만 미루기        : 🔴 WAIT 날은 건너뛰고, 다음 WAIT 아닌 날에 밀린 만큼 같이 삼
#     ※ 월말 영업일에는 남은 금액을 전부 산다 (어차피 사야 하므로)
#     ※ 모든 결정은 '그날까지 아는 정보'로만 한다. 예: "이번 달 BUY 날이 며칠일지"는
#       미리 알 수 없으므로 그걸로 금액을 나누지 않는다 (미래정보를 쓰면 가짜 성과가 나옴).
#
#   시그널은 대시보드와 똑같이 '그날까지의 최근 20일 종가 ±1σ 밴드'로 판정하고,
#   실제 매입은 다음 영업일 종가로 체결된다고 봅니다 (아침 자동 예측을 보고 그날 사는 흐름).
#
#   실행: python buy_backtest.py  → 콘솔 표 + data/buy_backtest.json
# ==============================================================================
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_core import BAND_SIGMA, load_dataset

KST = timezone(timedelta(hours=9))
OUT_PATH = Path(__file__).parent / "data" / "buy_backtest.json"
START = "2020-01-01"


def band_signals(close):
    """대시보드 make_signal()과 같은 기준: 밴드 내 위치 ≤0.33 BUY, ≤0.66 HOLD, 그 외 WAIT."""
    m = close.rolling(20).mean()
    sd = close.rolling(20).std()
    low, up = m - BAND_SIGMA * sd, m + BAND_SIGMA * sd
    pos = (close - low) / (up - low)
    sig = pd.Series("HOLD", index=close.index)
    sig[pos <= 0.33] = "BUY"
    sig[pos > 0.66] = "WAIT"
    sig[m.isna()] = None
    return sig


def run_month(g, rule):
    """한 달 매입 시뮬레이션. rule(signal, q, backlog) → (오늘 살 양, 새 backlog).
    반환: 평균 매입 환율 (달러 수량 가중)."""
    n = len(g)
    q = 1.0 / n
    remaining, backlog = 1.0, 0.0
    spent = 0.0
    for i, (sig, price) in enumerate(zip(g["signal"], g["exec"])):
        if i == n - 1:
            amt = remaining                      # 월말: 남은 금액 전부
        else:
            amt, backlog = rule(sig, q, backlog)
            amt = min(amt, remaining)
        spent += amt * price
        remaining -= amt
    return spent  # 총 1달러어치를 샀으므로 = 평균 환율


def rule_daily(sig, q, b):
    return q, b


def rule_scale(sig, q, b):
    return {"BUY": 2 * q, "HOLD": q, "WAIT": 0.0}[sig], b


def rule_defer(sig, q, b):
    if sig == "WAIT":
        return 0.0, b + q
    return q + b, 0.0


def main():
    today = datetime.now(KST).strftime("%Y-%m-%d")
    ml, _, _, _ = load_dataset(today, start="2019-06-01")
    close = ml["Close"]

    df = pd.DataFrame({"close": close, "signal": band_signals(close),
                       "exec": close.shift(-1)})  # 다음 영업일 종가로 체결
    df = df.loc[START:].dropna()
    df["month"] = df.index.to_period("M")
    # 진행 중인 이번 달은 제외 (한 달을 다 채운 달만 비교)
    df = df[df["month"] < pd.Period(today, "M")]

    rows = []
    for mth, g in df.groupby("month"):
        rows.append({
            "month": str(mth),
            "days": len(g),
            "buy_days": int((g["signal"] == "BUY").sum()),
            "매일 나눠 사기": run_month(g, rule_daily),
            "월초 한 번에": float(g["exec"].iloc[0]),
            "시그널 따라 조절": run_month(g, rule_scale),
            "WAIT 날만 미루기": run_month(g, rule_defer),
        })
    m = pd.DataFrame(rows)
    base = m["매일 나눠 사기"]
    n = len(m)

    print(f"[buy] 기간 {m['month'].iloc[0]} ~ {m['month'].iloc[-1]} ({n}개월) · "
          f"체결 = 다음 영업일 종가")
    summary = {}
    lines = []
    for strat in ["월초 한 번에", "시그널 따라 조절", "WAIT 날만 미루기"]:
        diff = m[strat] - base                     # 음수 = 기준보다 싸게 삼
        pct = diff / base * 100
        ci = 1.96 * diff.std(ddof=1) / np.sqrt(n)
        by_year = diff.groupby(m["month"].str[:4]).mean().round(2).to_dict()
        summary[strat] = {
            "avg_diff_won": round(float(diff.mean()), 2),
            "ci95_won": round(float(ci), 2),
            "avg_diff_pct": round(float(pct.mean()), 3),
            "cheaper_months_pct": round(float((diff < 0).mean() * 100), 1),
            "by_year": by_year,
        }
        lines.append({
            "전략": strat,
            "평균 차이(원/달러)": round(float(diff.mean()), 2),
            "±95%": round(float(ci), 2),
            "차이(%)": round(float(pct.mean()), 3),
            "더 싸게 산 달%": round(float((diff < 0).mean() * 100), 1),
            "$100만당 절감(만원)": round(float(-diff.mean() * 1_000_000 / 10_000), 0),
        })
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + pd.DataFrame(lines).to_string(index=False))
        print("\n연도별 평균 차이(원/달러, 음수 = 더 싸게 삼)")
        print(pd.DataFrame({k: v["by_year"] for k, v in summary.items()}).to_string())
    print("※ 평균 차이 ± 범위가 0을 넘나들면 '효과가 있다고 말하기 어렵다'는 뜻입니다.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "period": [m["month"].iloc[0], m["month"].iloc[-1]],
        "months": int(n),
        "no_buy_months": int((m["buy_days"] == 0).sum()),
        "summary": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
