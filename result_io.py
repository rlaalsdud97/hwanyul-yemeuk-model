# ==============================================================================
# result_io.py
# run_forecast() 결과(dict)를 JSON 파일로 저장하고 다시 읽어오는 모듈.
#   - 매일 아침 GitHub Actions가 예측을 미리 돌려 data/latest.json 에 저장하고,
#     대시보드(app.py)는 그 파일을 읽어서 바로 보여줍니다 (학습 없음).
#   - pickle 대신 JSON을 쓰는 이유: 저장한 곳(GitHub Actions)과 읽는 곳
#     (Streamlit Cloud)의 pandas 버전이 달라도 안전하게 읽히도록.
#   - 무거운 라이브러리(xgboost 등)를 import하지 않아서 앱이 빨리 뜹니다.
# ==============================================================================
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LATEST_PATH = Path(__file__).parent / "data" / "latest.json"


def _clean(v):
    """numpy 타입 · NaN 을 JSON에 넣을 수 있는 값으로 변환."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    return v


def _df_to_records(df):
    if df is None:
        return None
    return _clean(df.to_dict(orient="records"))


def save_result(result, path=LATEST_PATH, extra=None):
    """run_forecast() 결과를 JSON으로 저장."""
    out = dict(result)
    out["forecast_table"] = _df_to_records(result["forecast_table"])
    hist = result["history"]
    out["history"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in pd.to_datetime(hist.index)],
        "values": [float(x) for x in hist.values],
    }
    direction = dict(result.get("direction") or {})
    direction["backtest_table"] = _df_to_records(direction.get("backtest_table"))
    out["direction"] = direction
    if extra:
        out.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_result(path=LATEST_PATH):
    """저장된 JSON을 run_forecast() 결과와 같은 모양(dict)으로 복원. 없으면 None."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    d["forecast_table"] = pd.DataFrame(d["forecast_table"])
    h = d["history"]
    d["history"] = pd.Series(h["values"], index=pd.to_datetime(h["dates"]))
    bt = (d.get("direction") or {}).get("backtest_table")
    if d.get("direction") is not None:
        d["direction"]["backtest_table"] = pd.DataFrame(bt) if bt else None
    return d


# ==============================================================================
# 📒 실전 예측 기록 (data/history.csv)
#   - 매일 자동 예측값을 한 줄씩 쌓고, 실제 종가가 나오면 채점합니다.
#   - 백테스트(과거 재계산)가 아니라 '그날 실제로 낸 예측'의 성적이라 가장 정직한 지표.
# ==============================================================================
HISTORY_PATH = Path(__file__).parent / "data" / "history.csv"

HISTORY_COLS = [
    "기준일", "기준가", "예측대상일", "몇일후", "예측값", "분류_상승확률",
    "실제값", "오차", "예측방향", "실제방향", "방향적중",
]


def _dir(diff):
    if diff is None or pd.isna(diff):
        return None
    return "▲" if diff > 0 else ("▼" if diff < 0 else "-")


def update_history(result, path=HISTORY_PATH):
    """오늘 예측을 기록에 추가(같은 기준일은 교체)하고, 실제값이 나온 과거 예측을 채점."""
    path = Path(path)
    if path.exists():
        hist = pd.read_csv(path, encoding="utf-8-sig", dtype={"기준일": str, "예측대상일": str})
    else:
        hist = pd.DataFrame(columns=HISTORY_COLS)

    closes = result["history"]  # 최근 30일 실제 종가 (index=날짜)
    base_date = str(result["target_date"])
    base = float(closes.iloc[-1])
    up_proba = (result.get("direction") or {}).get("up_proba")

    rows = []
    for h, (d, v) in enumerate(zip(result["forecast_dates"], result["ensemble_forecast"]), start=1):
        rows.append({
            "기준일": base_date, "기준가": round(base, 2), "예측대상일": d, "몇일후": h,
            "예측값": round(float(v), 2),
            "분류_상승확률": up_proba if h == 1 else None,
        })
    hist = hist[hist["기준일"] != base_date]
    hist = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    # 빈 열은 숫자형(float)으로 읽혀서 '▲' 같은 글자를 못 넣으므로 형식을 명시
    for c in ["기준가", "예측값", "분류_상승확률", "실제값", "오차"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    for c in ["예측방향", "실제방향", "방향적중"]:
        hist[c] = hist[c].astype(object)
    hist["몇일후"] = pd.to_numeric(hist["몇일후"]).astype(int)

    # 채점: 실제 종가가 확보된 예측대상일만
    actual = {d.strftime("%Y-%m-%d"): float(x) for d, x in closes.items()}
    for i, r in hist.iterrows():
        a = actual.get(str(r["예측대상일"]))
        if a is None and pd.notna(r.get("실제값")):
            continue  # 이미 채점된 행 유지
        if a is None:
            continue
        hist.at[i, "실제값"] = round(a, 2)
        hist.at[i, "오차"] = round(float(r["예측값"]) - a, 2)
        p_dir = _dir(float(r["예측값"]) - float(r["기준가"]))
        a_dir = _dir(a - float(r["기준가"]))
        hist.at[i, "예측방향"] = p_dir
        hist.at[i, "실제방향"] = a_dir
        hist.at[i, "방향적중"] = None if a_dir == "-" else bool(p_dir == a_dir)

    hist = hist.sort_values(["기준일", "몇일후"]).reindex(columns=HISTORY_COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(path, index=False, encoding="utf-8-sig")  # 엑셀에서 한글 안 깨지게
    return hist


def load_history(path=HISTORY_PATH):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"기준일": str, "예측대상일": str})


def summarize_history(hist):
    """채점된 예측의 성적 요약. 채점된 게 없으면 None."""
    if hist is None or hist.empty:
        return None
    done = hist[hist["실제값"].notna()]
    if done.empty:
        return None
    d1 = done[done["몇일후"] == 1]
    hit = d1["방향적중"].dropna().astype(str).str.lower() == "true"
    clf = d1.dropna(subset=["분류_상승확률"])
    clf = clf[clf["실제방향"].isin(["▲", "▼"])]
    clf_hit = ((clf["분류_상승확률"] >= 50) == (clf["실제방향"] == "▲"))
    return {
        "days": int(done["기준일"].nunique()),
        "n": int(len(done)),
        "mae": round(float(done["오차"].abs().mean()), 2),
        "mae_d1": round(float(d1["오차"].abs().mean()), 2) if len(d1) else None,
        "dir_n": int(len(hit)),
        "dir_rate": round(float(hit.mean() * 100), 1) if len(hit) else None,
        "clf_n": int(len(clf_hit)),
        "clf_rate": round(float(clf_hit.mean() * 100), 1) if len(clf_hit) else None,
    }
