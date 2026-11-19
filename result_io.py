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
