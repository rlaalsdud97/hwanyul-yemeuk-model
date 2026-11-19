# ==============================================================================
# daily_forecast.py — 매일 아침 자동 예측 (GitHub Actions에서 실행)
#   - 전날(한국시간)까지의 데이터로 예측해서 data/latest.json 에 저장합니다.
#   - 직접 실행도 가능:  python daily_forecast.py
#   - 실시간(네이버) 보정은 쓰지 않습니다: 장 시작 전이고, 해외 서버에서는
#     네이버 접속이 불안정하기 때문. 기준가는 모델이 받은 마지막 종가입니다.
# ==============================================================================
from datetime import datetime, timedelta, timezone

from forecast_core import run_forecast
from result_io import save_result, update_history

KST = timezone(timedelta(hours=9))
FORECAST_DAYS = 5


def main():
    now = datetime.now(KST)
    target_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")  # 마지막으로 끝난 거래일 기준
    print(f"[daily] 기준일 {target_date} · {FORECAST_DAYS}영업일 예측 시작")

    result = run_forecast(
        target_date=target_date,
        forecast_days=FORECAST_DAYS,
        use_realtime=False,
        log=print,
    )
    path = save_result(result, extra={
        "source": "auto",
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
    })
    print(f"[daily] 저장 완료: {path}")
    hist = update_history(result)
    print(f"[daily] 예측 기록 {len(hist)}행 (채점 완료 {int(hist['실제값'].notna().sum())}행)")


if __name__ == "__main__":
    main()
