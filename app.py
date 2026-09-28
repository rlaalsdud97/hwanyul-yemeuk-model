# ==============================================================================
# app.py — USD/KRW 환율 예측 대시보드 (Streamlit)
#   실행:  streamlit run app.py
#   구성:  ⚙️ 사이드바 설정 | 🎯 요약카드 | 📈 대화형 그래프 | 📋 예측표 | 📊 성능지표
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from forecast_core import run_forecast

# ------------------------------------------------------------------
# 페이지 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="환율 예측 대시보드",
    page_icon="💱",
    layout="wide",
)

st.title("💱 USD/KRW 환율 예측 대시보드")
st.caption("KOSPI · VIX 결합 앙상블(XGBoost + ARIMA + Prophet) 퀀트 파이프라인")

# ------------------------------------------------------------------
# 사이드바 — 설정
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 예측 설정")
    target_date = st.date_input(
        "기준일 (과거 날짜 = 백테스트)",
        value=pd.to_datetime("2026-09-28"),
    )
    forecast_days = st.slider("예측 기간 (영업일)", min_value=1, max_value=10, value=5)
    use_realtime = st.checkbox("네이버 실시간 환율 보정 사용", value=True)
    st.markdown("---")
    run_btn = st.button("🚀 예측 실행", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        "⚠️ 방향성 정확도가 50% 부근이면 방향 예측은 신뢰하기 어렵습니다. "
        "실거래 판단은 참고용으로만 사용하세요."
    )


# ------------------------------------------------------------------
# 예측 실행 (버튼 클릭 시). 결과는 세션에 캐시.
# ------------------------------------------------------------------
def _do_run():
    progress = st.empty()
    logs = []

    def _log(msg):
        logs.append(str(msg))
        progress.info("🔄 " + str(msg))

    with st.spinner("모델 학습 및 예측 중... (30초~1분 소요)"):
        result = run_forecast(
            target_date=str(target_date),
            forecast_days=int(forecast_days),
            use_realtime=bool(use_realtime),
            log=_log,
        )
    progress.empty()
    st.session_state["result"] = result


if run_btn:
    try:
        _do_run()
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        st.stop()

# 최초 진입 안내
if "result" not in st.session_state:
    st.info("👈 왼쪽 사이드바에서 설정을 확인하고 **[🚀 예측 실행]** 버튼을 눌러주세요.")
    st.stop()

result = st.session_state["result"]

# ------------------------------------------------------------------
# 상단 알림 (크롤링/조정 상태)
# ------------------------------------------------------------------
if result.get("adjusted_note"):
    st.warning("⚠️ " + result["adjusted_note"])
if result.get("crawl_ok"):
    st.success("✅ " + result["crawl_msg"])
else:
    st.warning("⚠️ " + result["crawl_msg"])

# ------------------------------------------------------------------
# 🎯 요약 카드
# ------------------------------------------------------------------
st.subheader("🎯 핵심 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("기준가 (Spot)", f"{result['current_rate']:,.2f} 원")
c2.metric("트레이딩 시그널", result["today_signal"])
c3.metric("밴드 위치 (%B)", f"{result['percent_b']}%",
          help="0%=밴드 하단(저가), 100%=밴드 상단(고가)")
c4.metric("Best 단일모델", f"{result['best_model']}",
          help=f"Score {result['best_score']} (낮을수록 우수)")

# 시그널 해설
sig = result["today_signal"]
if "BUY" in sig:
    st.success("→ 최근 20일 흐름 대비 **저가 구간**입니다. 분할 매수를 고려할 만합니다.")
elif "HOLD" in sig:
    st.info("→ 최근 흐름의 **중립 구간**입니다. 하루 더 추이를 관찰하는 것을 권장합니다.")
else:
    st.warning("→ 최근 흐름 대비 **고가 구간**입니다. 매수를 보류하고 조정을 기다리는 전략을 권장합니다.")

st.markdown("---")

# ------------------------------------------------------------------
# 📈 대화형 그래프 (Plotly)
# ------------------------------------------------------------------
st.subheader(f"📈 {result['forecast_days']}일 예측 차트")

history = result["history"]  # pandas Series (index=날짜, value=종가)
hist_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(history.index)]
hist_values = history.values.tolist()

fdates = result["forecast_dates"]
# 실제 마지막점 ~ 예측 첫점을 잇기 위해 연결
connect_dates = [hist_dates[-1]] + fdates
current_rate = result["current_rate"]

fig = go.Figure()

# 과거 실제
fig.add_trace(go.Scatter(
    x=hist_dates, y=hist_values, mode="lines",
    name="실제 (최근 30일)", line=dict(color="#2c3e50", width=3),
))

# 시그널 밴드 (수평 영역)
fig.add_hrect(y0=result["lower_band"], y1=result["upper_band"],
              fillcolor="#8e44ad", opacity=0.10, line_width=0,
              annotation_text="시그널 밴드 (±1σ)", annotation_position="top left")
fig.add_hline(y=result["band_center"], line=dict(color="#8e44ad", width=1, dash="dot"),
              annotation_text=f"20일 평균 {result['band_center']}", annotation_position="bottom left")

# 각 모델 예측선
fig.add_trace(go.Scatter(
    x=connect_dates, y=[current_rate] + result["xgb_forecast"], mode="lines",
    name=f"XGBoost ({result['weights']['XGBoost']}%)",
    line=dict(color="#27ae60", width=1.5, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=connect_dates, y=[current_rate] + result["arima_forecast"], mode="lines",
    name=f"ARIMA ({result['weights']['ARIMA']}%)",
    line=dict(color="#2980b9", width=1.5, dash="dashdot"),
))
fig.add_trace(go.Scatter(
    x=connect_dates, y=[current_rate] + result["prophet_forecast"], mode="lines",
    name=f"Prophet ({result['weights']['Prophet']}%)",
    line=dict(color="#f39c12", width=1.5, dash="dot"),
))
# 최종 앙상블 (강조)
fig.add_trace(go.Scatter(
    x=connect_dates, y=[current_rate] + result["ensemble_forecast"], mode="lines+markers",
    name="🔥 최종 앙상블", line=dict(color="#c0392b", width=4),
    marker=dict(size=8, symbol="square"),
))

fig.update_layout(
    height=520,
    xaxis_title="날짜", yaxis_title="환율 (KRW/USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------
# 📋 예측표 + 📊 성능지표 (2열)
# ------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("📋 일자별 예측")
    table = result["forecast_table"].copy()
    table = table.rename(columns={
        "Date": "날짜", "Final": "🔥 최종", "Buy_Band": "👇 매수밴드",
        "Sell_Band": "☝️ 매도밴드", "Signal": "🎯 시그널",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    # CSV 다운로드
    csv = result["forecast_table"].to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 예측 결과 CSV 다운로드", data=csv,
                       file_name=f"forecast_{result['target_date']}.csv",
                       mime="text/csv")

with right:
    st.subheader("📊 모델 성능 (OOT 검증)")
    m = result.get("metrics", {})
    if m:
        st.metric("검증 표본 수", f"{m['sample_days']}일")
        mc1, mc2 = st.columns(2)
        mc1.metric("MAE (평균오차)", f"{m['mae']} 원")
        mc2.metric("RMSE", f"{m['rmse']} 원")
        st.metric("✅ 예측 정확도 (100-MAPE)", f"{m['accuracy']} %")

        da = m["directional_accuracy"]
        st.metric("🎯 방향성 정확도", f"{da} %",
                  help="오를지/내릴지 방향 적중률. 50%=동전던지기")
        # 방향성 해석
        if da < 50:
            st.error(f"방향성 {da}% — 동전던지기(50%)보다 낮습니다. "
                     "방향 예측은 신뢰하지 마세요.")
        elif da < 55:
            st.warning(f"방향성 {da}% — 동전던지기 수준입니다. 참고용으로만.")
        else:
            st.success(f"방향성 {da}% — 유의미한 수준입니다.")
    else:
        st.info("성능 지표를 계산할 수 없습니다 (검증 표본 부족).")

st.markdown("---")
st.caption(
    "본 대시보드는 학습/참고 목적의 예측 도구입니다. 실제 외환 거래의 투자 판단 "
    "및 그 결과에 대한 책임은 이용자 본인에게 있습니다."
)
