# ==============================================================================
# app.py — USD/KRW 환율 예측 대시보드 (Streamlit)
#   실행:  streamlit run app.py
#   구성:  ⚙️ 사이드바 설정 | 🎯 요약카드 | 📈 대화형 그래프 | 📋 예측표 | 📊 성능지표
# ==============================================================================
from datetime import date

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
st.caption("KOSPI · VIX 결합 앙상블(XGBoost + ARIMA, 선택: Prophet) 퀀트 파이프라인")

# ------------------------------------------------------------------
# 사이드바 — 설정
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 예측 설정")
    target_date = st.date_input(
        "기준일 (과거 날짜 = 백테스트)",
        value=date.today(),
    )
    forecast_days = st.slider("예측 기간 (영업일)", min_value=1, max_value=10, value=5)
    use_realtime = st.checkbox("네이버 실시간 환율 보정 사용", value=True)
    with st.expander("🔧 고급 설정 (켜면 느려짐)"):
        use_prophet = st.checkbox(
            "Prophet 모델 포함", value=False,
            help="앙상블 가중치 상한이 5%라 결과 영향은 작고, 학습 시간은 가장 깁니다.",
        )
        compare_regression = st.checkbox(
            "백테스트에서 회귀 모델 방향성도 비교", value=False,
            help="켜면 백테스트 학습 횟수가 2배가 됩니다.",
        )
    st.markdown("---")
    run_btn = st.button("🚀 예측 실행", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        "⚠️ 방향성 정확도가 50% 부근이면 방향 예측은 신뢰하기 어렵습니다. "
        "실거래 판단은 참고용으로만 사용하세요."
    )


# ------------------------------------------------------------------
# 예측 실행 (버튼 클릭 시).
#  - 같은 설정이면 30분 동안 캐시된 결과를 바로 반환 (재학습 없음)
#  - 캐시 함수 안에서는 화면 요소를 그릴 수 없어서 진행 로그 대신 스피너만 사용
# ------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecast(target_date, forecast_days, use_realtime, use_prophet,
                    compare_regression):
    return run_forecast(
        target_date=target_date,
        forecast_days=forecast_days,
        use_realtime=use_realtime,
        use_prophet=use_prophet,
        compare_regression=compare_regression,
    )


def _do_run():
    with st.spinner("모델 학습 및 예측 중..."):
        result = cached_forecast(
            str(target_date), int(forecast_days), bool(use_realtime),
            bool(use_prophet), bool(compare_regression),
        )
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
# 🎯 내일 방향 예측 (분류 모델)
# ------------------------------------------------------------------
direction = result.get("direction", {})
if direction:
    st.subheader(f"🎯 내일({direction.get('next_date','')}) 방향 예측")
    st.caption("방향 전용 분류 모델(XGBClassifier). 값이 아니라 '오를지/내릴지'를 직접 학습합니다.")

    d1, d2, d3 = st.columns(3)
    label = direction["label"]
    if direction["up"]:
        d1.metric("예측 방향", "▲ 상승", delta="UP", delta_color="normal")
    else:
        d1.metric("예측 방향", "▼ 하락", delta="DOWN", delta_color="inverse")
    d2.metric("상승 확률", f"{direction['up_proba']}%",
              help="모델이 계산한 내일 상승 확률. 50%보다 높으면 상승 예측.")
    d3.metric("예측 확신도", f"{direction['confidence']}%",
              help="예측 방향에 대한 확신 정도. 50%에 가까울수록 불확실.")

    # 방향성 정확도 비교 (회귀 vs 분류)
    dm = direction.get("metrics", {})
    if dm:
        clf_da = dm.get("classifier_da")
        reg_da = dm.get("regression_da")
        base_da = dm.get("baseline_da")
        reg_txt = f"  vs  🟢 회귀(기존) {reg_da}%" if reg_da is not None else ""
        st.markdown(
            f"**방향성 정확도** (최근 {dm.get('test_days','?')}일 walk-forward 검증): "
            f"🔴 분류 **{clf_da}%**{reg_txt}  vs  ⚪ 기준선(항상상승) {base_da}%"
        )
        if clf_da is not None:
            if clf_da >= 55:
                st.success(f"분류 모델 방향성 {clf_da}% — 유의미한 수준입니다. (동전던지기 50%)")
            elif clf_da >= 50:
                st.info(f"분류 모델 방향성 {clf_da}% — 동전던지기 수준입니다. 참고용으로만.")
            else:
                st.warning(f"분류 모델 방향성 {clf_da}% — 신뢰하기 어렵습니다.")

    # 확신도 경고
    if direction["confidence"] < 55:
        st.caption("⚠️ 확신도가 낮습니다(50%대). 방향이 불확실하니 참고용으로만 보세요.")
    st.caption("⚠️ 추세 전환점(급등·급락 시작)에서는 방향 예측이 특히 약해질 수 있습니다.")

    st.markdown("---")

# ------------------------------------------------------------------
# 🔍 백테스트 검증: 과거에 방향을 맞췄는지 눈으로 확인 (월별 조회)
# ------------------------------------------------------------------
bt = direction.get("backtest_table") if direction else None

if bt is not None and len(bt) > 0:
    with st.expander(
        "🔍 백테스트 검증 — 과거에 방향을 맞췄나요?",
        expanded=True
    ):
        st.caption(
            "과거 각 날짜에 다음 날의 환율 방향을 예측했다고 가정하고 "
            "실제 결과와 비교합니다. 초록색은 적중, 빨간색은 오답입니다."
        )

        # 날짜 형식 변환
        bt = bt.copy()
        bt["날짜"] = pd.to_datetime(bt["날짜"])

        # 전체 기간 적중률
        overall_hit_rate = bt["_hit"].astype(bool).mean() * 100

        st.metric(
            "전체 기간 방향 적중률",
            f"{overall_hit_rate:.1f}%",
            help=f"전체 {len(bt)}개 검증 날짜 기준"
        )

        # 월별 선택 메뉴
        available_months = sorted(
            bt["날짜"].dt.to_period("M").astype(str).unique(),
            reverse=True
        )

        month_options = ["전체 기간"] + available_months

        selected_month = st.selectbox(
            "조회할 기간 선택",
            month_options,
            index=0,
            key="backtest_month"
        )

        # 선택한 기간만 표시
        if selected_month == "전체 기간":
            bt_view = bt.copy()
        else:
            bt_view = bt[
                bt["날짜"].dt.to_period("M").astype(str)
                == selected_month
            ].copy()

        hits = bt_view["_hit"].astype(bool).tolist()
        closes = bt_view["_close"].tolist()
        bt_dates = bt_view["날짜"]

        hit_rate = (
            sum(hits) / len(hits) * 100
            if hits else 0
        )

        st.subheader(
            f"{selected_month} · 방향 적중률 {hit_rate:.1f}%"
            if selected_month != "전체 기간"
            else f"전체 기간 · 방향 적중률 {hit_rate:.1f}%"
        )
        st.caption(
            f"검증 날짜 {len(bt_view)}일 · "
            f"적중 {sum(hits)}일 · 오답 {len(hits) - sum(hits)}일"
        )

        # 그래프 데이터 분리
        hit_x = [d for d, h in zip(bt_dates, hits) if h]
        hit_y = [c for c, h in zip(closes, hits) if h]
        miss_x = [d for d, h in zip(bt_dates, hits) if not h]
        miss_y = [c for c, h in zip(closes, hits) if not h]

        bt_fig = go.Figure()

        bt_fig.add_trace(go.Scatter(
            x=bt_dates,
            y=closes,
            mode="lines",
            name="실제 환율",
            line=dict(color="#2c3e50", width=2)
        ))

        bt_fig.add_trace(go.Scatter(
            x=hit_x,
            y=hit_y,
            mode="markers",
            name="방향 적중",
            marker=dict(color="#27ae60", size=9)
        ))

        bt_fig.add_trace(go.Scatter(
            x=miss_x,
            y=miss_y,
            mode="markers",
            name="방향 오답",
            marker=dict(color="#e74c3c", size=9)
        ))

        bt_fig.update_layout(
            height=380,
            title=f"방향성 백테스트 · {hit_rate:.1f}%",
            xaxis_title="날짜",
            yaxis_title="USD/KRW",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(bt_fig, use_container_width=True)

        # 날짜별 상세 표
        st.markdown("**날짜별 적중 상세**")

        show_cols = [
            "날짜", "현재가", "실제_다음날",
            "실제_방향", "예측_방향", "상승확률", "적중"
        ]

        st.dataframe(
            bt_view[show_cols].assign(날짜=bt_view["날짜"].dt.strftime("%Y-%m-%d")),
            use_container_width=True,
            hide_index=True
        )
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
if result.get("prophet_forecast"):
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
