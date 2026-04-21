"""한국 주식 포트폴리오 매니저 - Streamlit 메인 앱."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.constants import (
    APP_TITLE,
    DEFAULT_REBALANCE_BAND,
    DEFAULT_RISK_FREE_RATE,
    MARKET_OPTIONS,
    OPTIMIZATION_STRATEGIES,
    SIDEBAR_TITLE,
)
from src.data import fetch_dividend_data, fetch_price_data, fetch_stock_list, validate_tickers
from src.optimizer import run_optimization
from src.analyzer import (
    calculate_cumulative_returns,
    calculate_monthly_returns,
    calculate_performance_metrics,
    calculate_portfolio_returns,
    generate_quantstats_report,
)
from src.rebalancer import calculate_current_weights, check_rebalance_needed
from src.charts import (
    create_allocation_pie_chart,
    create_cumulative_returns_chart,
    create_drawdown_chart,
    create_monthly_returns_heatmap,
    create_weight_comparison_chart,
)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(f"📊 {APP_TITLE}")

    # Select all 옵션 숨기기
    st.markdown("""
        <script>
        const observer = new MutationObserver(() => {
            document.querySelectorAll('li[role="option"]').forEach(el => {
                if (el.textContent.trim() === 'Select all' ||
                    el.textContent.trim().match(/^Select \\d+ matches$/)) {
                    el.style.display = 'none';
                }
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
    """, unsafe_allow_html=True)

    render_sidebar()

    if "optimization_result" in st.session_state:
        render_dashboard()
    else:
        st.info("왼쪽 사이드바에서 종목을 선택하고 '분석 시작' 버튼을 눌러주세요.")


def render_sidebar():
    st.sidebar.header(SIDEBAR_TITLE)

    # 시장 선택
    market = st.sidebar.selectbox("시장", MARKET_OPTIONS)

    # 종목 리스트 로드
    with st.spinner("종목 리스트 로딩 중..."):
        try:
            stock_list = fetch_stock_list(market)
        except Exception as e:
            import traceback
            st.sidebar.error(f"종목 리스트를 가져올 수 없습니다: {e}")
            st.sidebar.code(traceback.format_exc())
            return

    # 종목 선택 (코드 - 이름 형식)
    stock_options = {
        f"{row['Name']} ({row['Code']})": row["Code"]
        for _, row in stock_list.iterrows()
    }
    selected_labels = st.sidebar.multiselect(
        "종목 선택 (2~20개)",
        options=list(stock_options.keys()),
        default=[],
        max_selections=20,
    )
    selected_tickers = tuple(stock_options[label] for label in selected_labels)

    # 종목명 매핑 저장
    ticker_name_map = {
        row["Code"]: row["Name"]
        for _, row in stock_list[stock_list["Code"].isin(selected_tickers)].iterrows()
    }

    # 기간 설정 (년/월 선택)
    today = datetime.now()
    # 당월 제외: 지난달 마지막 날까지만 선택 가능
    last_selectable = today.replace(day=1) - timedelta(days=1)

    # 시작: 최대 10년 전부터
    min_start_year = last_selectable.year - 10
    start_months = _generate_month_options(min_start_year, 1, last_selectable.year, last_selectable.month)
    default_start_idx = _find_default_index(start_months, last_selectable.year - 3, last_selectable.month)
    start_label = st.sidebar.selectbox("시작 (년/월)", start_months, index=default_start_idx)
    start_year, start_month = _parse_month_label(start_label)
    start_date = datetime(start_year, start_month, 1)

    # 종료: 시작월 이후 ~ 지난달까지
    end_months = _generate_month_options(start_year, start_month + 1, last_selectable.year, last_selectable.month)
    if not end_months:
        st.sidebar.error("시작월 이후에 선택 가능한 종료월이 없습니다.")
        return
    end_label = st.sidebar.selectbox("종료 (년/월)", end_months, index=len(end_months) - 1)
    end_year, end_month = _parse_month_label(end_label)
    # 종료월의 마지막 날
    if end_month == 12:
        end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)

    # 전략 선택
    strategy = st.sidebar.selectbox("최적화 전략", OPTIMIZATION_STRATEGIES)

    # 상세 설정
    with st.sidebar.expander("상세 설정"):
        risk_free_rate = st.number_input(
            "무위험 이자율 (%)",
            value=DEFAULT_RISK_FREE_RATE * 100,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        ) / 100

        rebalance_band = st.slider(
            "리밸런싱 밴드 (%)",
            min_value=1,
            max_value=20,
            value=int(DEFAULT_REBALANCE_BAND * 100),
        ) / 100

    # 분석 시작 버튼
    if st.sidebar.button("🚀 분석 시작", use_container_width=True):
        is_valid, error_msg = validate_tickers(selected_tickers)
        if not is_valid:
            st.sidebar.error(error_msg)
            return

        with st.spinner("데이터 수집 및 분석 중..."):
            try:
                prices = fetch_price_data(
                    selected_tickers,
                    str(start_date),
                    str(end_date),
                )
                result = run_optimization(prices, strategy, risk_free_rate)
                portfolio_returns = calculate_portfolio_returns(prices, result.weights)
                metrics = calculate_performance_metrics(portfolio_returns, risk_free_rate)
                current_weights = calculate_current_weights(prices, result.weights)

                # 배당 데이터 수집
                dividend_data = fetch_dividend_data(
                    selected_tickers,
                    str(start_date),
                    str(end_date),
                )

                st.session_state.optimization_result = result
                st.session_state.prices = prices
                st.session_state.portfolio_returns = portfolio_returns
                st.session_state.metrics = metrics
                st.session_state.current_weights = current_weights
                st.session_state.ticker_name_map = ticker_name_map
                st.session_state.rebalance_band = rebalance_band
                st.session_state.risk_free_rate = risk_free_rate
                st.session_state.dividend_data = dividend_data

            except Exception as e:
                st.sidebar.error(f"분석 실패: {e}")


def render_dashboard():
    result = st.session_state.optimization_result
    metrics = st.session_state.metrics
    portfolio_returns = st.session_state.portfolio_returns
    prices = st.session_state.prices
    current_weights = st.session_state.current_weights
    ticker_name_map = st.session_state.ticker_name_map
    rebalance_band = st.session_state.rebalance_band

    # 상단 KPI 카드
    st.subheader(f"전략: {result.strategy}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("기대 수익률", f"{result.expected_return:.2%}")
    col2.metric("변동성", f"{result.volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    col4.metric("MDD", f"{metrics.max_drawdown:.2%}")

    dividend_data = st.session_state.dividend_data

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 포트폴리오 비중",
        "📈 성과 분석",
        "📅 월별 수익률",
        "💰 배당금",
        "⚖️ 리밸런싱",
        "📥 리포트",
    ])

    with tab1:
        render_allocation_tab(result, ticker_name_map)

    with tab2:
        render_performance_tab(portfolio_returns, metrics)

    with tab3:
        render_monthly_tab(portfolio_returns)

    with tab4:
        render_dividend_tab(result, dividend_data, ticker_name_map)

    with tab5:
        render_rebalance_tab(result, current_weights, ticker_name_map, rebalance_band)

    with tab6:
        render_report_tab(portfolio_returns)


def render_allocation_tab(result, ticker_name_map):
    col1, col2 = st.columns(2)

    with col1:
        weights_data = [
            {
                "종목": ticker_name_map.get(ticker, ticker),
                "코드": ticker,
                "비중": f"{weight:.1%}",
            }
            for ticker, weight in sorted(
                result.weights.items(), key=lambda x: x[1], reverse=True
            )
            if weight > 0.001
        ]
        st.dataframe(pd.DataFrame(weights_data), use_container_width=True, hide_index=True)

    with col2:
        fig = create_allocation_pie_chart(result.weights, ticker_name_map)
        st.plotly_chart(fig, use_container_width=True)


def render_performance_tab(portfolio_returns, metrics):
    # 누적 수익률 차트
    fig_cum = create_cumulative_returns_chart(portfolio_returns)
    st.plotly_chart(fig_cum, use_container_width=True)

    # 성과 지표 테이블
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**수익 지표**")
        st.markdown(f"- 총 수익률: **{metrics.total_return:.2%}**")
        st.markdown(f"- 연 수익률 (CAGR): **{metrics.annualized_return:.2%}**")
        st.markdown(f"- 최고 일간 수익: **{metrics.best_day:.2%}**")
        st.markdown(f"- 최저 일간 수익: **{metrics.worst_day:.2%}**")
        st.markdown(f"- 승률: **{metrics.win_rate:.2%}**")

    with col2:
        st.markdown("**위험 지표**")
        st.markdown(f"- 변동성: **{metrics.volatility:.2%}**")
        st.markdown(f"- Sharpe Ratio: **{metrics.sharpe_ratio:.2f}**")
        st.markdown(f"- Sortino Ratio: **{metrics.sortino_ratio:.2f}**")
        st.markdown(f"- Calmar Ratio: **{metrics.calmar_ratio:.2f}**")
        st.markdown(f"- VaR (95%): **{metrics.var_95:.2%}**")

    # Drawdown 차트
    fig_dd = create_drawdown_chart(portfolio_returns)
    st.plotly_chart(fig_dd, use_container_width=True)


def render_monthly_tab(portfolio_returns):
    monthly = calculate_monthly_returns(portfolio_returns)
    if monthly.empty:
        st.warning("월별 수익률 데이터가 충분하지 않습니다.")
        return

    fig = create_monthly_returns_heatmap(monthly)
    st.plotly_chart(fig, use_container_width=True)

    # 연도별 합계
    yearly = monthly.sum(axis=1)
    st.markdown("**연도별 수익률**")
    for year, ret in yearly.items():
        st.markdown(f"- {year}년: **{ret:.2%}**")


def render_rebalance_tab(result, current_weights, ticker_name_map, band):
    # 비중 비교 차트
    fig = create_weight_comparison_chart(
        result.weights, current_weights, ticker_name_map
    )
    st.plotly_chart(fig, use_container_width=True)

    # 리밸런싱 알림
    needs_rebalance, alerts = check_rebalance_needed(
        result.weights, current_weights, band, ticker_name_map
    )

    if needs_rebalance:
        st.warning(f"⚠️ {len(alerts)}개 종목이 밴드(±{band:.0%})를 초과했습니다.")
        for alert in alerts:
            st.markdown(
                f"- **{alert.name}** ({alert.ticker}): "
                f"목표 {alert.target_weight:.1%} → 현재 {alert.current_weight:.1%} "
                f"(편차 {alert.deviation:.1%}) → **{alert.action}** 필요"
            )
    else:
        st.success(f"✅ 모든 종목이 밴드(±{band:.0%}) 내에 있습니다. 리밸런싱이 필요하지 않습니다.")


def render_dividend_tab(result, dividend_data, ticker_name_map):
    st.markdown("선택한 종목들의 배당 정보입니다. (최근 배당 기준)")

    if dividend_data.empty:
        st.warning("배당 데이터를 가져올 수 없습니다.")
        return

    # 비중과 합쳐서 테이블 구성
    rows = []
    total_weighted_yield = 0.0

    for _, row in dividend_data.iterrows():
        code = row["Code"]
        weight = result.weights.get(code, 0.0)
        name = ticker_name_map.get(code, code)
        div_yield = row["DividendYield"]
        dividend = row["Dividend"]
        weighted_yield = div_yield * weight

        total_weighted_yield += weighted_yield

        rows.append({
            "종목": name,
            "코드": code,
            "비중": f"{weight:.1%}",
            "배당 수익률": f"{div_yield:.2f}%" if div_yield > 0 else "-",
            "주당 배당금": f"{dividend:,.0f}원" if dividend > 0 else "-",
            "기여도": f"{weighted_yield:.2f}%" if weighted_yield > 0 else "-",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 포트폴리오 전체 배당 수익률
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("포트폴리오 배당 수익률", f"{total_weighted_yield:.2f}%")

    has_dividend = dividend_data[dividend_data["DividendYield"] > 0]
    col2.metric("배당 지급 종목", f"{len(has_dividend)} / {len(dividend_data)}개")

    if total_weighted_yield > 0:
        st.info(
            f"💡 이 포트폴리오의 예상 연간 배당 수익률은 **{total_weighted_yield:.2f}%** 입니다. "
            f"주가 수익률과 별도로 추가 수익이 발생합니다."
        )
    else:
        st.info("선택한 종목 중 배당을 지급하는 종목이 없습니다.")


def render_report_tab(portfolio_returns):
    st.markdown("QuantStats HTML 리포트를 생성하여 다운로드할 수 있습니다.")

    if st.button("📄 리포트 생성"):
        with st.spinner("리포트 생성 중... (30초~1분 소요)"):
            html = generate_quantstats_report(portfolio_returns)
            if html:
                st.download_button(
                    label="📥 리포트 다운로드 (HTML)",
                    data=html,
                    file_name="portfolio_report.html",
                    mime="text/html",
                )
                st.success("리포트가 생성되었습니다. 위 버튼을 눌러 다운로드하세요.")
            else:
                st.error("리포트 생성에 실패했습니다.")


def _generate_month_options(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> list[str]:
    """년/월 선택 옵션 리스트를 생성한다."""
    options = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        options.append(f"{year}년 {month:02d}월")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return options


def _parse_month_label(label: str) -> tuple[int, int]:
    """'2023년 04월' 형식에서 (year, month)를 추출한다."""
    parts = label.replace("년", "").replace("월", "").split()
    return int(parts[0]), int(parts[1])


def _find_default_index(options: list[str], target_year: int, target_month: int) -> int:
    """기본 선택 인덱스를 찾는다 (3년 전 기본)."""
    target = f"{target_year}년 {target_month:02d}월"
    try:
        return options.index(target)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
