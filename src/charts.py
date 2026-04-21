"""Plotly 차트 생성 모듈."""

import pandas as pd
import plotly.graph_objects as go

from src.analyzer import calculate_cumulative_returns, calculate_drawdown


def create_cumulative_returns_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "누적 수익률",
) -> go.Figure:
    """누적 수익률 라인 차트를 생성한다."""
    cumulative = calculate_cumulative_returns(portfolio_returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=(cumulative - 1) * 100,
        mode="lines",
        name="포트폴리오",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>수익률: %{y:.2f}%<extra></extra>",
    ))

    if benchmark_returns is not None:
        bench_cum = calculate_cumulative_returns(benchmark_returns)
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=(bench_cum - 1) * 100,
            mode="lines",
            name="벤치마크",
            line=dict(color="#aaaaaa", width=1, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="수익률 (%)",
        template="plotly_white",
        hovermode="x unified",
        height=400,
    )
    return fig


def create_allocation_pie_chart(
    weights: dict[str, float],
    ticker_name_map: dict[str, str] | None = None,
    title: str = "자산 배분",
) -> go.Figure:
    """자산 배분 도넛 차트를 생성한다."""
    name_map = ticker_name_map or {}
    filtered = {k: v for k, v in weights.items() if v > 0.001}

    labels = [name_map.get(k, k) for k in filtered]
    values = list(filtered.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
        hovertemplate="%{label}<br>비중: %{percent}<extra></extra>",
    )])
    fig.update_layout(title=title, height=400, template="plotly_white")
    return fig


def create_monthly_returns_heatmap(
    monthly_returns: pd.DataFrame,
    title: str = "월별 수익률 (%)",
) -> go.Figure:
    """월별 수익률 히트맵을 생성한다."""
    values_pct = monthly_returns.values * 100
    text = [[f"{v:.1f}" if not pd.isna(v) else "" for v in row] for row in values_pct]

    month_labels = [f"{m}월" for m in monthly_returns.columns]

    fig = go.Figure(data=go.Heatmap(
        z=values_pct,
        x=month_labels,
        y=[str(y) for y in monthly_returns.index],
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        hovertemplate="%{y}년 %{x}<br>수익률: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(title=title, height=300, template="plotly_white")
    return fig


def create_drawdown_chart(
    portfolio_returns: pd.Series,
    title: str = "Drawdown",
) -> go.Figure:
    """Drawdown 영역 차트를 생성한다."""
    dd = calculate_drawdown(portfolio_returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values * 100,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color="#d62728", width=1),
        fillcolor="rgba(214,39,40,0.3)",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=300,
    )
    return fig


def create_weight_comparison_chart(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    ticker_name_map: dict[str, str] | None = None,
    title: str = "목표 vs 현재 비중",
) -> go.Figure:
    """목표 비중과 현재 비중을 비교하는 바 차트를 생성한다."""
    name_map = ticker_name_map or {}
    tickers = list(target_weights.keys())
    labels = [name_map.get(t, t) for t in tickers]

    target_vals = [target_weights.get(t, 0) * 100 for t in tickers]
    current_vals = [current_weights.get(t, 0) * 100 for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="목표", x=labels, y=target_vals, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="현재", x=labels, y=current_vals, marker_color="#ff7f0e"))

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="비중 (%)",
        template="plotly_white",
        height=400,
    )
    return fig
