"""포트폴리오 성과 분석 모듈."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import quantstats as qs

from src.constants import DEFAULT_RISK_FREE_RATE


@dataclass(frozen=True)
class PerformanceMetrics:
    """성과 지표를 담는 불변 데이터 클래스."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    best_day: float
    worst_day: float
    var_95: float


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """포트폴리오 일별 수익률을 계산한다.

    Args:
        prices: 종가 DataFrame
        weights: 종목코드 -> 비중 딕셔너리

    Returns:
        포트폴리오 일별 수익률 Series
    """
    daily_returns = prices.pct_change().dropna()
    weight_values = np.array([weights.get(col, 0.0) for col in daily_returns.columns])
    portfolio_returns = (daily_returns.values * weight_values).sum(axis=1)
    return pd.Series(portfolio_returns, index=daily_returns.index, name="Portfolio")


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """누적 수익률을 계산한다."""
    return (1 + returns).cumprod()


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown 시리즈를 계산한다."""
    cumulative = calculate_cumulative_returns(returns)
    running_max = cumulative.expanding().max()
    return (cumulative - running_max) / running_max


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> PerformanceMetrics:
    """종합 성과 지표를 계산한다.

    Raises:
        ValueError: 수익률 데이터가 비어있을 때
    """
    if returns.empty:
        raise ValueError("수익률 데이터가 비어있습니다.")

    return PerformanceMetrics(
        total_return=_safe_stat(qs.stats.comp, returns),
        annualized_return=_safe_stat(qs.stats.cagr, returns),
        volatility=_safe_stat(qs.stats.volatility, returns),
        sharpe_ratio=_safe_stat(qs.stats.sharpe, returns, rf=risk_free_rate),
        sortino_ratio=_safe_stat(qs.stats.sortino, returns, rf=risk_free_rate),
        max_drawdown=_safe_stat(qs.stats.max_drawdown, returns),
        calmar_ratio=_safe_stat(qs.stats.calmar, returns),
        win_rate=_safe_stat(qs.stats.win_rate, returns),
        best_day=float(returns.max()) if len(returns) > 0 else 0.0,
        worst_day=float(returns.min()) if len(returns) > 0 else 0.0,
        var_95=_safe_stat(qs.stats.value_at_risk, returns),
    )


def _safe_stat(func, returns: pd.Series, **kwargs) -> float:
    """QuantStats 함수 호출을 안전하게 수행한다."""
    try:
        result = func(returns, **kwargs)
        if result is None or (isinstance(result, float) and np.isnan(result)):
            return 0.0
        return float(result)
    except Exception:
        return 0.0


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """월별 수익률 테이블을 계산한다.

    Returns:
        DataFrame with index=연도, columns=월(1-12), values=월간수익률
    """
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    return table.pivot(index="Year", columns="Month", values="Return")


def generate_quantstats_report(
    returns: pd.Series,
    title: str = "Portfolio Report",
) -> str | None:
    """QuantStats HTML 리포트를 생성한다.

    Returns:
        HTML 문자열 또는 None (실패 시)
    """
    import os
    import tempfile

    filepath = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            filepath = f.name

        qs.reports.html(returns, output=filepath, title=title)

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    finally:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
