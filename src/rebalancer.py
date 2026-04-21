"""포트폴리오 리밸런싱 모듈."""

from dataclasses import dataclass

import pandas as pd

from src.constants import DEFAULT_REBALANCE_BAND


@dataclass(frozen=True)
class RebalanceAlert:
    """리밸런싱 알림을 담는 불변 데이터 클래스."""

    ticker: str
    name: str
    target_weight: float
    current_weight: float
    deviation: float
    action: str  # "매수" 또는 "매도"


def calculate_current_weights(
    prices: pd.DataFrame,
    initial_weights: dict[str, float],
) -> dict[str, float]:
    """초기 비중과 가격 변동을 반영한 현재 비중을 계산한다."""
    if len(prices) < 2:
        return dict(initial_weights)

    first_prices = prices.iloc[0]
    last_prices = prices.iloc[-1]
    price_changes = last_prices / first_prices

    current_values = {
        ticker: initial_weights.get(ticker, 0.0) * float(price_changes.get(ticker, 1.0))
        for ticker in prices.columns
    }

    total = sum(current_values.values())
    if total == 0:
        return dict(initial_weights)

    return {ticker: value / total for ticker, value in current_values.items()}


def check_rebalance_needed(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    band: float = DEFAULT_REBALANCE_BAND,
    ticker_name_map: dict[str, str] | None = None,
) -> tuple[bool, list[RebalanceAlert]]:
    """리밸런싱이 필요한지 확인하고 알림 목록을 생성한다."""
    name_map = ticker_name_map or {}
    alerts = []

    for ticker, target in target_weights.items():
        current = current_weights.get(ticker, 0.0)
        deviation = abs(current - target)

        if deviation > band:
            action = "매도" if current > target else "매수"
            alerts.append(RebalanceAlert(
                ticker=ticker,
                name=name_map.get(ticker, ticker),
                target_weight=target,
                current_weight=current,
                deviation=deviation,
                action=action,
            ))

    alerts_sorted = sorted(alerts, key=lambda a: a.deviation, reverse=True)
    return len(alerts_sorted) > 0, alerts_sorted
