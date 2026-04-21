"""포트폴리오 최적화 모듈."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import (
    BlackLittermanModel,
    EfficientFrontier,
    HRPOpt,
    expected_returns,
    risk_models,
)
from scipy.optimize import minimize

from src.constants import (
    DEFAULT_RISK_FREE_RATE,
    MIN_WEIGHT_PER_ASSET,
    STRATEGY_BLACK_LITTERMAN,
    STRATEGY_EQUAL_WEIGHT,
    STRATEGY_HRP,
    STRATEGY_MAX_SHARPE,
    STRATEGY_MIN_VOLATILITY,
    STRATEGY_RISK_PARITY,
)


@dataclass(frozen=True)
class OptimizationResult:
    """최적화 결과를 담는 불변 데이터 클래스."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    strategy: str


def optimize_max_sharpe(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """최대 Sharpe Ratio 포트폴리오를 구한다.

    Raises:
        ValueError: 모든 종목의 기대수익률이 무위험 이자율보다 낮을 때
    """
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    if mu.max() <= risk_free_rate:
        best_ticker = mu.idxmax()
        best_return = mu.max()
        raise ValueError(
            f"추천 불가: 선택한 종목들의 과거 수익률({best_return:.1%})이 "
            f"무위험 이자율({risk_free_rate:.1%})보다 낮습니다. "
            f"국채에 투자하는 것이 더 유리합니다. "
            f"다른 종목을 선택하거나 분석 기간을 변경해주세요."
        )

    ef = EfficientFrontier(mu, S, weight_bounds=(MIN_WEIGHT_PER_ASSET, 1.0))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    return OptimizationResult(
        weights=dict(weights),
        expected_return=perf[0],
        volatility=perf[1],
        sharpe_ratio=perf[2],
        strategy=STRATEGY_MAX_SHARPE,
    )


def optimize_min_volatility(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """최소 변동성 포트폴리오를 구한다."""
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(MIN_WEIGHT_PER_ASSET, 1.0))
    ef.min_volatility()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    return OptimizationResult(
        weights=dict(weights),
        expected_return=perf[0],
        volatility=perf[1],
        sharpe_ratio=perf[2],
        strategy=STRATEGY_MIN_VOLATILITY,
    )


def optimize_risk_parity(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """Risk Parity 포트폴리오를 구한다."""
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov().values
    n_assets = len(prices.columns)

    def risk_parity_objective(weights):
        portfolio_var = weights @ cov_matrix @ weights
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib
        target_contrib = portfolio_var / n_assets
        return np.sum((risk_contrib - target_contrib) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0.01, 1.0) for _ in range(n_assets))
    initial = np.array([1.0 / n_assets] * n_assets)

    result = minimize(
        risk_parity_objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights_arr = result.x / result.x.sum()
    weights_dict = {
        col: round(float(w), 4) for col, w in zip(prices.columns, weights_arr)
    }

    mu_annual = returns.mean() * 252
    port_return = float(np.dot(weights_arr, mu_annual.values))
    port_vol = float(np.sqrt(weights_arr @ (cov_matrix * 252) @ weights_arr))
    port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=port_sharpe,
        strategy=STRATEGY_RISK_PARITY,
    )


def optimize_equal_weight(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """동일 비중 포트폴리오를 구한다."""
    n_assets = len(prices.columns)
    w = 1.0 / n_assets
    weights_dict = {col: round(w, 4) for col in prices.columns}

    returns = prices.pct_change().dropna()
    weights_arr = np.array([w] * n_assets)
    cov_matrix = returns.cov().values * 252
    mu_annual = returns.mean().values * 252

    port_return = float(np.dot(weights_arr, mu_annual))
    port_vol = float(np.sqrt(weights_arr @ cov_matrix @ weights_arr))
    port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return OptimizationResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=port_sharpe,
        strategy=STRATEGY_EQUAL_WEIGHT,
    )


def optimize_hrp(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """HRP (Hierarchical Risk Parity) 포트폴리오를 구한다.

    비슷한 종목끼리 클러스터링한 뒤, 그룹별로 위험을 동등 배분한다.
    """
    returns = prices.pct_change().dropna()

    hrp = HRPOpt(returns)
    weights = hrp.optimize()
    weights_dict = {k: round(float(v), 4) for k, v in weights.items()}

    perf = hrp.portfolio_performance()

    return OptimizationResult(
        weights=weights_dict,
        expected_return=perf[0],
        volatility=perf[1],
        sharpe_ratio=(perf[0] - risk_free_rate) / perf[1] if perf[1] > 0 else 0.0,
        strategy=STRATEGY_HRP,
    )


def optimize_black_litterman(
    prices: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """Black-Litterman 포트폴리오를 구한다.

    시가총액 기반 시장 균형 수익률에서 출발하여 최적 비중을 계산한다.
    (사용자 전망 없이 시장 균형만 사용하는 기본 버전)
    """
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # 시가총액 대용: 각 종목의 최근 가격을 시가총액 비중으로 사용
    market_caps = prices.iloc[-1]
    mcaps = {col: float(market_caps[col]) for col in prices.columns}

    bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_free_rate=risk_free_rate)
    bl_returns = bl.bl_returns()

    ef = EfficientFrontier(bl_returns, S, weight_bounds=(MIN_WEIGHT_PER_ASSET, 1.0))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    return OptimizationResult(
        weights=dict(weights),
        expected_return=perf[0],
        volatility=perf[1],
        sharpe_ratio=perf[2],
        strategy=STRATEGY_BLACK_LITTERMAN,
    )


_STRATEGY_MAP = {
    STRATEGY_MAX_SHARPE: optimize_max_sharpe,
    STRATEGY_MIN_VOLATILITY: optimize_min_volatility,
    STRATEGY_RISK_PARITY: optimize_risk_parity,
    STRATEGY_EQUAL_WEIGHT: optimize_equal_weight,
    STRATEGY_HRP: optimize_hrp,
    STRATEGY_BLACK_LITTERMAN: optimize_black_litterman,
}


def run_optimization(
    prices: pd.DataFrame,
    strategy: str,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> OptimizationResult:
    """전략 이름에 따라 적절한 최적화 함수를 호출한다.

    Raises:
        ValueError: 알 수 없는 전략명
    """
    func = _STRATEGY_MAP.get(strategy)
    if func is None:
        raise ValueError(f"알 수 없는 전략: {strategy}")
    return func(prices, risk_free_rate)
