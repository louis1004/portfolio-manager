"""한국 주식 데이터 수집 모듈."""

import pandas as pd
import streamlit as st

from src.constants import MARKET_ALL, MARKET_ETF, MARKET_KOSDAQ, MARKET_KOSPI, MIN_DATA_DAYS


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_list(market: str = MARKET_ALL) -> pd.DataFrame:
    """한국 주식 종목 리스트를 가져온다.

    Args:
        market: "KOSPI", "KOSDAQ", 또는 "전체"

    Returns:
        DataFrame with columns: [Code, Name, Market]
    """
    try:
        import FinanceDataReader as fdr

        # 주식 종목
        df = fdr.StockListing("KRX")
        stocks = pd.DataFrame({
            "Code": df["Code"].astype(str).str.zfill(6),
            "Name": df["Name"],
            "Market": df["Market"],
        })

        # ETF 종목
        etf_df = fdr.StockListing("ETF/KR")
        etfs = pd.DataFrame({
            "Code": etf_df["Code"].astype(str).str.zfill(6),
            "Name": etf_df["Name"],
            "Market": MARKET_ETF,
        })

        result = pd.concat([stocks, etfs], ignore_index=True)
    except Exception:
        result = _fetch_stock_list_fallback()

    if market == MARKET_KOSPI:
        result = result[result["Market"] == MARKET_KOSPI]
    elif market == MARKET_KOSDAQ:
        result = result[result["Market"] == MARKET_KOSDAQ]
    elif market == MARKET_ETF:
        result = result[result["Market"] == MARKET_ETF]

    return result.sort_values("Name").reset_index(drop=True)


def _fetch_stock_list_fallback() -> pd.DataFrame:
    """pykrx를 사용한 종목 리스트 폴백."""
    from pykrx import stock
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    kospi_tickers = stock.get_market_ticker_list(today, market="KOSPI")
    kosdaq_tickers = stock.get_market_ticker_list(today, market="KOSDAQ")

    rows = []
    for ticker in kospi_tickers:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"Code": ticker, "Name": name, "Market": MARKET_KOSPI})
    for ticker in kosdaq_tickers:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"Code": ticker, "Name": name, "Market": MARKET_KOSDAQ})

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """선택된 종목들의 종가 데이터를 가져온다.

    Args:
        tickers: 종목 코드 튜플
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)

    Returns:
        DataFrame with DatetimeIndex, columns=종목코드, values=종가

    Raises:
        ValueError: 데이터가 충분하지 않을 때
    """
    prices = pd.DataFrame()
    failed_tickers = []

    for ticker in tickers:
        series = _fetch_single_ticker(ticker, start_date, end_date)
        if series is not None and len(series) > 0:
            prices[ticker] = series
        else:
            failed_tickers.append(ticker)

    if prices.empty:
        raise ValueError("선택한 종목의 데이터를 가져올 수 없습니다.")

    if failed_tickers:
        raise ValueError(
            f"다음 종목의 데이터를 가져올 수 없습니다: {', '.join(failed_tickers)}"
        )

    # 각 종목별 데이터 시작일 확인
    ticker_starts = {col: prices[col].first_valid_index() for col in prices.columns}
    latest_start = max(ticker_starts.values())

    prices = prices.ffill().dropna()

    if len(prices) < MIN_DATA_DAYS:
        # 어떤 종목이 문제인지 안내
        short_tickers = [
            f"{ticker}(시작: {start.strftime('%Y-%m-%d')})"
            for ticker, start in ticker_starts.items()
            if start == latest_start
        ]
        raise ValueError(
            f"겹치는 데이터가 {len(prices)}일뿐입니다 (최소 {MIN_DATA_DAYS}일 필요). "
            f"가장 늦게 시작하는 종목: {', '.join(short_tickers)}. "
            f"해당 종목을 제외하거나 시작일을 조정해주세요."
        )

    return prices


def _fetch_single_ticker(
    ticker: str, start_date: str, end_date: str
) -> pd.Series | None:
    """단일 종목의 종가를 가져온다. FDR 실패 시 pykrx 폴백."""
    try:
        import FinanceDataReader as fdr

        df = fdr.DataReader(ticker, start_date, end_date)
        if df is not None and not df.empty:
            return df["Close"]
    except Exception:
        pass

    try:
        from pykrx import stock

        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        df = stock.get_market_ohlcv(start, end, ticker)
        if df is not None and not df.empty:
            return df["종가"]
    except Exception:
        pass

    return None


def search_stocks(query: str, stock_list: pd.DataFrame) -> pd.DataFrame:
    """종목명 또는 코드로 종목을 검색한다.

    Args:
        query: 검색어
        stock_list: fetch_stock_list의 결과

    Returns:
        매칭된 종목들의 DataFrame (새 객체)
    """
    if not query.strip():
        return stock_list.copy()

    query_lower = query.strip().lower()
    name_match = stock_list["Name"].str.lower().str.contains(query_lower, na=False)
    code_match = stock_list["Code"].str.startswith(query.strip())

    return stock_list[name_match | code_match].reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_data(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """종목별 배당 정보를 가져온다.

    Args:
        tickers: 종목 코드 튜플
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)

    Returns:
        DataFrame with columns: [Code, Name, DividendYield, Dividend]
        - DividendYield: 배당 수익률 (%)
        - Dividend: 주당 배당금 (원)
    """
    from pykrx import stock

    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    rows = []

    for ticker in tickers:
        try:
            df = stock.get_market_dividend_yield(start, end, ticker)
            if df is not None and not df.empty:
                # 가장 최근 배당 데이터 사용
                last_valid = df[df["배당수익률"] > 0]
                if not last_valid.empty:
                    latest = last_valid.iloc[-1]
                    rows.append({
                        "Code": ticker,
                        "DividendYield": float(latest["배당수익률"]),
                        "Dividend": float(latest["주당배당금"]),
                    })
                else:
                    rows.append({"Code": ticker, "DividendYield": 0.0, "Dividend": 0.0})
            else:
                rows.append({"Code": ticker, "DividendYield": 0.0, "Dividend": 0.0})
        except Exception:
            rows.append({"Code": ticker, "DividendYield": 0.0, "Dividend": 0.0})

    return pd.DataFrame(rows)


def validate_tickers(tickers: tuple[str, ...]) -> tuple[bool, str]:
    """선택된 종목 수가 유효한지 검증한다."""
    from src.constants import MAX_STOCK_COUNT, MIN_STOCK_COUNT

    if len(tickers) < MIN_STOCK_COUNT:
        return False, f"최소 {MIN_STOCK_COUNT}개 종목을 선택해주세요."
    if len(tickers) > MAX_STOCK_COUNT:
        return False, f"최대 {MAX_STOCK_COUNT}개까지 선택할 수 있습니다."
    return True, ""
