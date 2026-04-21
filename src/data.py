"""한국 주식 데이터 수집 모듈."""

import pandas as pd
import streamlit as st

from src.constants import (
    MARKET_ALL,
    MARKET_ETF,
    MARKET_KOSDAQ,
    MARKET_KOSPI,
    MAX_STOCK_COUNT,
    MIN_DATA_DAYS,
    MIN_STOCK_COUNT,
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_list(market: str = MARKET_ALL) -> pd.DataFrame:
    """한국 주식 종목 리스트를 가져온다.

    Args:
        market: "KOSPI", "KOSDAQ", "ETF", 또는 "전체"

    Returns:
        DataFrame with columns: [Code, Name, Market]
    """
    # 1차: 번들 CSV (항상 동작)
    result = _load_bundled_stock_list()

    # CSV가 없으면 온라인 시도
    if result.empty:
        errors = []
        try:
            result = _fetch_via_fdr()
        except Exception as e:
            errors.append(f"FDR: {e}")

        if result.empty:
            try:
                result = _fetch_stock_list_fallback()
            except Exception as e:
                errors.append(f"pykrx: {e}")

        if result.empty:
            raise ValueError(
                f"종목 리스트를 가져올 수 없습니다. {'; '.join(errors)}"
            )

    # 시장 필터
    if market != MARKET_ALL:
        filtered = result[result["Market"] == market]
        if not filtered.empty:
            result = filtered

    return result.sort_values("Name").reset_index(drop=True)


def _empty_stock_df() -> pd.DataFrame:
    """빈 종목 DataFrame을 생성한다."""
    return pd.DataFrame(columns=["Code", "Name", "Market"])


def _load_bundled_stock_list() -> pd.DataFrame:
    """번들된 CSV 파일에서 종목 리스트를 로드한다."""
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "stock_list.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        return _empty_stock_df()

    try:
        df = pd.read_csv(csv_path, dtype={"Code": str, "Name": str, "Market": str})
        df["Code"] = df["Code"].str.zfill(6)
        return df
    except Exception:
        return _empty_stock_df()


def _fetch_via_fdr() -> pd.DataFrame:
    """FinanceDataReader로 종목 리스트를 가져온다."""
    import FinanceDataReader as fdr

    parts = []

    # 주식: 여러 listing 방식 시도
    for listing_type in ["KRX", "KOSPI", "KOSDAQ"]:
        try:
            df = fdr.StockListing(listing_type)
            if df is not None and not df.empty:
                normalized = _normalize_any_df(df, listing_type)
                if not normalized.empty:
                    parts.append(normalized)
                    if listing_type == "KRX":
                        break  # KRX로 전체를 가져왔으면 개별 시장은 불필요
        except Exception:
            continue

    # ETF
    try:
        etf_df = fdr.StockListing("ETF/KR")
        if etf_df is not None and not etf_df.empty:
            etfs = _normalize_any_df(etf_df, MARKET_ETF)
            if not etfs.empty:
                parts.append(etfs)
    except Exception:
        pass

    if not parts:
        raise ValueError("FDR에서 데이터를 가져올 수 없습니다.")

    return pd.concat(parts, ignore_index=True)


def _normalize_any_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """어떤 형태의 DataFrame이든 Code/Name/Market으로 정규화한다."""
    # 코드 컬럼 찾기
    code_col = None
    for candidate in ["Code", "Symbol", "ISU_SRT_CD", "종목코드", "Ticker"]:
        if candidate in df.columns:
            code_col = candidate
            break

    # 이름 컬럼 찾기
    name_col = None
    for candidate in ["Name", "ISU_ABBRV", "종목명", "종목"]:
        if candidate in df.columns:
            name_col = candidate
            break

    if code_col is None or name_col is None:
        return _empty_stock_df()

    # 시장 컬럼 찾기
    market_col = None
    for candidate in ["Market", "MKT_NM", "시장구분"]:
        if candidate in df.columns:
            market_col = candidate
            break

    result = pd.DataFrame({
        "Code": df[code_col].astype(str).str.zfill(6),
        "Name": df[name_col].astype(str),
    })

    if source == MARKET_ETF:
        result["Market"] = MARKET_ETF
    elif market_col:
        result["Market"] = df[market_col].astype(str)
    elif source in (MARKET_KOSPI, "KOSPI"):
        result["Market"] = MARKET_KOSPI
    elif source in (MARKET_KOSDAQ, "KOSDAQ"):
        result["Market"] = MARKET_KOSDAQ
    else:
        result["Market"] = MARKET_KOSPI

    return result


def _fetch_stock_list_fallback() -> pd.DataFrame:
    """pykrx를 사용한 종목 리스트 폴백."""
    from pykrx import stock
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    rows = []

    try:
        kospi_tickers = stock.get_market_ticker_list(today, market="KOSPI")
        for ticker in kospi_tickers:
            name = stock.get_market_ticker_name(ticker)
            rows.append({"Code": ticker, "Name": name, "Market": MARKET_KOSPI})
    except Exception:
        pass

    try:
        kosdaq_tickers = stock.get_market_ticker_list(today, market="KOSDAQ")
        for ticker in kosdaq_tickers:
            name = stock.get_market_ticker_name(ticker)
            rows.append({"Code": ticker, "Name": name, "Market": MARKET_KOSDAQ})
    except Exception:
        pass

    if not rows:
        raise ValueError("pykrx에서도 데이터를 가져올 수 없습니다.")

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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividend_data(tickers: tuple[str, ...]) -> pd.DataFrame:
    """종목별 최신 배당 정보를 yfinance에서 가져온다.

    Args:
        tickers: 종목 코드 튜플

    Returns:
        DataFrame with columns: [Code, DividendYield, Dividend]
        - DividendYield: 배당 수익률 (소수, 예: 0.035 = 3.5%)
        - Dividend: 연간 주당 배당금 (원)
    """
    rows = []
    for ticker in tickers:
        div_yield, div_amount = _fetch_single_dividend(ticker)
        rows.append({
            "Code": ticker,
            "DividendYield": div_yield,
            "Dividend": div_amount,
        })

    return pd.DataFrame(rows)


def _fetch_single_dividend(ticker: str) -> tuple[float, float]:
    """단일 종목의 배당 정보를 가져온다. 여러 방법을 순차 시도.

    Returns:
        (배당수익률(%), 연간배당금(원))
    """
    import yfinance as yf

    for suffix in [".KS", ".KQ"]:
        try:
            yf_ticker = yf.Ticker(f"{ticker}{suffix}")

            # 방법 1: .info에서 직접 가져오기
            info = yf_ticker.info or {}
            div_yield = info.get("dividendYield", 0) or 0
            div_rate = info.get("dividendRate", 0) or 0

            if div_yield > 0 or div_rate > 0:
                return float(div_yield), float(div_rate)

            # 방법 2: 배당 히스토리에서 계산
            dividends = yf_ticker.dividends
            if dividends is not None and not dividends.empty:
                # 최근 1년간 배당금 합산
                dividends.index = dividends.index.tz_localize(None)
                one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
                recent = dividends[dividends.index >= one_year_ago]
                annual_div = float(recent.sum()) if not recent.empty else float(dividends.tail(4).sum())

                if annual_div > 0:
                    # 현재 주가로 수익률 계산
                    hist = yf_ticker.history(period="5d")
                    if hist is not None and not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                        if price > 0:
                            return annual_div / price, annual_div

        except Exception:
            continue

    return 0.0, 0.0


def fetch_market_caps(tickers: tuple[str, ...]) -> dict[str, float]:
    """종목별 시가총액을 CSV에서 가져온다.

    Args:
        tickers: 종목 코드 튜플

    Returns:
        종목코드 -> 시가총액 딕셔너리
    """
    stock_list = _load_bundled_stock_list()

    if "Marcap" not in stock_list.columns:
        return {ticker: 1.0 for ticker in tickers}

    marcap_map = dict(zip(stock_list["Code"], stock_list["Marcap"]))
    result = {}
    for ticker in tickers:
        cap = marcap_map.get(ticker, 0)
        result[ticker] = float(cap) if cap and cap > 0 else 1.0

    return result


def validate_tickers(tickers: tuple[str, ...]) -> tuple[bool, str]:
    """선택된 종목 수가 유효한지 검증한다."""
    if len(tickers) < MIN_STOCK_COUNT:
        return False, f"최소 {MIN_STOCK_COUNT}개 종목을 선택해주세요."
    if len(tickers) > MAX_STOCK_COUNT:
        return False, f"최대 {MAX_STOCK_COUNT}개까지 선택할 수 있습니다."
    return True, ""
