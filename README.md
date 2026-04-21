# 한국 주식 포트폴리오 매니저

한국 주식/ETF 종목을 선택하면 최적 포트폴리오 비중을 계산하고, 성과를 분석해주는 웹 대시보드입니다.

## 주요 기능

- **종목 선택**: KOSPI, KOSDAQ, ETF 3,970개 종목 지원
- **포트폴리오 최적화**: 6가지 전략
  - 최대 샤프 비율 (Max Sharpe)
  - 최소 변동성 (Min Volatility)
  - 리스크 패리티 (Risk Parity)
  - 동일 비중 (Equal Weight)
  - HRP - 계층적 위험 배분 (Hierarchical Risk Parity)
  - Black-Litterman (시장 균형)
- **성과 분석**: Sharpe, Sortino, MDD, VaR, Calmar 등
- **배당금 추적**: 종목별 배당 수익률, 포트폴리오 전체 배당 수익률
- **리밸런싱 알림**: 밴드 기반 비중 이탈 감지
- **차트**: 누적 수익률, 자산 배분 파이차트, 월별 수익률 히트맵, Drawdown
- **리포트**: QuantStats HTML 리포트 다운로드

## 사용된 오픈소스

| 라이브러리 | 라이선스 | 용도 |
|-----------|---------|------|
| [Streamlit](https://streamlit.io/) | Apache 2.0 | 웹 대시보드 UI |
| [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) | MIT | 한국 주식 가격 데이터 수집 |
| [pykrx](https://github.com/sharebook-kr/pykrx) | MIT | FDR 실패 시 가격 데이터 폴백 |
| [yfinance](https://github.com/ranaroussi/yfinance) | Apache 2.0 | 배당 데이터 수집 |
| [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | MIT | Max Sharpe, Min Vol, HRP, Black-Litterman 최적화 |
| [scipy](https://scipy.org/) | BSD | Risk Parity 최적화 |
| [QuantStats](https://github.com/ranaroussi/quantstats) | Apache 2.0 | 성과 지표 계산, HTML 리포트 |
| [Plotly](https://plotly.com/) | MIT | 인터랙티브 차트 |
| [pandas](https://pandas.pydata.org/) | BSD | 데이터 처리 |
| [numpy](https://numpy.org/) | BSD | 수치 연산 |

## 프로젝트 구조

```
portfolio-manager/
├── app.py                 # Streamlit 메인 앱 (UI)
├── src/
│   ├── constants.py       # 상수 정의
│   ├── data.py            # 데이터 수집 (FDR, pykrx, yfinance)
│   ├── optimizer.py       # 포트폴리오 최적화 (6가지 전략)
│   ├── analyzer.py        # 성과 분석 (QuantStats)
│   ├── rebalancer.py      # 리밸런싱 알림
│   └── charts.py          # Plotly 차트 생성
├── data/
│   └── stock_list.csv     # 한국 종목 리스트 (KOSPI+KOSDAQ+ETF)
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/louis1004/portfolio-manager.git
cd portfolio-manager

# 의존성 설치
pip install -r requirements.txt

# 실행
streamlit run app.py
```

## 배포

[Streamlit Cloud](https://portfolio-manager-louis.streamlit.app/)에 배포되어 있습니다.

## 아키텍처

```
사용자 브라우저
    ↓
Streamlit Cloud (app.py)
    ↓
┌─────────────────────────────────────┐
│ data.py                             │
│ ├── 종목 리스트: 번들 CSV           │
│ ├── 주가 데이터: FinanceDataReader  │
│ │   └── 폴백: pykrx                │
│ └── 배당 데이터: yfinance           │
├─────────────────────────────────────┤
│ optimizer.py                        │
│ ├── PyPortfolioOpt (4가지 전략)     │
│ └── scipy (Risk Parity)            │
├─────────────────────────────────────┤
│ analyzer.py  → QuantStats          │
│ rebalancer.py                       │
│ charts.py    → Plotly              │
└─────────────────────────────────────┘
```

## 사용법

1. 사이드바에서 시장 선택 (전체 / KOSPI / KOSDAQ / ETF)
2. 종목 2~20개 선택
3. 분석 기간 설정 (시작/종료 년월)
4. 최적화 전략 선택
5. "분석 시작" 클릭
6. 탭에서 결과 확인
   - 포트폴리오 비중
   - 성과 분석
   - 월별 수익률
   - 배당금
   - 리밸런싱
   - 리포트 다운로드
