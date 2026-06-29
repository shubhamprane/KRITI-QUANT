# KRITI-QUANT

A systematic, long-only equity trading strategy built for **KRITI 2026**, targeting the Indian equity market (NSE). The system employs a **Dual-Sleeve Hybrid Architecture** that combines an IPO Penny Stock Momentum engine with a robust Convex Portfolio Optimization core, delivering exceptional risk-adjusted returns on a 10-year backtest.

---

## Performance Highlights (2010–2020 Backtest)

| Metric | Strategy | Nifty 500 |
|---|---|---|
| CAGR | **37.43%** | 8.61% |
| Annualized Volatility | 17.39% | 15.05% |
| Maximum Drawdown | **-25.08%** | -31.82% |
| Sharpe Ratio | **1.947** | 0.633 |
| Sortino Ratio | 2.301 | — |
| Information Ratio | 1.274 | — |
| Up-Capture vs Nifty 500 | 52.9% | — |
| Down-Capture vs Nifty 500 | **19.6%** | — |
| Annualized Turnover | 14.3x | — |

**Rolling Outperformance vs Nifty 500**

| Window | Avg Outperformance | Worst Underperformance |
|---|---|---|
| 1-Year | 33.3% | -0.7% |
| 3-Year | 190.4% | +77.7% |
| 5-Year | 582.5% | +307.2% |

---

## Architecture Overview

The strategy bifurcates capital into two independent, uncorrelated engines:

```
Total Capital (₹50,00,000)
├── IPO Momentum Sleeve        (default 30%)
│   ├── Universe: IPOs ≤ ₹40/share, 5-day stabilization rule
│   ├── Trailing stop-loss (20% from HWM)
│   ├── Hierarchical 3-tier priority queue
│   └── Drawdown-triggered liquidation (30% → 50% exit, 40% → full exit)
│
└── Convex Portfolio Optimization Sleeve  (default 70%)
    ├── Multi-factor stock selector (momentum, volatility, liquidity, beta)
    ├── Hysteresis-based turnover filter (entry rank ≤ 65, exit rank ≥ 79)
    ├── Distributionally robust convex optimization (CVXPY)
    └── Multi-period receding-horizon policy (H=2)
```

> If the Day-1 universe exceeds 1,200 stocks, allocation shifts defensively to 10% IPO / 90% Core.

---

## Strategy Components

### 1. Stock Selector (Core Sleeve)

A cross-sectional multi-factor ranking model that scores every stock daily using a weighted composite:

```
S = 0.30·Z(Mom60) + 0.20·Z(Mom20) − 0.30·Z(Vol60) + 0.10·Z(Liq20) + 0.05·Z(NATR14) + 0.05·Z(Beta60)
```

All features are Z-scored cross-sectionally and clipped to [−3, 3] to suppress outliers.

A **hysteresis filter** (enter at rank ≤ 65, exit at rank ≥ 79) controls turnover at the universe boundary, reducing unnecessary churn against the 0.536% round-trip transaction cost.

### 2. Convex Portfolio Optimization

At each weekly rebalancing node, the optimizer solves a **distributionally robust** allocation problem:

```
w* = argmax { α·μᵀw − λ·‖ρ⊙w‖₁ − γ·[max(wᵀΣ_diag·w, wᵀΣ_factor·w) + η·‖Λw‖₁²] }
```

Calibrated parameters: `α=1, λ=0.05, γ=15, η=0.01`

Constraints enforce long-only, no leverage (‖w‖₁ ≤ 1), and per-stock cap of 20%.

A **two-period receding-horizon** extension looks ahead one step to account for future transaction costs, only executing the first-period trade at each date.

### 3. IPO Momentum Sleeve

Captures high-alpha breakouts in micro-cap IPOs:

- **5-day stability rule** — enters only after initial listing-day volatility subsides
- **Price ceiling of ₹40/share** — targets high price-elasticity penny stocks
- **Dynamic slot capacity** — scales from 5 to 20 slots based on backtest duration
- **Hierarchical priority queue** — Fresh IPOs → Missed Opportunities (LIFO) → TSL-Hit Recovery (dual hysteresis re-entry)
- **Asset-level trailing stop** at 20% drawdown from high-water mark
- **Sleeve-level drawdown circuit breakers** at 30% and 40% with automatic capital reallocation to the core sleeve

---

## Repository Structure

```
KRITI-QUANT/
├── Kriti_Quant.py                    # Main strategy implementation
├── Approach_Kriti_Quant.pdf          # Full methodology & documentation report
├── Performance_Report_Kriti_Quant.pdf # Backtest performance report with charts
└── README.md
```

**Outputs** (generated at runtime in `strategy_output/`):

```
strategy_output/
├── equity_curve.png
├── drawdown_curve.png
├── position_count.png
├── turnover.png
├── rolling_outperformance.png
├── monthly_heatmap.png
├── trade_log.csv
├── daily_nav.csv
├── rolling_outperformance.csv
├── metrics_summary.csv
└── performance_report.txt
```

---

## Setup & Usage

### Requirements

```bash
pip install cvxpy cvxportfolio pandas numpy matplotlib
```

### Data

The strategy expects NSE daily OHLCV data in one of the following formats:

- `nse_prices_complete 1.parquet` *(primary)*
- `nse_prices_complete.parquet`
- `nse_prices_complete.csv`

Supported column schemas (auto-detected):

| Dev Format | Competition Format |
|---|---|
| `tradedate` | `date` |
| `fid` | `symbol` |
| `open/high/low/close` | `open/high/low/close` |
| `traded_volume` | `volume` |
| `traded_value` | `value` |
| `in_NSE500` | `in_nse500` |

### Run

```bash
python Kriti_Quant.py
```

The script will auto-detect the data format, print a live progress log, and write all outputs to `strategy_output/`.

---

## Key Design Decisions

**Execution price** is the OHLC average `(O+H+L+C)/4` rather than close-only, reducing sensitivity to intraday noise. All trades settle at T+1.

**Transaction cost** is modeled at 0.268% one-way (0.536% round-trip), applied proportionally to traded value. Cash balances are prevented from going negative via a floor in the cash recursion.

**Integer share constraint** — continuous optimal weights are floored to whole shares; no fractional ownership.

**Covariance estimation** uses a factor model (`cvxportfolio`) plus a diagonal fallback; the optimizer penalizes the *worst-case* across both, preventing over-reliance on a single risk model.

---

## Competition Constraints (KRITI 2026)

- Initial capital: ₹50,00,000
- Portfolio size: 1–100 stocks at all times
- No fractional shares
- Long-only (no shorts, no leverage)
- Up to 100% cash allowed
- Round-trip transaction cost: 0.536%
- Must hold at least 1 position every day (buys cheapest available stock if needed)

---
