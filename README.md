# Statistical Arbitrage Demo

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-orange.svg)
![NumPy](https://img.shields.io/badge/numpy-2.5.1-lightblue.svg)
![Bybit](https://img.shields.io/badge/Exchange-Bybit-yellow.svg)

An experimental statistical arbitrage (pairs trading) bot for cryptocurrency perpetual futures using Bybit Demo Trading.

The project screens historical cryptocurrency pairs, tracks their spread with a Kalman filter, and evaluates mean-reversion entries using rolling variance and an estimated expected value (EV).

The trading loop polls HTTP prices approximately every three minutes, plus processing time. This is a portfolio and research demo; live-money operation and profitability have not been validated.

---

## Core Features

### 1. Quantitative Pairs Discovery
The scanners evaluate historical relationships between pairs:
- **Stationarity Testing**: `scanner.py` applies Augmented Dickey-Fuller (ADF) tests across two timeframes; `structural_scanner.py` provides a separate Johansen screen.
- **Out-of-Sample Validation**: Splits historical data into 70/30 train/test sets to validate the stationarity of out-of-sample spreads.
- **False Discovery Rate (FDR) Control**: Applies the **Benjamini-Hochberg (BH)** procedure to adjust p-values for multiple hypothesis testing.

### 2. Real-Time Dynamic Modeling
The execution loop updates these models on each price sample:
- **2D Kalman Filter**: Continuously updates the hedge ratio (Beta) and spread intercept (Alpha) using a dynamic state-space model. The process noise scale is selected using Ljung-Box residual p-values: the first candidate above 0.05, or the highest p-value if none passes.
- **Welford’s Online Algorithm**: Updates the rolling mean and variance of the spread in $O(1)$ time. Other calculations still operate on the full window.

### 3. Asynchronous Execution Engine
The execution layer uses Python's `asyncio` to submit the two legs concurrently:
- **Concurrent Order Routing**: Uses `asyncio.gather` and worker threads to submit both legs. REST acknowledgements are followed by order-status polling and position reconciliation before position state changes. Concurrent submission does not make the two orders atomic.
- **Expected Value (EV) Gating**: Models the spread as an Ornstein-Uhlenbeck (AR1) mean-reverting process to calculate the spread's half-life. Entry orders are strictly blocked if the expected gross profit fails to exceed the simulated round-trip taker fees.

### 4. Entry Filters and Exit Rules
The demo includes these checks; they do not guarantee loss prevention:
- **Beta Spike Detection**: Utilizes an online Rolling Median Absolute Deviation (MAD) via dual priority heaps. If the Kalman Beta shifts too violently (indicating a structural break), new entries are blocked for that iteration.
- **Dynamic Stationarity Halving**: Continuously runs ADF tests on the live spread. If the pair temporarily loses stationarity while holding a position, the stop-loss is dynamically tightened (halved) to force an early exit.
- **Stoploss Cooldown**: Hard-blocks immediate re-entries after a stop-out until the Z-score naturally mean-reverts to a safe baseline.

---

## System Architecture

```text
statArb/
├── src/
│   ├── main.py                    # Main event loop and Live Dashboard
│   ├── scanner.py                 # Multi-timeframe ADF pair screener
│   ├── structural_scanner.py      # Johansen Cointegration screener
│   ├── execution/
│   │   ├── executor.py            # Async Pybit order routing and lot sizing
│   │   ├── strategy.py            # State-machine for Long/Short transitions
│   │   ├── dashboard.py           # Real-time CLI terminal UI
│   │   └── trade_logger.py        # CSV trade journal and PnL tracker
│   ├── models/
│   │   ├── kalman.py              # 2D Kalman Filter for Alpha/Beta
│   │   ├── cointegration.py       # Live ADF testing and stationarity checks
│   │   ├── expected_value.py      # Half-life and EV calculation
│   │   ├── beta_spike.py          # MAD outlier detection for structural breaks
│   │   └── welford.py             # O(1) Rolling Variance for Z-Scores
│   └── data/                      # Pybit HTTP wrappers
```

---

## Installation & Usage

### Prerequisites
- **Python 3.12+** (local regression suite tested on Python 3.12)
- Bybit **Demo Trading** API credentials with Unified Trading (linear perpetual) permissions. The runner sets `demo=True`, `testnet=False`; mainnet and testnet credentials are not interchangeable with demo credentials.
- Use one-way position mode and dedicate the selected symbols to this bot. Both symbols must be flat at startup; do not trade them manually or with another bot while it runs.

### Setup
1. Clone the repository and configure the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and replace the placeholders with demo credentials. `.env` is ignored by Git:
   ```env
   KEY=your_bybit_demo_api_key
   SECRET=your_bybit_demo_api_secret
   ```

### 1. Run the Scanner
Identify the most statistically significant pairs currently trading on the exchange:
```bash
python src/scanner.py
```
*Note: The scanner will output pairs that pass the Benjamini-Hochberg FDR threshold, ranked by their adjusted out-of-sample joint p-values. This scanner uses public market data and does not require your API credentials.*

### 2. Launch the Execution Engine
Update the target pair in `src/main.py` based on the scanner's output, then start the bot:
```bash
python src/main.py
```
The CLI dashboard prints Z-scores, beta, estimated EV, and stationarity flags on each iteration.

### Execution limits and recovery
- Startup requires successful instrument and account queries and no open position on either selected symbol. An existing pair is rejected because entry model state is not persisted.
- New entries require no existing positions or open orders on the selected symbols.
- Confirmed partial or cancelled entries are unwound using actual exchange position sizes. Exits use those sizes rather than recalculating quantities from the current beta.
- Unknown order outcomes, failed unwinds, or position mismatches halt the process. Halting does **not** guarantee positions are closed. Inspect both symbols and their open orders in Bybit, reconcile them manually, and only then restart. Do not configure automatic restart.
- Stop-loss decisions run in the polling loop; there are no exchange-native protective stop orders. Interrupting the process does not close positions.
- CSV PnL uses sampled prices and estimated fees, not an audited execution ledger.

### Tests
```bash
python -m unittest discover -s tests -v
python -m pip check
```
Tests use an offline exchange double; they do not read `.env` or submit real orders. GitHub Actions runs them on Python 3.12. Direct dependency versions in `requirements.txt` match the tested environment; transitive dependencies are not locked.

The order confirmation flow follows Bybit’s [place-order](https://bybit-exchange.github.io/docs/v5/order/create-order) and [order-status](https://bybit-exchange.github.io/docs/v5/order/open-order) APIs. No end-to-end exchange execution test has been performed as part of this publication cleanup.

---

## Disclaimer
This software is provided for educational and portfolio demonstration purposes only. It is not financial advice. Quantitative models are subject to structural market breaks and execution risks (slippage, API latency, exchange downtime). **Trade at your own risk.**
