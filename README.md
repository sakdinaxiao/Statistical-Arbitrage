# High-Frequency Statistical Arbitrage Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-orange.svg)
![NumPy](https://img.shields.io/badge/numpy-1.26%2B-lightblue.svg)
![Bybit](https://img.shields.io/badge/Exchange-Bybit-yellow.svg)

An institutional-grade, asynchronous statistical arbitrage (pairs trading) bot built for cryptocurrency perpetual futures on the Bybit exchange. 

Designed for robustness and speed, this engine identifies cointegrated cryptocurrency pairs, tracks their spread using a dynamically updating Kalman Filter, and executes mean-reversion trades based on online variance tracking and real-time Expected Value (EV) modeling. 

This project demonstrates advanced quantitative finance concepts, low-latency asynchronous execution, and rigorous risk management.

---

## 🚀 Core Features

### 1. Quantitative Pairs Discovery
The engine includes a robust pair-scanner designed to eliminate spurious correlations and discover true cointegration:
- **Stationarity Testing**: Employs both Augmented Dickey-Fuller (ADF) and Johansen Cointegration tests across multiple time horizons.
- **Out-of-Sample Validation**: Splits historical data into 70/30 train/test sets to validate the stationarity of out-of-sample spreads.
- **False Discovery Rate (FDR) Control**: Applies the **Benjamini-Hochberg (BH)** procedure to correct for multiple hypothesis testing, strictly limiting false positive pairings.

### 2. Real-Time Dynamic Modeling
Instead of relying on rigid, backward-looking moving averages, the bot adapts to market microstructure instantly:
- **2D Kalman Filter**: Continuously updates the hedge ratio (Beta) and spread intercept (Alpha) using a dynamic state-space model. The process noise covariance (Q) is auto-tuned by minimizing the Ljung-Box Q-statistic p-value on residuals.
- **Welford’s Online Algorithm**: Computes running mean and variance of the spread with $O(1)$ time complexity, ensuring ultra-low latency Z-score calculations without the overhead of array reallocations.

### 3. Asynchronous Execution Engine
Built on Python's `asyncio`, the execution layer ensures minimal slippage:
- **Concurrent Order Routing**: Leverages `asyncio.TaskGroup` to dispatch multi-leg orders (Long Asset X / Short Asset Y) simultaneously to the Bybit Unified Trading API.
- **Expected Value (EV) Gating**: Models the spread as an Ornstein-Uhlenbeck (AR1) mean-reverting process to calculate the spread's half-life. Entry orders are strictly blocked if the expected gross profit fails to exceed the simulated round-trip taker fees.

### 4. Advanced Risk & Volatility Management
Capital preservation is hardcoded into the pipeline:
- **Beta Spike Detection**: Utilizes an online Rolling Median Absolute Deviation (MAD) via dual priority heaps. If the Kalman Beta shifts too violently (indicating a structural break), trading is halted.
- **Dynamic Stationarity Halving**: Continuously runs ADF tests on the live spread. If the pair temporarily loses stationarity while holding a position, the stop-loss is dynamically tightened (halved) to force an early exit.
- **Stoploss Cooldown**: Hard-blocks immediate re-entries after a stop-out until the Z-score naturally mean-reverts to a safe baseline.

---

## 🛠️ System Architecture

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
│   └── data/                      # Pybit WebSocket/HTTP wrappers
```

---

## ⚙️ Installation & Usage

### Prerequisites
- **Python 3.10+**
- A Bybit API Key and Secret with Unified Trading (Linear Perpetual) permissions.

### Setup
1. Clone the repository and configure the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root directory:
   ```env
   KEY=your_bybit_api_key
   SECRET=your_bybit_api_secret
   ```

### 1. Run the Scanner
Identify the most statistically significant pairs currently trading on the exchange:
```bash
python src/scanner.py
```
*Note: The scanner will output pairs that pass the Benjamini-Hochberg FDR threshold, ranked by their out-of-sample joint p-values.*

### 2. Launch the Execution Engine
Update the target pair in `src/main.py` based on the scanner's output, then start the bot:
```bash
python src/main.py
```
The CLI dashboard will launch, streaming real-time Z-scores, Beta, EV, and active stationarity flags.

---

## 📊 Disclaimer
This software is provided for educational and portfolio demonstration purposes only. It is not financial advice. Quantitative models are subject to structural market breaks and execution risks (slippage, API latency, exchange downtime). **Trade at your own risk.**
