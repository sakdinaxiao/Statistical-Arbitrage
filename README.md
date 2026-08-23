# StatArb: Statistical Arbitrage Bot for Bybit

StatArb is a pair-trading bot that executes statistical arbitrage strategies on Bybit linear perpetual contracts. It tracks the spread between two cointegrated cryptocurrencies using a Kalman Filter and executes trades when the spread dynamically diverges beyond statistical thresholds. 

The repository includes a rigorous pair scanner for discovery and a real-time execution engine for trading.

## Features

- **Pair Scanner (`scanner.py`)**: Evaluates crypto universes to find highly cointegrated pairs. Conducts ADF (Augmented Dickey-Fuller) stationarity tests over fast/slow timeframes and applies Benjamini-Hochberg (BH) False Discovery Rate (FDR) corrections to eliminate false positives.
- **Dynamic Spread Tracking**: Uses a 2D Kalman Filter to continuously update the spread (beta and alpha) and Welford's online algorithm to maintain accurate rolling variance and Z-scores.
- **Adaptive Execution**: 
  - Dynamic entry thresholds based on historical Z-score percentiles.
  - Live stationarity re-testing (aborting trades if the spread becomes non-stationary).
  - Beta spike detection to prevent entries during volatile structural breaks.
  - Expected Value (EV) checks to ensure trades beat exchange fees before entry.
- **Live Terminal Dashboard**: A rich CLI dashboard that displays Z-scores, Beta, live Expected Value (EV), stationarity flags, and current positions.

## Prerequisites

- Python 3.10+
- A Bybit API Key and Secret (Linear Perpetual permissions required).

## Installation

1. Clone the repository and navigate into it.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies (e.g., `numpy`, `statsmodels`, `python-dotenv`):
   ```bash
   pip install numpy statsmodels python-dotenv
   ```

4. Set up your environment variables:
   Create a `.env` file in the root directory and add your Bybit credentials:
   ```env
   KEY=your_bybit_api_key
   SECRET=your_bybit_api_secret
   ```

## Usage

### 1. Scanning for Pairs
Before trading, you should run the scanner to find cointegrated pairs that are statistically significant and tradeable:
```bash
python src/scanner.py
```
This will output a list of valid pairs (e.g., `ETHUSDT` / `BTCUSDT`) ranked by their BH FDR adjusted p-values.

### 2. Live Trading
Once you have identified a tradeable pair, update the `symbols` list in `src/main.py` (line ~368):
```python
symbols = ["ETHUSDT", "BTCUSDT"] # Put the cheaper coin first
```
Then start the trading bot:
```bash
python src/main.py
```

## Strategy Details

- **Entry Logic**: The bot goes long on the underperforming asset and short on the outperforming asset when the spread's Z-score crosses the dynamic entry threshold.
- **Exit Logic**: Positions are closed when the spread reverts to the mean (Z-score hits the profit target) or if the stop-loss threshold is breached.
- **Safety Measures**: 
  - Stoploss cooldown: If stopped out, the bot pauses entries until the Z-score cools down to prevent continuous losses.
  - Hedged Sizes: The position sizing is beta-neutralized.

## Disclaimer
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
