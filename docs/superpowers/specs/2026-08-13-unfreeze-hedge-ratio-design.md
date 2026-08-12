# Fix 3: Unfreeze the Hedge Ratio Design

## Context
The Kalman filter currently uses a hardcoded process noise (`Q`) scaling factor of `1e-5`. This is too small, causing the filter to under-adapt (residuals are autocorrelated, beta lags reality). The optimal `q_frac` is pair-specific. We need to introduce it as a parameter and automatically tune it on bot startup so it dynamically adapts to the specific pair's historical data.

## Architecture & Components

### 1. `Kalman_2D` Constructor (`src/models/kalman.py`)
- Change `__init__(self, initx, inity)` to `__init__(self, initx, inity, q_frac=1e-3)`.
- Update `self.Q` matrix calculation to use `q_frac` instead of `1e-5`.
  ```python
  self.Q = np.array([[self.p[0,0] * q_frac, 0],
                     [0, self.p[1,1] * q_frac]])
  ```
- The rest of the QR update and square-root filter logic remains untouched.

### 2. Tuning Method (`src/models/kalman.py`)
- Add a `@classmethod` to `Kalman_2D` called `tune_q_frac(cls, logx, logy, warmup_len)`.
- **Sweep Logic**:
  - Iterate through a list of candidate scales: `[1e-5, 1e-4, 3e-4, 1e-3, 3e-3]`.
  - For each `q`:
    - Initialize a temporary `Kalman_2D` instance using `logx[:warmup_len]` and `logy[:warmup_len]`, with `q_frac=q`.
    - Loop from `warmup_len` to the end of the arrays, calling `.update()` and storing the residuals (`et`) and `beta` values.
    - Run the Ljung-Box test on the residuals: `acorr_ljungbox(ets, lags=[20], return_df=True)`.
  - **Selection Criteria**:
    - Pick the *smallest* `q_frac` where the Ljung-Box p-value > 0.05 (residuals are white).
    - If no `q_frac` clears 0.05, fallback to the one with the highest p-value.
  - Return the selected `q_frac` and its p-value so startup can log both.

### 3. Startup Integration (`src/main.py`)
- Inside `PairTrading.initialize_math_obj()`:
  - After calculating `self.window` (the warmup length), call `Kalman_2D.tune_q_frac(self.past_log_x, self.past_log_y, self.window)`.
  - Store the resulting optimal `q_frac`.
  - Pass this optimal `q_frac` into the initialization of the bot's live `self.kalman` object:
    `self.kalman = Kalman_2D(initx, inity, q_frac=optimal_q)`.

## Trade-offs & Error Handling
- **Performance impact**: Running the sweep over 30 days of 3m candles for 5 different `q` values will add a few seconds to the bot startup time. Since this only happens at startup, this is acceptable.
- **Welford Z-Score Constraints**: As specified, `welford.py` continues to evaluate open positions against `frozen_alpha` / `frozen_beta` from entry. The tuned live beta adapts the model, but exit goalposts do not move mid-trade.

## Testing
- Ensure `main.py` properly logs the chosen `q_frac` and its p-value on startup for transparency.
