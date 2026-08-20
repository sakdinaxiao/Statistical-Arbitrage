import asyncio
import sys
import os
from dotenv import load_dotenv
# --- DASHBOARD START ---
from execution.dashboard import LiveDashboard
# --- DASHBOARD END ---
import numpy as np
from data.bybit_data import BybitService
from execution.executor import OrderExecutor
from execution.strategy import StatArbStrategy
from models.welford import WelfordZScore
from models.expected_value import ExpectedValueCalculator
from data.enums import State, Action
from models.cointegration import Cointegrate
from models.kalman import Kalman_2D
from models.welford import WelfordZScore
from execution.trade_logger import TradeLogger
from models.beta_spike import BetaChecker
import statsmodels.api as sm


class PairTrading:
    def __init__(self,key,secret,symbol_list=[],qty_y=0.01):
        self.ENTRY_PERCENTILE = 90
        self.STOPLOSS_GAP = 2.5
        self.MAX_BAR = 8
        self.TIMEFRAME = 180 # 3min
        self.DAYS = (30*1) #1months

        self.qty_y = qty_y
        self.key = key
        self.secret = secret
        self.symbol_list = symbol_list
        self.coin_x = symbol_list[0]
        self.coin_y = symbol_list[1]
        # Bumped to 0.1% (0.001) to act as a buffer for both exchange fees AND bid-ask slippage on market orders
        self.FEERATE = 0.001 
        
        self.window: int
        self.needs_reset = False # stoploss cooldown: bans re-entry until z-score cools off
        
        self.max_bars = int((self.MAX_BAR*60) / (self.TIMEFRAME/60)) #8 hours
    
        if len(self.symbol_list) != 2:
            print("main: symbol missing")
            return 
            
    async def initialize_account(self):
        self.bybit = BybitService(
            key=self.key,
            secret=self.secret,
            testnet=False,
            demo=True
        )
        
        self.rules = await self.bybit.get_instruments_info(self.symbol_list)
        
        status = self.bybit.get_account_status()
        positions = status.get("positions", []) if status else []
        self.recovered_state = self.bybit.sync_state(positions, self.symbol_list[1])
        
        print(f"main: Recovered state -> {self.recovered_state.name}")

        
        
    def update_dynamic_thresholds(self, spread_series):
        # --- DYNAMIC Z-SCORE START ---
        # startup passes the full-history OLS spread; the loop passes the live rolling spread
        # (called only while flat, so thresholds stay frozen while holding)
        spread_arr = np.asarray(spread_series, dtype=float)
        spread_mean = np.mean(spread_arr)
        spread_std = np.std(spread_arr)

        if spread_std == 0.0 or np.isnan(spread_std):
            self.dynamic_entry = 1.5
            self.dynamic_stoploss = 3.0
        else:
            z_scores = np.abs((spread_arr - spread_mean) / spread_std)
            percentile = np.percentile(z_scores, self.ENTRY_PERCENTILE)

            if np.isnan(percentile):
                self.dynamic_entry = 1.5
                self.dynamic_stoploss = 3.0
            else:
                self.dynamic_entry = max(1.2, min(percentile, 3.0))
                self.dynamic_stoploss = min(self.dynamic_entry + self.STOPLOSS_GAP,4.6)

        print(f"main: dynamically calculated entry z-score: {self.dynamic_entry:.3f}, stoploss: {self.dynamic_stoploss:.3f}")
        # --- DYNAMIC Z-SCORE END ---

    async def initialize_math_obj(self):
        past_price_x = await self.bybit.get_past_price(self.coin_x, str(self.TIMEFRAME//60), days=self.DAYS)
        past_price_y = await self.bybit.get_past_price(self.coin_y, str(self.TIMEFRAME//60), days=self.DAYS)

        if past_price_x is None or past_price_y is None:
            print("main: failed to initialize past price")
            raise Exception("Failed to fetch historical data")
            
        #Align timestamps chronologically so X and Y match perfectly
        common_times = sorted(set(past_price_x.keys()).intersection(set(past_price_y.keys())))

        #Extract prices into numpy arrays
        prices_x = np.array([past_price_x[t] for t in common_times])
        prices_y = np.array([past_price_y[t] for t in common_times])
        
        self.past_log_x = np.log(prices_x)
        self.past_log_y = np.log(prices_y)
        
        #get dynamic window
        ols_model = sm.OLS(self.past_log_y, sm.add_constant(self.past_log_x)).fit()                                                                       
        first_alpha = ols_model.params[0]                                                                                                   
        first_beta = ols_model.params[1]                                                                                                    
        historical_spread = self.past_log_y - (first_beta * self.past_log_x + first_alpha)      

        self.update_dynamic_thresholds(historical_spread)


        self.ev_calculator = ExpectedValueCalculator(self.qty_y, self.FEERATE,self.max_bars)
        first_hl = self.ev_calculator.cal_half_life(historical_spread)
        self.window = int(first_hl * 2)
        
        if len(common_times) < self.window:
            print(f"main: not enough overlapping historical data (found {len(common_times)})")
            raise Exception("Not enough historical data to initialize math objects")

        print(f"main: successfully prepared {len(common_times)} log prices")

        # test the short-term 8h stationarity window
        self.cointegration = Cointegrate(self.TIMEFRAME, self.past_log_x, self.past_log_y)
        self.cointegration.trade_stationary_flag = self.cointegration.spread_stationaryTest(self.past_log_x[-self.max_bars:], self.past_log_y[-self.max_bars:])
        if self.cointegration.trade_stationary_flag is False:
            print("main: stationary flag is False. Stopping the machine.")
            sys.exit(1)


        #kalman spreaed z_score 
        initx=self.past_log_x[-self.window:]
        inity=self.past_log_y[-self.window:]

        optimal_q, q_p_value = Kalman_2D.tune_q_frac(self.past_log_x,self.past_log_y,self.window)
        print(f"main: selected Kalman q_frac: {optimal_q:g}, Ljung-Box p-value: {q_p_value:.6f}")
        self.kalman = Kalman_2D(initx,inity,q_frac=optimal_q)

        self.welford = WelfordZScore(self.window,initx,inity,self.kalman.alpha,self.kalman.beta)

        self.beta_checker = BetaChecker(self.window)

        print("main: initialized math")

    
    def initialize_others(self):
        self.logger = TradeLogger(self.coin_x, self.coin_y, self.FEERATE)

        self.strategy = StatArbStrategy(
            self.coin_x,
            self.coin_y,
            entry=getattr(self, 'dynamic_entry', 1.5),
            stoploss=getattr(self, 'dynamic_stoploss', 4.0)
        )

        self.executor = OrderExecutor(
            self.bybit.session,
            qty_y=self.qty_y,
            logger=self.logger,
            rules=self.rules,
            initial_state=self.recovered_state
                                 )        
        
        print("main: initialized others")
        
    async def initialize_all(self):
        await self.initialize_account()
        await self.initialize_math_obj()
        self.initialize_others()
        
        print("main: initialized all")



    async def main(self):
        await self.initialize_all()

        # --- DASHBOARD SETUP START ---
        self.dashboard = LiveDashboard()
        self.dashboard.start()
        # --- DASHBOARD SETUP END ---

        while True:
            try:
                #grab current prices
                current_prices = await self.bybit.get_current_price(self.symbol_list)

                if current_prices is None:
                    print("main: prices error")
                    await asyncio.sleep(1)
                    continue

                raw_x = current_prices.get("prices", {}).get(self.coin_x)
                raw_y = current_prices.get("prices", {}).get(self.coin_y)

                if raw_x is None or raw_y is None:
                    print("main: prices error (symbols not found)")
                    await asyncio.sleep(1)
                    continue

                logPrice_x = np.log(raw_x)
                logPrice_y = np.log(raw_y)

                is_holding = (self.executor.state != State.NoPosition)

                #snapshot freeze: bands recalculate only while flat; an open trade
                #keeps the exact entry/stoploss it was entered with
                if not is_holding:
                    self.update_dynamic_thresholds(self.welford.get_spread_series())
                    self.strategy.entry = self.dynamic_entry

                alpha, beta , et = self.kalman.update(logPrice_x,logPrice_y)
                is_Beta_spike = self.beta_checker.beta_spike_check(beta)

                if is_Beta_spike:
                    print("main: beta spike detected. Retesting stationary and skipping entries.")
                    self.cointegration.force_retest(self.max_bars)

                z_score  = self.welford.z_score_cal(logPrice_x,logPrice_y,alpha,beta,is_holding, spread=et)


                #fresh half-life from the just-updated spread series drives the retest cadence
                fresh_half_life = self.ev_calculator.cal_half_life(self.welford.get_spread_series())

                #update staionary list (dynamic cadence: 1h flat, 3 * half-life while holding)
                self.cointegration.update(logPrice_x, logPrice_y, self.max_bars, is_holding=is_holding, half_life_bars=fresh_half_life)

                #check for stationary while trading
                is_trade_stat = self.cointegration.trade_stationary_flag
                if not is_trade_stat:
                    print("main: it's not stationary. Skipping entries.")

                self.strategy.stoploss = getattr(self, 'dynamic_stoploss', 4.0)

                # --- RESET RULE START ---
                # stoploss cooldown lifts only when z cools below half the entry line
                if self.needs_reset and z_score is not None and not np.isnan(z_score) and abs(z_score) < self.strategy.entry / 2.0:
                    self.needs_reset = False
                    print("main: z-score cooled off. Stoploss cooldown lifted.")
                # --- RESET RULE END ---

                signal = self.strategy.create_signal(
                    z_score=z_score,
                    beta=beta,
                    state=self.executor.state,
                    price_x=raw_x,
                    price_y=raw_y
                )
                

                if signal.action != Action.HOLD and signal.action != Action.INVALID:
                    is_entry = (signal.action in [Action.SY_LX, Action.SX_LY])

                    #stoploss cooldown bans all new entries until the z-score resets
                    if is_entry and self.needs_reset:
                        print("main: stoploss cooldown active. Blocking entry.")

                        # --- LOGGER WIRE START ---
                        if getattr(self, 'logger', None) is not None:
                            self.logger.log_blocked(signal, "stoploss cooldown active")
                        # --- LOGGER WIRE END ---

                        signal.action = Action.HOLD
                        is_entry = False

                    # force stationary test right before entry
                    if is_entry:
                        print("main: Forcing stationarity test before entry.")
                        self.cointegration.force_retest(self.max_bars)
                        is_trade_stat = self.cointegration.trade_stationary_flag

                    #check for blocking
                    if is_entry and (not is_trade_stat or is_Beta_spike):
                        print("main: Market non-stationary or beta spike. Blocking entry.")

                        # --- LOGGER WIRE START ---
                        if getattr(self, 'logger', None) is not None:
                            self.logger.log_blocked(signal, "Market non-stationary or beta spike")
                        # --- LOGGER WIRE END ---

                        signal.action = Action.HOLD
                        is_entry = False

                    if is_entry:
                        spread_series = self.welford.get_spread_series()                                          
                        profitable, ev = self.ev_calculator.assess(z_score, beta, raw_y, spread_series)

                        #check for EV
                        if not profitable:
                            print(f"main: Trade {signal.action.name} rejected EV is negative")

                            # --- LOGGER WIRE START ---
                            if getattr(self, 'logger', None) is not None:
                                self.logger.log_blocked(signal, "EV is negative")
                            # --- LOGGER WIRE END ---

                            signal.action = Action.HOLD
                    
                    #closing
                    if signal.action != Action.HOLD:                                                                              
                            await self.executor.execute_signal(signal)

                            #stoploss hit -> ban re-entry until the z-score resets
                            if signal.action == Action.EXIT_LOSS:
                                self.needs_reset = True
                                print("main: stoploss hit. Cooldown active until z-score resets.")

                # --- DASHBOARD UPDATE START ---
                try:
                    is_trade_stat = getattr(self, 'cointegration', None) and self.cointegration.trade_stationary_flag
                    pair_str = f"{self.coin_x} / {self.coin_y}"
                    
                    current_ev = None
                    try:
                        spread_series = self.welford.get_spread_series()
                        if len(spread_series) >= 2 and z_score is not None and not np.isnan(z_score) and not np.isnan(beta):
                            sigma = float(np.std(spread_series))
                            notional_y = self.qty_y * raw_y
                            expected_profit = abs(z_score) * sigma * notional_y
                            cost = self.FEERATE * notional_y * (1 + abs(beta)) * 2
                            current_ev = expected_profit - cost
                    except Exception:
                        pass
                        
                    self.dashboard.update(
                        self.executor.state.name, 
                        z_score, 
                        beta, 
                        is_trade_stat,
                        pair_str, 
                        raw_x, 
                        raw_y, 
                        current_ev, 
                        getattr(self, 'dynamic_entry', 1.5),
                        getattr(self, 'dynamic_stoploss', 4.0)
                    )
                except Exception:
                    pass
                # --- DASHBOARD UPDATE END ---

                await asyncio.sleep(self.TIMEFRAME)
            except Exception as e:
                import traceback
                print(f'CRITICAL ERROR in main loop: {e}')
                traceback.print_exc()
                await asyncio.sleep(5)
                continue


if __name__ == "__main__":
    load_dotenv()
    
    # Make sure these match the variable names in your .env file
    api_key = os.getenv("KEY")
    api_secret = os.getenv("SECRET")
    
    if not api_key or not api_secret:
        print("Error: API_KEY and API_SECRET must be set in the .env file.")
        sys.exit(1)
        
    # Define your pairs here cheaper one comefirst
    symbols = ["ETHUSDT", "BTCUSDT"]
    
    bot = PairTrading(key=api_key, secret=api_secret, symbol_list=symbols)
    
    try:
        asyncio.run(bot.main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
