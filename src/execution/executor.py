from data.enums import Action,State
from data.dataclasses import Signal
import asyncio
from decimal import Decimal, ROUND_DOWN

class OrderExecutor:
    def __init__(self,session,qty_y,logger, rules=None, initial_state=State.NoPosition):
        self.session = session
        self.state = initial_state
        self.qty_y = qty_y
        self.logger = logger
        self.rules = rules or {}
        
    def _round_qty(self, qty, symbol):
        step, min_qty = self.rules.get(symbol, ("0.001", "0.001"))
        step_d = Decimal(str(step))
        min_qty_d = Decimal(str(min_qty))
        
        q = (Decimal(str(qty)) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
        
        if q < min_qty_d:
            q = min_qty_d
            
        return q
    
    async def _place_order(self,side,symbol,qty,reduceOnly=False):
        try:
            # pybit HTTP is blocking; run it off the event loop so the two legs
            # placed via TaskGroup actually fire in parallel.
            order = await asyncio.to_thread(
                self.session.place_order,
                category="linear",
                symbol=symbol,
                side=side,
                qty=str(qty),
                timeInForce="IOC",
                orderType="Market",
                reduceOnly=reduceOnly,
            )

            return symbol, order
        except Exception as e:
            print(f"excutor: Error: {e}")

    async def execute_signal(self,signal: Signal):
        action = signal.action
        
        if action in (Action.EXIT_LOSS, Action.EXIT_PROFIT, Action.EXIT_REGIME):
            if self.state == State.short_y:
                y_side = "Buy"          # close SHORT y
            elif self.state == State.long_y:
                y_side = "Sell"         # close LONG y
            else:
                print("executor: exit signal but no open position -- skipping")
                return None
            reduceOnly = True
            new_state = State.NoPosition

        elif action == Action.SY_LX:
            if self.state != State.NoPosition:
                print("executor: entry signal but already in position -- skipping")
                return None
            y_side = "Sell"
            reduceOnly = False
            new_state = State.short_y

        elif action == Action.SX_LY:
            if self.state != State.NoPosition:
                print("executor: entry signal but already in position -- skipping")
                return None
            y_side = "Buy"
            reduceOnly = False
            new_state = State.long_y

        if action == Action.HOLD or action == Action.INVALID:
            return None
        
        x_side = "Sell" if y_side == "Buy" else "Buy"

        #lot size rounding 
        raw_qty_x = abs(signal.beta) * self.qty_y * (signal.price_y / signal.price_x)
        qty_x = float(self._round_qty(raw_qty_x, signal.coin_x))
        qty_y_rounded = float(self._round_qty(self.qty_y, signal.coin_y))

        val_x = qty_x * signal.price_x
        val_y = qty_y_rounded * signal.price_y

        #pump up to minimum value
        final_qty_y = qty_y_rounded
        if val_x < 5.5 or val_y < 5.5:
            raw_val_x = raw_qty_x * signal.price_x
            raw_val_y = self.qty_y * signal.price_y
            scale = max(6.0 / raw_val_x if raw_val_x > 0 else 1, 6.0 / raw_val_y if raw_val_y > 0 else 1)
            qty_x = float(self._round_qty(raw_qty_x * scale, signal.coin_x))
            final_qty_y = float(self._round_qty(self.qty_y * scale, signal.coin_y))
        
        try:
            async with asyncio.TaskGroup() as tg:
                order_y = tg.create_task(self._place_order(y_side,signal.coin_y, final_qty_y,reduceOnly=reduceOnly))
                order_x = tg.create_task(self._place_order(x_side,signal.coin_x, qty_x,reduceOnly=reduceOnly))
        except Exception as e:
            print(f"executor: cannot place order error {e}")

        res_y = order_y.result()
        res_x = order_x.result()

        print("--Execution--")
        print(f"Y {signal.coin_y}: {y_side} {final_qty_y} -> {res_y}")
        print(f"X {signal.coin_x}: {x_side} {qty_x} -> {res_x}")       

        realized_pnl = ""
        if action in (Action.EXIT_LOSS, Action.EXIT_PROFIT, Action.EXIT_REGIME) and res_y is not None and res_x is not None:
            await asyncio.sleep(2)
            try:
                pnl_y = await asyncio.to_thread(self.session.get_closed_pnl, category="linear", symbol=signal.coin_y, limit=1)
                pnl_x = await asyncio.to_thread(self.session.get_closed_pnl, category="linear", symbol=signal.coin_x, limit=1)
                val_y = float(pnl_y["result"]["list"][0]["closedPnl"]) if pnl_y.get("result", {}).get("list") else 0.0
                val_x = float(pnl_x["result"]["list"][0]["closedPnl"]) if pnl_x.get("result", {}).get("list") else 0.0
                realized_pnl = round(val_y + val_x, 4)
            except Exception as e:
                print(f"executor: failed to fetch pnl for logger: {e}")

        #log order
        if self.logger is not None:
            self.logger.log_order(signal, qty_x, final_qty_y, res_y is not None, res_x is not None, realized_pnl)

        #if one leg fail
        if res_y is None or res_x is None:
            if reduceOnly:
                print("executor: WARNING one exit leg failed -- leaving remaining leg for next retry")
            else:
                print("executor: WARNING one entry leg failed -- unwinding the filled leg to stay flat")
                # close whichever leg DID fill so we don't carry a naked position
                unwind_failed = False
                if res_y is not None:
                    uy = await self._place_order("Sell" if y_side == "Buy" else "Buy", signal.coin_y, self.qty_y, reduceOnly=True)
                    if uy is None: unwind_failed = True
                if res_x is not None:
                    ux = await self._place_order("Sell" if x_side == "Buy" else "Buy", signal.coin_x, qty_x, reduceOnly=True)
                    if ux is None: unwind_failed = True
                
                if unwind_failed:
                    print("CRITICAL ERROR: Failed to unwind naked leg! Exiting immediately to prevent unhedged risk.")
                    import sys
                    sys.exit(1)
            return None
            
        
        self.state = new_state
        return res_y, res_x
