from data.enums import Action,State
from data.dataclasses import Signal
import asyncio
from decimal import Decimal, ROUND_DOWN

class ExecutionHalted(RuntimeError):
    """Exchange state is uncertain; require manual reconciliation."""


class OrderExecutor:
    def __init__(self,session,qty_y,logger, rules=None, initial_state=State.NoPosition):
        self.session = session
        self.state = initial_state
        self.qty_y = qty_y
        self.logger = logger
        self.rules = rules or {}
        self.halted = False
        
    def _round_qty(self, qty, symbol):
        step, min_qty = self.rules.get(symbol, ("0.001", "0.001"))
        step_d = Decimal(str(step))
        min_qty_d = Decimal(str(min_qty))
        
        q = (Decimal(str(qty)) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
        
        if q < min_qty_d:
            q = min_qty_d
            
        return q
    
    async def _place_order(self,side,symbol,qty,reduceOnly=False):
        order = await asyncio.to_thread(
            self.session.place_order, category="linear", symbol=symbol,
            side=side, qty=str(qty), timeInForce="IOC", orderType="Market",
            reduceOnly=reduceOnly, positionIdx=0,
        )
        if order.get("retCode") != 0 or not order.get("result", {}).get("orderId"):
            raise ExecutionHalted(f"Order submission was not confirmed for {symbol}")
        order_id = order["result"]["orderId"]
        # An acknowledgement is not a fill. Query this exact order until terminal.
        for attempt in range(10):
            response = await asyncio.to_thread(
                self.session.get_open_orders, category="linear", symbol=symbol,
                orderId=order_id,
            )
            if response.get("retCode") != 0:
                raise ExecutionHalted(f"Cannot confirm order {order_id} for {symbol}")
            for detail in response["result"]["list"]:
                if detail["orderId"] != order_id:
                    continue
                status = detail["orderStatus"]
                if status in ("Filled", "Cancelled", "Rejected", "Deactivated", "PartiallyFilledCanceled"):
                    filled = Decimal(detail["cumExecQty"])
                    if not filled.is_finite() or not 0 <= filled <= Decimal(str(qty)):
                        raise ExecutionHalted(f"Invalid fill quantity for {symbol}")
                    return symbol, detail
            if attempt < 9:
                await asyncio.sleep(0.5)
        raise ExecutionHalted(f"Order {order_id} for {symbol} is still unconfirmed")

    async def _positions(self, symbols):
        positions = {}
        for symbol in symbols:
            response = await asyncio.to_thread(self.session.get_positions, category="linear", symbol=symbol)
            if response.get("retCode") != 0:
                raise ExecutionHalted(f"Cannot read position for {symbol}")
            rows = response["result"]["list"]
            if not rows or response["result"].get("nextPageCursor"):
                raise ExecutionHalted(f"Incomplete position response for {symbol}")
            for p in rows:
                size = Decimal(p["size"])
                if p["symbol"] != symbol or p["positionIdx"] != 0 or not size.is_finite() or size < 0:
                    raise ExecutionHalted(f"Invalid position or unsupported hedge mode for {symbol}")
                if size > 0:
                    if symbol in positions or p["side"] not in ("Buy", "Sell"):
                        raise ExecutionHalted(f"Ambiguous position for {symbol}")
                    positions[symbol] = (p["side"], size)
        return positions

    async def _orders(self, requests):
        # Wait for both workers even if one fails: to_thread cannot cancel a submitted order.
        results = await asyncio.gather(*(self._place_order(*r) for r in requests), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise ExecutionHalted("An order outcome is unknown; automatic execution stopped") from result
        return results

    async def _close_positions(self, symbols):
        positions = await self._positions(symbols)
        requests = [("Sell" if side == "Buy" else "Buy", symbol, qty, True)
                    for symbol, (side, qty) in positions.items()]
        results = await self._orders(requests)
        if any(result[1]["orderStatus"] != "Filled" or Decimal(result[1]["cumExecQty"]) != request[2]
               for result, request in zip(results, requests)) or await self._positions(symbols):
            raise ExecutionHalted("Could not confirm that both legs are closed")
        return results, positions

    async def execute_signal(self, signal: Signal):
        if self.halted:
            raise ExecutionHalted("Executor is halted; reconcile positions and orders manually")
        try:
            return await self._execute_signal(signal)
        except Exception as e:
            self.halted = True
            if isinstance(e, ExecutionHalted):
                raise
            raise ExecutionHalted("Execution failed; reconcile positions and orders manually") from e

    async def _execute_signal(self,signal: Signal):
        action = signal.action
        if action in (Action.HOLD, Action.INVALID):
            return None
        symbols = (signal.coin_x, signal.coin_y)
        if action in (Action.EXIT_LOSS, Action.EXIT_PROFIT, Action.EXIT_REGIME):
            if self.state == State.NoPosition:
                return None
            results, positions = await self._close_positions(symbols)
            self.state = State.NoPosition
            if self.logger is not None:
                self.logger.log_order(signal, float(positions.get(signal.coin_x, ("", 0))[1]),
                                      float(positions.get(signal.coin_y, ("", 0))[1]), True, True)
            return results
        if action not in (Action.SY_LX, Action.SX_LY) or self.state != State.NoPosition:
            return None
        if await self._positions(symbols):
            raise ExecutionHalted("Entry blocked: pair already has an exchange position")
        for symbol in symbols:
            response = await asyncio.to_thread(self.session.get_open_orders, category="linear", symbol=symbol, openOnly=0)
            if response.get("retCode") != 0 or response["result"]["list"] or response["result"].get("nextPageCursor"):
                raise ExecutionHalted("Entry blocked: open orders exist or could not be checked")
        if any(not Decimal(str(v)).is_finite() or v <= 0
               for v in (signal.beta, signal.price_x, signal.price_y, self.qty_y)):
            raise ExecutionHalted("Entry requires positive finite beta, prices and quantity")
        y_side = "Sell" if action == Action.SY_LX else "Buy"
        x_side = "Sell" if y_side == "Buy" else "Buy"
        new_state = State.short_y if y_side == "Sell" else State.long_y

        raw_qty_x = abs(signal.beta) * self.qty_y * (signal.price_y / signal.price_x)
        qty_x = float(self._round_qty(raw_qty_x, signal.coin_x))
        qty_y_rounded = float(self._round_qty(self.qty_y, signal.coin_y))

        val_x = qty_x * signal.price_x
        val_y = qty_y_rounded * signal.price_y

        final_qty_y = qty_y_rounded
        if val_x < 5.5 or val_y < 5.5:
            raw_val_x = raw_qty_x * signal.price_x
            raw_val_y = self.qty_y * signal.price_y
            scale = max(6.0 / raw_val_x if raw_val_x > 0 else 1, 6.0 / raw_val_y if raw_val_y > 0 else 1)
            qty_x = float(self._round_qty(raw_qty_x * scale, signal.coin_x))
            final_qty_y = float(self._round_qty(self.qty_y * scale, signal.coin_y))
        
        res_y, res_x = await self._orders([
            (y_side, signal.coin_y, final_qty_y, False),
            (x_side, signal.coin_x, qty_x, False),
        ])
        expected = {}
        fully_filled = True
        for result, side, qty in ((res_y, y_side, final_qty_y), (res_x, x_side, qty_x)):
            symbol, detail = result
            filled = Decimal(detail["cumExecQty"])
            if filled > 0:
                expected[symbol] = (side, filled)
            fully_filled &= detail["orderStatus"] == "Filled" and filled == Decimal(str(qty))
        if await self._positions(symbols) != expected:
            raise ExecutionHalted("Confirmed fills do not match the exchange positions")
        if not fully_filled:
            # Both orders are terminal, so close any actual fills, including partial fills.
            await self._close_positions(symbols)
            if self.logger is not None:
                self.logger.log_order(signal, qty_x, final_qty_y, False, False)
            return None
        self.state = new_state
        if self.logger is not None:
            self.logger.log_order(signal, qty_x, final_qty_y, True, True)
        return res_y, res_x
