import asyncio
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.bybit_data import BybitService
from data.dataclasses import Signal
from data.enums import Action, State
from execution import executor
from main import PairTrading
from models.expected_value import ExpectedValueCalculator


def position(symbol, side, size):
    return {"symbol": symbol, "side": side, "size": str(size), "positionIdx": 0}


class ExpectedValueTests(unittest.TestCase):
    def test_invalid_estimate_clears_previous_half_life(self):
        for spread in ([1, 2, 4, 8, 16], [], [1], [1, 1, 1], [1, float("nan"), 2]):
            with self.subTest(spread=spread):
                calc = ExpectedValueCalculator(1, .001)
                self.assertIsNotNone(calc.cal_half_life([8, 4, 2, 1, .5]))
                self.assertIsNone(calc.cal_half_life(spread))
                self.assertIsNone(calc.halflife)
                self.assertFalse(calc.assess(2, 1, 100, spread)[0])


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_flat_account_can_start(self):
        with patch("main.BybitService") as service:
            service.return_value.get_instruments_info = AsyncMock(return_value={"X": ("1", "1"), "Y": ("1", "1")})
            service.return_value.get_account_status.return_value = {"positions": []}
            service.return_value.sync_state.return_value = State.NoPosition
            bot = PairTrading("unused", "unused", ["X", "Y"])
            with contextlib.redirect_stdout(io.StringIO()):
                await bot.initialize_account()
            self.assertEqual(bot.recovered_state, State.NoPosition)

    def test_recovery_fetches_all_position_pages(self):
        service = BybitService.__new__(BybitService)
        service.session = Mock()
        service.session.get_wallet_balance.return_value = {"result": {"list": [{}]}}
        row = dict(position("X", "Buy", 1), avgPrice="1", markPrice="1")
        service.session.get_positions.side_effect = [
            {"retCode": 0, "result": {"list": [], "nextPageCursor": "next"}},
            {"retCode": 0, "result": {"list": [row], "nextPageCursor": ""}},
        ]
        status = service.get_account_status()
        self.assertEqual(status["positions"], [row])
        self.assertEqual(service.session.get_positions.call_args.kwargs["cursor"], "next")

    async def test_missing_instrument_rules_block_startup(self):
        with patch("main.BybitService") as service:
            service.return_value.get_instruments_info = AsyncMock(return_value={"X": ("1", "1")})
            with self.assertRaises(RuntimeError):
                await PairTrading("unused", "unused", ["X", "Y"]).initialize_account()
            service.return_value.get_account_status.assert_not_called()

    async def test_failed_or_incomplete_account_snapshot_blocks_startup(self):
        for status in (None, {}, {"positions": None}):
            with self.subTest(status=status), patch("main.BybitService") as service:
                service.return_value.get_instruments_info = AsyncMock(return_value={"X": ("1", "1"), "Y": ("1", "1")})
                service.return_value.get_account_status.return_value = status
                bot = PairTrading("unused", "unused", ["X", "Y"])
                with self.assertRaises(RuntimeError):
                    await bot.initialize_account()

    def test_recovery_checks_both_legs(self):
        service = BybitService.__new__(BybitService)
        for positions in ([position("X", "Buy", 1)], [position("Y", "Sell", 1)],
                          [position("X", "Buy", 1), position("Y", "Buy", 1)]):
            with self.subTest(positions=positions), self.assertRaises(RuntimeError):
                service.sync_state(positions, "X", "Y")
        self.assertEqual(service.sync_state([], "X", "Y"), State.NoPosition)
        self.assertEqual(service.sync_state([position("X", "Buy", 1), position("Y", "Sell", 1)], "X", "Y"), State.short_y)

    async def test_open_pair_requires_manual_recovery(self):
        with patch("main.BybitService") as service:
            service.return_value.get_instruments_info = AsyncMock(return_value={"X": ("1", "1"), "Y": ("1", "1")})
            service.return_value.get_account_status.return_value = {"positions": [position("X", "Buy", 1), position("Y", "Sell", 1)]}
            service.return_value.sync_state.return_value = State.short_y
            with self.assertRaises(RuntimeError):
                await PairTrading("unused", "unused", ["X", "Y"]).initialize_account()

    async def test_invalid_initial_half_life_has_clear_error(self):
        bot = PairTrading("unused", "unused", ["X", "Y"])
        bot.bybit = Mock()
        bot.bybit.get_past_price = AsyncMock(return_value={i: 100 + i for i in range(100)})
        with patch("main.ExpectedValueCalculator") as calc, contextlib.redirect_stdout(io.StringIO()):
            calc.return_value.cal_half_life.return_value = None
            with self.assertRaisesRegex(RuntimeError, "half-life"):
                await bot.initialize_math_obj()


class Exchange:
    """An offline exchange: acknowledgements precede terminal execution reports."""
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.orders = {}
        self.positions = {}
        self.requests = []
        self.queries = 0

    def place_order(self, **request):
        self.requests.append(request)
        order_id = str(len(self.requests))
        self.orders[order_id] = request
        return {"retCode": 0, "result": {"orderId": order_id}}

    def get_open_orders(self, **request):
        if "orderId" not in request:
            return {"retCode": 0, "result": {"list": []}}
        self.queries += 1
        order = self.orders[request["orderId"]]
        status, fraction = self.outcomes.get(order["symbol"], ("Filled", 1)) if not order["reduceOnly"] else ("Filled", 1)
        qty = float(order["qty"]) * fraction
        if status != "New":
            self.positions[order["symbol"]] = position(order["symbol"], order["side"], 0 if order["reduceOnly"] else qty)
        return {"retCode": 0, "result": {"list": [{"orderId": request["orderId"], "orderStatus": status, "cumExecQty": str(qty)}]}}

    def get_positions(self, **request):
        return {"retCode": 0, "result": {"list": [self.positions.get(request["symbol"], position(request["symbol"], "", 0))]}}


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exchange = Exchange()
        self.logger = Mock()
        self.bot = executor.OrderExecutor(self.exchange, 1, self.logger, {"X": (".01", ".01"), "Y": (".01", ".01")})
        self.signal = Signal(Action.SY_LX, 2, 1, "X", "Y", 100, 200)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    async def test_entry_requires_confirmed_fills_and_positions(self):
        await self.bot.execute_signal(self.signal)
        self.assertGreaterEqual(self.exchange.queries, 2)
        self.assertEqual(self.bot.state, State.short_y)
        self.logger.log_order.assert_called_once()

    async def test_partial_entry_unwinds_actual_fills(self):
        self.exchange.outcomes["X"] = ("Cancelled", .5)
        await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.NoPosition)
        closes = {r["symbol"]: r for r in self.exchange.requests if r["reduceOnly"]}
        self.assertEqual(set(closes), {"X", "Y"})
        self.assertEqual(float(closes["X"]["qty"]), 1)
        self.assertTrue(all(float(p["size"]) == 0 for p in self.exchange.positions.values()))

    async def test_unknown_outcome_halts_and_blocks_further_orders(self):
        self.exchange.outcomes["X"] = ("New", 0)
        with patch("execution.executor.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(executor.ExecutionHalted):
                await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.NoPosition)
        count = len(self.exchange.requests)
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(len(self.exchange.requests), count)
        self.logger.log_order.assert_not_called()

    async def test_submission_timeout_halts_even_if_other_leg_fills(self):
        original = self.exchange.place_order
        def place(**request):
            if request["symbol"] == "X":
                raise TimeoutError("response lost")
            return original(**request)
        self.exchange.place_order = place
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertTrue(self.bot.halted)

    async def test_exit_uses_exchange_size_not_current_beta(self):
        await self.bot.execute_signal(self.signal)
        self.signal.action = Action.EXIT_PROFIT
        self.signal.beta = 3
        await self.bot.execute_signal(self.signal)
        closes = {r["symbol"]: r for r in self.exchange.requests if r["reduceOnly"]}
        self.assertEqual(float(closes["X"]["qty"]), 2)
        self.assertEqual(float(closes["Y"]["qty"]), 1)
        self.assertEqual(self.bot.state, State.NoPosition)

    async def test_existing_position_blocks_entry(self):
        self.exchange.positions["X"] = position("X", "Buy", 1)
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(self.exchange.requests, [])

    async def test_pending_order_blocks_entry(self):
        self.exchange.get_open_orders = Mock(return_value={"retCode": 0, "result": {"list": [{"orderId": "previous"}]}})
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(self.exchange.requests, [])

    async def test_delayed_fills_are_polled_before_marking_entry(self):
        original = self.exchange.get_open_orders
        seen = set()
        def query(**request):
            order_id = request.get("orderId")
            if order_id and order_id not in seen:
                seen.add(order_id)
                self.assertEqual(self.bot.state, State.NoPosition)
                return {"retCode": 0, "result": {"list": [{"orderId": order_id, "orderStatus": "New", "cumExecQty": "0"}]}}
            return original(**request)
        self.exchange.get_open_orders = query
        with patch("execution.executor.asyncio.sleep", new_callable=AsyncMock):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.short_y)
        self.assertEqual(len(seen), 2)

    async def test_position_mismatch_halts_without_logging_fill(self):
        original = self.exchange.get_positions
        def query(**request):
            response = original(**request)
            if self.exchange.requests and request["symbol"] == "X":
                response["result"]["list"] = [position("X", "Buy", .1)]
            return response
        self.exchange.get_positions = query
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.NoPosition)
        self.logger.log_order.assert_not_called()

    async def test_partial_fill_with_stale_flat_snapshot_halts(self):
        self.exchange.outcomes["X"] = ("Cancelled", .5)
        self.exchange.get_positions = lambda **r: {"retCode": 0, "result": {"list": [position(r["symbol"], "", 0)]}}
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertTrue(self.bot.halted)
        self.logger.log_order.assert_not_called()

    async def test_failed_close_keeps_position_state_and_halts(self):
        await self.bot.execute_signal(self.signal)
        self.logger.reset_mock()
        self.signal.action = Action.EXIT_LOSS
        original = self.exchange.get_open_orders
        def query(**request):
            order_id = request.get("orderId")
            if order_id and self.exchange.orders[order_id]["reduceOnly"]:
                return {"retCode": 0, "result": {"list": [{"orderId": order_id, "orderStatus": "Rejected", "cumExecQty": "0"}]}}
            return original(**request)
        self.exchange.get_open_orders = query
        with self.assertRaises(executor.ExecutionHalted):
            await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.short_y)
        self.logger.log_order.assert_not_called()

    async def test_both_cancelled_entries_leave_no_position(self):
        self.exchange.outcomes = {symbol: ("Cancelled", 0) for symbol in ("X", "Y")}
        await self.bot.execute_signal(self.signal)
        self.assertEqual(self.bot.state, State.NoPosition)
        self.assertFalse(self.bot.halted)
        self.assertEqual(len(self.exchange.requests), 2)

    async def test_main_does_not_retry_execution_halt(self):
        bot = PairTrading("unused", "unused", ["X", "Y"])
        bot.initialize_all = AsyncMock()
        bot.bybit = Mock()
        bot.bybit.get_current_price = AsyncMock(side_effect=executor.ExecutionHalted("unknown order"))
        with patch("main.LiveDashboard"), self.assertRaises(executor.ExecutionHalted):
            await bot.main()
        bot.bybit.get_current_price.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
