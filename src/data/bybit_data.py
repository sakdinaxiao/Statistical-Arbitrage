import asyncio
import math
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError, InvalidRequestError
from data.enums import State

class BybitService:
    def __init__(self,key,secret,testnet=False,demo=False):
        self.testnet = testnet
        self.session = HTTP(
            testnet=testnet,
            api_key = key,
            api_secret = secret,
            demo=demo
        )
        
    async def get_current_price(self,symbols=[]):
        try:
            api_loads = {"prices":{},"time":0}
            response = self.session.get_tickers(category="linear")

            if response["retCode"] != 0: 
                print(f"api: {response.get('retMsg')}")
                return None
            if "result" not in response:
                print(f"api: response error")
                return None
             
            time = response["time"]
            api_loads["time"] = time

            for ticker in response.get("result", {}).get("list", []):
                symbol = ticker.get("symbol")

                if symbol in symbols:
                    try:
                        bid = float(ticker.get("bid1Price", 0))
                        ask = float(ticker.get("ask1Price", 0))

                        if bid > 0 and ask > 0:
                            api_loads["prices"][symbol] = (bid + ask) / 2
                    except (ValueError,TypeError):
                        print(f"api: Cant calcualte price of symbol {symbol}")
                        return None
                    
            missing = [s for s in symbols if s not in api_loads["prices"]]
            if missing:
                print(f"api: requested symbols not found in tickers response: {missing}")
                return None
            
            return api_loads
            
        except FailedRequestError as e:
            print(f"api: request failed (network/timeout): {e}")
            return None
        except InvalidRequestError as e:
            print(f"api: bybit reject request: {e}")
            return None
        except Exception as e:
            print(f"api: unexpected error: {e}")
            return None
    
    async def get_past_price(self,symbol,time_frame,days,limit=1000): #do it once per a coin
        try:
            now = datetime.now()
            endtime = int(now.timestamp() * 1000)
            starttime = int((now - timedelta(days=days)).timestamp() * 1000)
            all_candles = []

            while endtime > starttime:
                response = await asyncio.to_thread(
                    self.session.get_kline,
                    category="linear",
                    symbol=symbol,
                    interval=str(time_frame),
                    limit=limit,
                    start=starttime,
                    end=endtime
                )

                if not isinstance(response, dict) or not isinstance(response.get("result"), dict):
                    print(f"api: malformed response for {symbol}: {response}")
                    return None

                candles = response.get("result", {}).get("list")

                if not candles: break

                all_candles.extend(candles)
                
                old_candle = int(candles[-1][0])
                endtime = old_candle -1

                await asyncio.sleep(0.1)

            candle_map = {int(candle[0]): float(candle[4]) for candle in all_candles}
            number = len(candle_map)

            print(f"fetched total of {number} {symbol}'s candles")

            return candle_map
        
        except FailedRequestError as e:
            print(f"api: request failed (network/timeout): {e}")
            return None
        except InvalidRequestError as e:
            print(f"api: bybit reject request: {e}")
            return None
        except Exception as e:
            print(f"api: unexpected error: {e}")
            return None
    
    def get_account_identity(self):
    # static account info for the dashboard header; fetched once at startup.
        try:
            result = self.session.get_api_key_information()["result"]
            return {"uid": result.get("userID", ""), "note": result.get("note", "")}
        except FailedRequestError as e:
            print(f"api: request failed (network/timeout): {e}")
            return None
        except InvalidRequestError as e:
            print(f"api: bybit reject request: {e}")
            return None

    def get_account_status(self):
        # live wallet + open positions for the dashboard; refreshed each candle.
        try:
            wallet = self.session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
            pos_list = []
            cursor = ""
            while True:
                response = self.session.get_positions(category="linear", settleCoin="USDT", cursor=cursor)
                if response.get("retCode") != 0:
                    raise RuntimeError("Position lookup failed")
                pos_list.extend(response["result"]["list"])
                next_cursor = response["result"].get("nextPageCursor", "")
                if not next_cursor:
                    break
                if next_cursor == cursor:
                    raise RuntimeError("Position pagination did not advance")
                cursor = next_cursor

            positions = [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "size": p["size"],
                    "positionIdx": p["positionIdx"],
                    "avgPrice": p["avgPrice"],
                    "markPrice": p["markPrice"],
                }
                for p in pos_list
            ]

            return {
                "equity": wallet.get("totalEquity", ""),
                "available": wallet.get("totalAvailableBalance", ""),
                "positions": positions,
            }
        except FailedRequestError as e:
            print(f"api: request failed (network/timeout): {e}")
            return None
        except InvalidRequestError as e:
            print(f"api: bybit reject request: {e}")
            return None


    async def get_instruments_info(self, symbols: list):
        # Fetches lot size and min order qty for requested symbols
        try:
            rules = {}
            for sym in symbols:
                response = self.session.get_instruments_info(category="linear", symbol=sym)
                if response.get("retCode") == 0 and response.get("result", {}).get("list"):
                    item = response["result"]["list"][0]
                    lot_filter = item.get("lotSizeFilter", {})
                    qty_step = lot_filter.get("qtyStep", "0.001")
                    min_qty = lot_filter.get("minOrderQty", "0.001")
                    rules[sym] = (str(qty_step), str(min_qty))
            return rules
        except Exception as e:
            print(f"api: failed to fetch instrument info: {e}")
            return None

    def sync_state(self, positions: list, coin_x: str, coin_y: str) -> State:
        active = {}
        for p in positions:
            if p["symbol"] not in (coin_x, coin_y):
                continue
            size = float(p["size"])
            if not math.isfinite(size) or size < 0 or p["positionIdx"] != 0:
                raise RuntimeError("Invalid position or unsupported hedge mode")
            if size > 0:
                if p["symbol"] in active or p["side"] not in ("Buy", "Sell"):
                    raise RuntimeError("Ambiguous pair positions")
                active[p["symbol"]] = p["side"]
        if not active:
            return State.NoPosition
        if len(active) != 2 or active[coin_x] == active[coin_y]:
            raise RuntimeError("Unbalanced pair found; reconcile both legs manually")
        return State.short_y if active[coin_y] == "Sell" else State.long_y
