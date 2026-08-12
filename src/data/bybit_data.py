import asyncio
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
            pos_list = self.session.get_positions(category="linear", settleCoin="USDT")["result"]["list"]

            positions = [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "size": p["size"],
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

    def sync_state(self, positions: list, coin_y: str) -> State:
        # Check open positions to recover the bot's state without making a second API call
        if not positions:
            return State.NoPosition

        for p in positions:
            if p["symbol"] == coin_y and float(p["size"]) > 0:
                if p["side"] == "Sell":
                    return State.short_y
                elif p["side"] == "Buy":
                    return State.long_y
                    
        return State.NoPosition
