import os
import sys
import asyncio
from dotenv import load_dotenv
from main import PairTrading

if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.getenv("KEY")
    api_secret = os.getenv("SECRET")
    
    if not api_key or not api_secret:
        print("Error: API_KEY and API_SECRET must be set in the .env file.")
        sys.exit(1)
        
    symbols = ["ARBUSDT", "WIFUSDT"]
    # WIF price is ~$0.154. Adjust qty_y as needed (e.g., qty_y=100.0 is ~$15.4, meeting the $10 minimum)
    bot = PairTrading(key=api_key, secret=api_secret, symbol_list=symbols, qty_y=100.0)
    
    try:
        asyncio.run(bot.main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
