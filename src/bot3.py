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
        
    symbols = ["NEARUSDT", "AVAXUSDT"]
    # AVAX price is ~$6.566. Adjust qty_y as needed (e.g., qty_y=2.5 is ~$16.4, meeting the $10 minimum)
    bot = PairTrading(key=api_key, secret=api_secret, symbol_list=symbols, qty_y=2.5)
    
    try:
        asyncio.run(bot.main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
