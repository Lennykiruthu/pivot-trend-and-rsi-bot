# Import necessary packages
import os
import time
from datetime import datetime, timezone
import pandas as pd
from binance.client import Client
import asyncio
from pivot_rsi_indicator import PivotRSIIndicator
from telegram import Bot
from telegram import Update, ForceReply 
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load the .env from parent directory (relative to this file)
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)

# ------------------------- CONFIG ------------------------- #
API_KEY    = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
BOT_TOKEN  = os.environ.get("BOT_TOKEN")
CHAT_ID    = os.environ.get("CHAT_ID")

client = Client(API_KEY, API_SECRET)

# Initialize Pivot RSI Indicator
indicator = PivotRSIIndicator(
    pivot_period=27,
    pivot_num=30,
    rsi_length=14,
    rsi_buy_level=10,
    rsi_sell_level=80,
    buy_rsi_range=0,
    sell_rsi_range=0,
    pivot_sell_range=35,
    skip_candles=5,
    buy_alert=False,
    sell_alert=True
)

# Configure timeframes with their respective asset counts
TIMEFRAME_CONFIG = {
    "1h": {"top": 20, "bottom": 25}
    # "1d": {"top": 50, "bottom": 50}
}

# Fetch all tradable perpetual symbols
symbols = [
    s['symbol']
    for s in client.futures_exchange_info()['symbols']
    if s['contractType'] == 'PERPETUAL' and s['status'] == 'TRADING'
]

# ------------------------- TELEGRAM COMMAND ------------------------- #
async def send_top_gainers_losers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Responds to /assets command with top gainers and bottom losers
    """
    if "all_tickers_by_timeframe" not in context.bot_data:
        await update.message.reply_text("Data not available yet. Try again in a minute.")
        return

    text = ""
    for timeframe, tickers in context.bot_data["all_tickers_by_timeframe"].items():
        text += f"\n{'='*30}\n{timeframe.upper()} Timeframe\n{'='*30}\n"
        
        if tickers['top']:
            text += f"📈 Top {len(tickers['top'])} Gainers:\n"
            for t in tickers['top']:
                text += f"{t['symbol']}: {t['priceChangePercent']}%\n"

        if tickers['bottom']:
            text += f"\n📉 Bottom {len(tickers['bottom'])} Losers:\n"
            for t in tickers['bottom']:
                text += f"{t['symbol']}: {t['priceChangePercent']}%\n"

    await update.message.reply_text(text)

# ------------------------- CLEANUP FUNCTION ------------------------- #
def cleanup_old_alerts(bot_data, max_symbols=200):
    """Keep only the most recent alerts to prevent memory buildup"""
    last_alerts = bot_data.get("last_alerts", {})
    if len(last_alerts) > max_symbols:
        # Keep only the most recent entries (this is a simple approach)
        items = list(last_alerts.items())[-max_symbols:]
        bot_data["last_alerts"] = dict(items)
        print(f"Cleaned up alert history, keeping {len(bot_data['last_alerts'])} entries")    

# ------------------------- BACKGROUND TASK ------------------------- #
async def fetch_and_alerts(app, sleep_seconds=60):
    """
    Continuously fetches Binance data, analyzes signals, and sends alerts
    """
    cycle_count = 0  # Track cycles for periodic cleanup
    
    while True:
        try:
            cycle_count += 1
            
            # Periodic cleanup every 100 cycles to prevent memory buildup
            if cycle_count % 100 == 0:
                cleanup_old_alerts(app.bot_data)
            
            # 1️⃣ Fetch and sort all tickers once
            tickers = client.futures_ticker()
            ticker_dict = {t['symbol']: t for t in tickers if t['symbol'] in symbols}
            sorted_tickers = sorted(
                ticker_dict.values(),
                key=lambda x: float(x['priceChangePercent']),
                reverse=True
            )

            # Store tickers for each timeframe based on config
            app.bot_data["all_tickers_by_timeframe"] = {}
            for timeframe, config in TIMEFRAME_CONFIG.items():
                app.bot_data["all_tickers_by_timeframe"][timeframe] = {
                    'top': sorted_tickers[:config['top']],
                    'bottom': sorted_tickers[-config['bottom']:]
                }

            # 2️⃣ Fetch OHLCV data for each timeframe
            all_data = {}
            print(f"\nCollecting data for timeframes: {list(TIMEFRAME_CONFIG.keys())}")
            start_fetching = time.perf_counter()

            limit = 500    
            below_limit_count = 0

            # Process each timeframe with its specific assets
            for timeframe, config in TIMEFRAME_CONFIG.items():
                timeframe_assets = (app.bot_data["all_tickers_by_timeframe"][timeframe]['top'] + 
                                   app.bot_data["all_tickers_by_timeframe"][timeframe]['bottom'])
                
                print(f"  {timeframe.upper()}: Fetching {len(timeframe_assets)} assets " +
                      f"(Top {config['top']} + Bottom {config['bottom']})")

                for t in timeframe_assets:
                    sym = t['symbol']
                    
                    # Initialize symbol dict if not exists
                    if sym not in all_data:
                        all_data[sym] = {}

                    # Fetch data for this timeframe
                    try:
                        klines = client.futures_klines(symbol=sym, interval=timeframe, limit=limit)
                        df = pd.DataFrame(klines, columns=[
                            "open_time","open","high","low","close","volume","close_time",
                            "quote_volume","trades","taker_base_vol","taker_quote_vol","ignore"
                        ])
                        for col in ["open","high","low","close","volume"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                        all_data[sym][timeframe] = df

                        if df.shape[0] != limit:
                            below_limit_count += 1

                    except Exception as e:
                        elapsed_fetching = time.perf_counter() - start_fetching
                        print(f"⚠️ Skipping {sym} ({timeframe.upper()}) after {elapsed_fetching:.2f} seconds: {e}")

            elapsed_fetching = time.perf_counter() - start_fetching
            print(f"Finished fetching in {elapsed_fetching/60:.2f} minutes")            
            print(f"{below_limit_count} assets have less than {limit} rows")            

            # 3️⃣ Analyze signals with Pivot RSI Indicator
            print("\nParsing all symbols data through Pivot RSI analysis...")
            start_analysis = time.perf_counter()
            results = indicator.analyze_symbols(all_data)

            # 4️⃣ Send alerts
            if results:
                elapsed_analysis = time.perf_counter() - start_analysis                
                print(f"Pivot RSI analysis complete in {elapsed_analysis/60:.2f} minutes")
                
                for timeframe in TIMEFRAME_CONFIG.keys():
                    if timeframe in results:
                        print(f"\nProcessing results for timeframe: {timeframe}")   
                        
                        start_processing_results = time.perf_counter()                  
                        
                        timeframe_signals = []
                        for symbol, signal_data in results[timeframe].items():
                            # Get the current signal
                            current_signal = signal_data['signal'].lower()

                            # Include signal type in alert key to distinguish buy/sell
                            alert_key = f"{symbol}_{timeframe}_{current_signal}"                            
                            
                            # Get last alerts dict
                            last_alerts = app.bot_data.get("last_alerts", {})
                            
                            # Check if signal changed
                            if (alert_key not in last_alerts or 
                                last_alerts[alert_key] != current_signal):
                                
                                # Update tracking for all signals
                                last_alerts[alert_key] = current_signal
                                app.bot_data["last_alerts"] = last_alerts
                                
                                # Only send alert for non-neutral signals
                                if current_signal != "neutral":
                                    utc_now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                                    
                                    # Get additional data for the alert
                                    rsi_value = signal_data.get('rsi', 'N/A')
                                    trend_value = signal_data.get('trend', 'N/A')
                                    
                                    # Format RSI value
                                    rsi_str = f"{rsi_value:.2f}" if isinstance(rsi_value, (int, float)) else str(rsi_value)
                                    
                                    # Create alert message
                                    alert_msg = (f"{signal_data['signal']} on {symbol} at {utc_now} "
                                               f"(RSI: {rsi_str}, Trend: {trend_value})\n")
                                    
                                    timeframe_signals.append(alert_msg)
                                    print(f"New signal: {current_signal} for {alert_key}")                              
                            else:
                                # This is the correct place for duplicate filtering message
                                if current_signal != "neutral":                                
                                    print(f"Duplicate signal filtered: {current_signal} for {alert_key}")

                        if timeframe_signals:
                            elapsed_processing_results = time.perf_counter() - start_processing_results
                            print(f"{timeframe.upper()} results processed in {elapsed_processing_results/60:.2f} minutes")

                            message = f"🔔 {timeframe.upper()} Pivot RSI Signals:\n" + "".join(timeframe_signals)
                            print(message)

                            await app.bot.send_message(chat_id=CHAT_ID, text=message)
                            await asyncio.sleep(1)
                        else:
                            print(f"No new signals for {timeframe.upper()}")

        except Exception as e:
            error_msg = f"Error during fetch: {e}"
            await app.bot.send_message(chat_id=CHAT_ID, text=error_msg)
            print(f"Error in fetch_and_alerts: {e}")

        # ✅ Small adaptive sleep to prevent API rate limits
        await asyncio.sleep(sleep_seconds)

# ------------------------- POST INIT CALLBACK ------------------------- #
async def post_init_callback(application: Application):
    """Initialize background tasks after the bot starts"""
    # Initialize alert tracking dictionary in bot_data
    application.bot_data["last_alerts"] = {}
    print("Alert tracking initialized")

    # Create background task
    task = asyncio.create_task(fetch_and_alerts(application))
    # Store task reference in bot_data to prevent garbage collection
    application.bot_data["background_task"] = task
    print("Background task started successfully")

# ------------------------- MAIN ------------------------- #
def main():
    try:
        # Create application
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Add command handler
        app.add_handler(CommandHandler("assets", send_top_gainers_losers))

        # Set post_init callback (ensures background task starts after bot is ready)
        app.post_init = post_init_callback

        print("Starting Pivot RSI Bot...")
        print(f"Indicator Settings:")
        print(f"  - Pivot Period: {indicator.prd}")
        print(f"  - Pivot Num: {indicator.pnum}")
        print(f"  - RSI Length: {indicator.rsi_length}")
        print(f"  - RSI Buy Level: {indicator.rsi_buy_level}")
        print(f"  - RSI Sell Level: {indicator.rsi_sell_level}")
        print(f"Timeframes: {list(TIMEFRAME_CONFIG.keys())}")
        
        # Start polling
        app.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        print(f"Failed to start bot: {e}")

# ------------------------- RUN ------------------------- #
if __name__ == "__main__":
    main()