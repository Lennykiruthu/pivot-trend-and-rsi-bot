import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class PivotRSIIndicator:
    """
    Pivot Trend and RSI Standalone Indicator
    Enhanced for multi-symbol, multi-timeframe analysis
    """
    
    def __init__(self, 
                 # Pivot Settings
                 pivot_period: int = 4,
                 pivot_num: int = 3,
                 
                 # RSI Settings
                 rsi_length: int = 14,
                 rsi_buy_level: int = 40,
                 rsi_sell_level: int = 60,
                 buy_rsi_range: int = 0,
                 sell_rsi_range: int = 0,
                 pivot_sell_range: int = 0,
                 skip_candles: int = 0,
                 buy_alert: bool = True,
                 sell_alert: bool = True):
        """
        Initialize Pivot RSI indicator with parameters
        
        Args:
            pivot_period: Pivot point period
            pivot_num: Number of pivot points to check
            rsi_length: RSI calculation period
            rsi_buy_level: RSI level for buy signals
            rsi_sell_level: RSI level for sell signals
            buy_rsi_range: Lookback period for buy range calculation
            sell_rsi_range: Lookback period for sell range calculation
            pivot_sell_range: Sell after up-pivot range
            skip_candles: Skip bars after alert
            buy_alert: Enable buy alerts
            sell_alert: Enable sell alerts
        """
        # Pivot parameters
        self.prd = pivot_period
        self.pnum = pivot_num
        
        # RSI parameters
        self.rsi_length = rsi_length
        self.rsi_buy_level = rsi_buy_level
        self.rsi_sell_level = rsi_sell_level
        self.buy_rsi_range = buy_rsi_range
        self.sell_rsi_range = sell_rsi_range
        self.pivot_sell_range = pivot_sell_range
        self.skip_candles = skip_candles
        self.buy_alert = buy_alert
        self.sell_alert = sell_alert
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_pivot_high(self, high: pd.Series, period: int) -> pd.Series:
        """Find pivot highs"""
        pivot_high = pd.Series(np.nan, index=high.index, dtype=float)
        
        for i in range(period, len(high) - period):
            left_max = high.iloc[i-period:i].max()
            right_max = high.iloc[i+1:i+period+1].max()
            
            if high.iloc[i] > left_max and high.iloc[i] > right_max:
                pivot_high.iloc[i] = high.iloc[i]
        
        return pivot_high
    
    def find_pivot_low(self, low: pd.Series, period: int) -> pd.Series:
        """Find pivot lows"""
        pivot_low = pd.Series(np.nan, index=low.index, dtype=float)
        
        for i in range(period, len(low) - period):
            left_min = low.iloc[i-period:i].min()
            right_min = low.iloc[i+1:i+period+1].min()
            
            if low.iloc[i] < left_min and low.iloc[i] < right_min:
                pivot_low.iloc[i] = low.iloc[i]
        
        return pivot_low
    
    def calculate_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate pivot trend"""
        ph = self.find_pivot_high(df['high'], self.prd)
        pl = self.find_pivot_low(df['low'], self.prd)
        
        # Initialize arrays for pivot levels
        ph_lev = deque([np.nan] * self.pnum, maxlen=self.pnum)
        pl_lev = deque([np.nan] * self.pnum, maxlen=self.pnum)
        
        trend = pd.Series(0, index=df.index)
        prev_trend = 0
        
        for i in range(len(df)):
            # Update pivot high levels
            if not np.isnan(ph.iloc[i]):
                ph_lev.appendleft(ph.iloc[i])
            
            # Update pivot low levels
            if not np.isnan(pl.iloc[i]):
                pl_lev.appendleft(pl.iloc[i])
            
            # Calculate low rate
            lrate = 0.0
            valid_pl = [x for x in pl_lev if not np.isnan(x)]
            if valid_pl:
                for pl_val in valid_pl:
                    if pl_val != 0:  # Avoid division by zero
                        rate = (df['close'].iloc[i] - pl_val) / pl_val
                        lrate += (rate / self.pnum)
            
            # Calculate high rate
            hrate = 0.0
            valid_ph = [x for x in ph_lev if not np.isnan(x)]
            if valid_ph and len(valid_ph) > 1:
                for ph_val in valid_ph[1:]:
                    if ph_val != 0:  # Avoid division by zero
                        rate = (df['close'].iloc[i] - ph_val) / ph_val
                        hrate += (rate / self.pnum)
            
            # Determine trend
            if hrate > 0 and lrate > 0:
                curr_trend = 1
            elif hrate < 0 and lrate < 0:
                curr_trend = -1
            else:
                curr_trend = prev_trend
            
            trend.iloc[i] = curr_trend
            prev_trend = curr_trend
        
        return trend
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate buy and sell signals for a single asset"""
        # Make a copy to avoid modifying original
        result = df.copy()
        
        # Calculate RSI
        result['rsi'] = self.calculate_rsi(result['close'], self.rsi_length)
        
        # Calculate trend
        result['trend'] = self.calculate_trend(result)
        
        # Initialize signal columns
        result['rsi_buy'] = False
        result['rsi_sell'] = False
        result['pivot_buy'] = False
        result['pivot_sell'] = False
        result['buy_signal'] = False
        result['sell_signal'] = False
        result['pivot_sell_signal'] = False
        
        # RSI conditions
        for i in range(1, len(result)):
            result.loc[result.index[i], 'rsi_buy'] = (
                result['rsi'].iloc[i] <= self.rsi_buy_level and 
                result['rsi'].iloc[i-1] > self.rsi_buy_level
            )
            result.loc[result.index[i], 'rsi_sell'] = (
                result['rsi'].iloc[i] >= self.rsi_sell_level and 
                result['rsi'].iloc[i-1] < self.rsi_sell_level
            )
        
        # Pivot conditions
        result['pivot_buy'] = result['trend'] == 1
        result['pivot_sell'] = result['trend'] == -1
        
        # Process signals with range calculations
        prev_bar_index = 0
        
        for i in range(1, len(result)):
            # Buy range calculation
            buy_result = False
            if self.buy_rsi_range > 0:
                if result['rsi_buy'].iloc[i] and result['trend'].iloc[i] == 1:
                    buy_count = 0
                    for j in range(1, min(self.buy_rsi_range + 1, i + 1)):
                        if result['trend'].iloc[i-j] == 1:
                            buy_count += 1
                        elif (i-j-1 >= 0 and 
                              result['trend'].iloc[i-j-1] == -1 and 
                              result['trend'].iloc[i-j] == 1):
                            break
                    
                    if 0 < buy_count < self.buy_rsi_range:
                        buy_result = True
            else:
                buy_result = result['rsi_buy'].iloc[i] and result['pivot_buy'].iloc[i]
            
            # Sell range calculation
            sell_result = False
            if self.sell_rsi_range > 0:
                if result['rsi_sell'].iloc[i] and result['trend'].iloc[i] == -1:
                    sell_count = 0
                    for j in range(1, min(self.sell_rsi_range + 1, i + 1)):
                        if result['trend'].iloc[i-j] == -1:
                            sell_count += 1
                        elif (i-j-1 >= 0 and 
                              result['trend'].iloc[i-j-1] == 1 and 
                              result['trend'].iloc[i-j] == -1):
                            break
                    
                    if 0 < sell_count < self.sell_rsi_range:
                        sell_result = True
            else:
                sell_result = result['rsi_sell'].iloc[i] and result['pivot_sell'].iloc[i]
            
            # Pivot sell range calculation
            pivot_range_sell_result = False
            if self.pivot_sell_range > 0:
                if result['rsi_sell'].iloc[i] and result['trend'].iloc[i] == 1:
                    for j in range(1, min(self.pivot_sell_range + 1, i + 1)):
                        if result['pivot_sell'].iloc[i-j]:
                            pivot_range_sell_result = True
                            break
            
            # Generate signals with skip candles logic
            if buy_result and (i >= prev_bar_index + self.skip_candles):
                result.loc[result.index[i], 'buy_signal'] = True
                prev_bar_index = i
            
            if sell_result and (i >= prev_bar_index + self.skip_candles):
                result.loc[result.index[i], 'sell_signal'] = True
                prev_bar_index = i
            
            if pivot_range_sell_result and (i >= prev_bar_index + self.skip_candles):
                result.loc[result.index[i], 'pivot_sell_signal'] = True
                prev_bar_index = i
        
        return {
            'buy_signal': result['buy_signal'],
            'sell_signal': result['sell_signal'],
            'pivot_sell_signal': result['pivot_sell_signal'],
            'rsi': result['rsi'],
            'trend': result['trend'],
            'rsi_buy': result['rsi_buy'],
            'rsi_sell': result['rsi_sell'],
            'pivot_buy': result['pivot_buy'],
            'pivot_sell': result['pivot_sell']
        }
    
    def analyze_symbols(self, symbol_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Analyze multiple symbols across multiple timeframes and return buy/sell signals
        
        Args:
            symbol_data: Dictionary with structure {symbol: {timeframe: DataFrame}}
                        where DataFrame has columns: open, high, low, close, volume
        
        Returns:
            Dictionary with structure {timeframe: {symbol: {signal: str}}}
        """
        # Define timeframes to analyze
        TIMEFRAMES = ["1h", "1d"]
        
        results = {}

        # Loop through each timeframe
        for timeframe in TIMEFRAMES:
            results[timeframe] = {}
        
            for symbol, timeframe_data in symbol_data.items():
                if timeframe in timeframe_data:
                    try:
                        data = timeframe_data[timeframe]
                        
                        # Generate signals for this symbol
                        signals = self.generate_signals(data)
                        
                        # Determine final signal based on latest bar
                        latest_buy = signals['buy_signal'].iloc[-1]
                        latest_sell = signals['sell_signal'].iloc[-1]
                        latest_pivot_sell = signals['pivot_sell_signal'].iloc[-1]
                        
                        if latest_buy:
                            signal = 'Buy'
                        elif latest_sell or latest_pivot_sell:
                            signal = 'Sell'
                        else:
                            signal = 'Neutral'
                        
                        results[timeframe][symbol] = {
                            "signal": signal,
                            "rsi": float(signals['rsi'].iloc[-1]),
                            "trend": int(signals['trend'].iloc[-1])
                        }
                        
                    except Exception as e:
                        print(f"Error analyzing {symbol} on {timeframe}: {e}")
                        results[timeframe][symbol] = {
                            "signal": "ERROR",
                            "error": str(e)
                        }
            
        return results
    
    def create_summary_table(self, results: Dict[str, Dict[str, Dict[str, str]]]) -> pd.DataFrame:
        """
        Create a summary table of all symbol signals across timeframes
        
        Args:
            results: Output from analyze_symbols()
            
        Returns:
            DataFrame with columns: Symbol, Timeframe, Signal, RSI, Trend
        """
        table_data = []
        
        for timeframe, symbols in results.items():
            for symbol, signal_data in symbols.items():
                row = {
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'Signal': signal_data.get('signal', 'Unknown')
                }
                
                # Add optional fields if available
                if 'rsi' in signal_data:
                    row['RSI'] = round(signal_data['rsi'], 2)
                if 'trend' in signal_data:
                    row['Trend'] = signal_data['trend']
                
                table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def get_buy_symbols(self, results: Dict[str, Dict[str, Dict[str, str]]], 
                       timeframe: Optional[str] = None) -> List[str]:
        """
        Get list of symbols with buy signals
        
        Args:
            results: Output from analyze_symbols()
            timeframe: Optional specific timeframe to filter by
            
        Returns:
            List of symbol names with buy signals
        """
        buy_symbols = []
        
        if timeframe:
            if timeframe in results:
                buy_symbols = [
                    symbol for symbol, data in results[timeframe].items() 
                    if data.get('signal') == 'Buy'
                ]
        else:
            # Get buy symbols from all timeframes
            for tf, symbols in results.items():
                buy_symbols.extend([
                    f"{symbol}({tf})" for symbol, data in symbols.items() 
                    if data.get('signal') == 'Buy'
                ])
        
        return buy_symbols
    
    def get_sell_symbols(self, results: Dict[str, Dict[str, Dict[str, str]]], 
                        timeframe: Optional[str] = None) -> List[str]:
        """
        Get list of symbols with sell signals
        
        Args:
            results: Output from analyze_symbols()
            timeframe: Optional specific timeframe to filter by
            
        Returns:
            List of symbol names with sell signals
        """
        sell_symbols = []
        
        if timeframe:
            if timeframe in results:
                sell_symbols = [
                    symbol for symbol, data in results[timeframe].items() 
                    if data.get('signal') == 'Sell'
                ]
        else:
            # Get sell symbols from all timeframes
            for tf, symbols in results.items():
                sell_symbols.extend([
                    f"{symbol}({tf})" for symbol, data in symbols.items() 
                    if data.get('signal') == 'Sell'
                ])
        
        return sell_symbols


# Example usage and testing
if __name__ == "__main__":
    print("Pivot Trend and RSI Indicator - Multi-Symbol Screener")
    print("=" * 60)
    
    # Create indicator instance
    indicator = PivotRSIIndicator(
        pivot_period=4,
        pivot_num=3,
        rsi_length=14,
        rsi_buy_level=40,
        rsi_sell_level=60,
        buy_rsi_range=0,
        sell_rsi_range=0,
        skip_candles=0
    )
    
    print("Indicator initialized with default parameters")
    print(f"Pivot period: {indicator.prd}")
    print(f"Pivot num: {indicator.pnum}")
    print(f"RSI length: {indicator.rsi_length}")
    print(f"RSI buy level: {indicator.rsi_buy_level}")
    print(f"RSI sell level: {indicator.rsi_sell_level}")
    
    print("\nTo use this indicator with Binance bot:")
    print("1. Import: from pivot_rsi_indicator import PivotRSIIndicator")
    print("2. Initialize: indicator = PivotRSIIndicator()")
    print("3. Analyze: results = indicator.analyze_symbols(all_data)")
    print("4. The structure matches VuManChu implementation for easy integration")
    
    # Demo with sample data
    print("\n" + "=" * 60)
    print("Creating sample data for demonstration...")
    
    # Create sample OHLC data for 2 symbols and 2 timeframes
    dates_1h = pd.date_range(start='2024-01-01', periods=500, freq='H')
    dates_1d = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    
    sample_data = {}
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        sample_data[symbol] = {}
        
        # 1h data
        close_1h = 100 + np.cumsum(np.random.randn(500) * 2)
        sample_data[symbol]['1h'] = pd.DataFrame({
            'open': close_1h + np.random.randn(500) * 0.5,
            'high': close_1h + abs(np.random.randn(500)) * 1.5,
            'low': close_1h - abs(np.random.randn(500)) * 1.5,
            'close': close_1h,
            'volume': abs(np.random.randn(500) * 1000)
        }, index=dates_1h)
        
        # 1d data
        close_1d = 100 + np.cumsum(np.random.randn(100) * 5)
        sample_data[symbol]['1d'] = pd.DataFrame({
            'open': close_1d + np.random.randn(100) * 1,
            'high': close_1d + abs(np.random.randn(100)) * 3,
            'low': close_1d - abs(np.random.randn(100)) * 3,
            'close': close_1d,
            'volume': abs(np.random.randn(100) * 5000)
        }, index=dates_1d)
    
    # Analyze symbols
    print("Analyzing symbols...")
    results = indicator.analyze_symbols(sample_data)
    
    # Display results
    print("\nResults by timeframe:")
    for timeframe, symbols in results.items():
        print(f"\n{timeframe.upper()} timeframe:")
        for symbol, data in symbols.items():
            print(f"  {symbol}: {data['signal']} (RSI: {data.get('rsi', 'N/A'):.2f}, Trend: {data.get('trend', 'N/A')})")
    
    # Create summary table
    summary = indicator.create_summary_table(results)
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    
    # Get buy/sell symbols
    buy_symbols = indicator.get_buy_symbols(results)
    sell_symbols = indicator.get_sell_symbols(results)
    
    print(f"\nBuy signals: {buy_symbols if buy_symbols else 'None'}")
    print(f"Sell signals: {sell_symbols if sell_symbols else 'None'}")