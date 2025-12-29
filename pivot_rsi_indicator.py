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
    Now supports separate RSI lengths for buy and sell signals
    FIXED: Bar-by-bar execution matching Pine Script exactly
    """
    
    def __init__(self, 
                 # Pivot Settings
                 pivot_period: int = 27,
                 pivot_num: int = 30,
                 
                 # RSI Settings
                 rsi_buy_length: int = 6,
                 rsi_sell_length: int = 14,
                 rsi_buy_level: int = 30,
                 rsi_sell_level: int = 60,
                 buy_rsi_range: int = 0,
                 sell_rsi_range: int = 0,
                 pivot_sell_range: int = 40,
                 skip_candles: int = 3,
                 buy_alert: bool = True,
                 sell_alert: bool = True):
        """
        Initialize Pivot RSI indicator with parameters
        
        Args:
            pivot_period: Pivot point period
            pivot_num: Number of pivot points to check
            rsi_buy_length: RSI calculation period for buy signals
            rsi_sell_length: RSI calculation period for sell signals
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
        self.rsi_buy_length = rsi_buy_length
        self.rsi_sell_length = rsi_sell_length
        self.rsi_buy_level = rsi_buy_level
        self.rsi_sell_level = rsi_sell_level
        self.buy_rsi_range = buy_rsi_range
        self.sell_rsi_range = sell_rsi_range
        self.pivot_sell_range = pivot_sell_range
        self.skip_candles = skip_candles
        self.buy_alert = buy_alert
        self.sell_alert = sell_alert
    
    def calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) manually.
        
        Parameters:
            close_prices (pd.Series): Series of closing prices.
            period (int): Lookback period for RSI calculation.
        
        Returns:
            pd.Series: RSI values.
        """
        # Calculate price differences
        delta = close_prices.diff()

        # Separate gains and losses
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Convert to pandas Series
        gain = pd.Series(gain, index=close_prices.index)
        loss = pd.Series(loss, index=close_prices.index)

        # Optionally use Wilder's smoothing (EMA-style)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi  
    
    def find_pivot_high_realtime(self, high: pd.Series, period: int) -> pd.Series:
        """
        Find pivot highs with REALTIME detection matching Pine Script
        Pivot is detected 'period' bars AFTER the actual pivot point
        """
        pivot_high = pd.Series(np.nan, index=high.index, dtype=float)
        
        # Start from period and go to len(high) (no -period at end)
        # This matches Pine Script which continues detecting pivots through current bar
        for i in range(period, len(high)):
            # Check if we have enough bars to look back
            if i >= period * 2:
                center_idx = i - period  # The actual pivot point is 'period' bars ago
                
                left_max = high.iloc[center_idx - period:center_idx].max()
                right_max = high.iloc[center_idx + 1:center_idx + period + 1].max()
                
                if high.iloc[center_idx] > left_max and high.iloc[center_idx] > right_max:
                    # Assign pivot at CURRENT bar (i), not at center
                    pivot_high.iloc[i] = high.iloc[center_idx]
        
        return pivot_high
    
    def find_pivot_low_realtime(self, low: pd.Series, period: int) -> pd.Series:
        """
        Find pivot lows with REALTIME detection matching Pine Script
        Pivot is detected 'period' bars AFTER the actual pivot point
        """
        pivot_low = pd.Series(np.nan, index=low.index, dtype=float)
        
        # Start from period and go to len(low) (no -period at end)
        for i in range(period, len(low)):
            # Check if we have enough bars to look back
            if i >= period * 2:
                center_idx = i - period  # The actual pivot point is 'period' bars ago
                
                left_min = low.iloc[center_idx - period:center_idx].min()
                right_min = low.iloc[center_idx + 1:center_idx + period + 1].min()
                
                if low.iloc[center_idx] < left_min and low.iloc[center_idx] < right_min:
                    # Assign pivot at CURRENT bar (i), not at center
                    pivot_low.iloc[i] = low.iloc[center_idx]
        
        return pivot_low
    
    def calculate_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate pivot trend using BAR-BY-BAR execution matching Pine Script
        """
        ph = self.find_pivot_high_realtime(df['high'], self.prd)
        pl = self.find_pivot_low_realtime(df['low'], self.prd)
        
        # Initialize trend array
        trend = pd.Series(0, index=df.index, dtype=int)
        
        # Arrays to store pivot levels (matching Pine Script array behavior)
        ph_lev = [np.nan] * self.pnum
        pl_lev = [np.nan] * self.pnum
        
        prev_trend = 0
        
        # Bar-by-bar execution
        for i in range(len(df)):
            # Update pivot high levels when new pivot detected
            if not np.isnan(ph.iloc[i]):
                # array.unshift in Pine Script = insert at beginning, pop from end
                ph_lev.insert(0, ph.iloc[i])
                ph_lev.pop()
            
            # Update pivot low levels when new pivot detected
            if not np.isnan(pl.iloc[i]):
                pl_lev.insert(0, pl.iloc[i])
                pl_lev.pop()
            
            # Calculate low rate (lrate)
            lrate = 0.0
            for j in range(len(pl_lev)):
                pl_val = pl_lev[j]
                if not np.isnan(pl_val) and pl_val != 0:
                    rate = (df['close'].iloc[i] - pl_val) / pl_val
                    lrate += (rate / self.pnum)
            
            # Calculate high rate (hrate)
            # Pine Script starts from index 1 (skips first element)
            hrate = 0.0
            for j in range(1, len(ph_lev)):
                ph_val = ph_lev[j]
                if not np.isnan(ph_val) and ph_val != 0:
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
        
        # Calculate RSI with separate lengths for buy and sell
        result['rsi_buy'] = self.calculate_rsi(result['close'], self.rsi_buy_length)
        result['rsi_sell'] = self.calculate_rsi(result['close'], self.rsi_sell_length)
        
        # Calculate trend
        result['trend'] = self.calculate_trend(result)
        
        # Initialize signal columns
        result['rsi_buy_signal'] = False
        result['rsi_sell_signal'] = False
        result['pivot_buy'] = False
        result['pivot_sell'] = False
        result['buy_signal'] = False
        result['sell_signal'] = False
        result['pivot_sell_signal'] = False
        
        # RSI conditions - using separate RSI calculations
        for i in range(1, len(result)):
            result.loc[result.index[i], 'rsi_buy_signal'] = (
                result['rsi_buy'].iloc[i] <= self.rsi_buy_level and 
                result['rsi_buy'].iloc[i-1] > self.rsi_buy_level
            )
            result.loc[result.index[i], 'rsi_sell_signal'] = (
                result['rsi_sell'].iloc[i] >= self.rsi_sell_level and 
                result['rsi_sell'].iloc[i-1] < self.rsi_sell_level
            )
        
        # Pivot conditions
        result['pivot_buy'] = result['trend'] == 1
        result['pivot_sell'] = result['trend'] == -1
        
        # Process signals with range calculations (bar-by-bar)
        prev_bar_index = 0
        
        for i in range(1, len(result)):
            # Buy range calculation
            buy_result = False
            if self.buy_rsi_range > 0:
                if result['rsi_buy_signal'].iloc[i] and result['trend'].iloc[i] == 1:
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
                buy_result = result['rsi_buy_signal'].iloc[i] and result['pivot_buy'].iloc[i]
            
            # Sell range calculation
            sell_result = False
            if self.sell_rsi_range > 0:
                if result['rsi_sell_signal'].iloc[i] and result['trend'].iloc[i] == -1:
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
                sell_result = result['rsi_sell_signal'].iloc[i] and result['pivot_sell'].iloc[i]
            
            # Pivot sell range calculation
            pivot_range_sell_result = False
            if self.pivot_sell_range > 0:
                if result['rsi_sell_signal'].iloc[i] and result['trend'].iloc[i] == 1:
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
            'rsi_buy': result['rsi_buy'],
            'rsi_sell': result['rsi_sell'],
            'trend': result['trend'],
            'rsi_buy_signal': result['rsi_buy_signal'],
            'rsi_sell_signal': result['rsi_sell_signal'],
            'pivot_buy': result['pivot_buy'],
            'pivot_sell': result['pivot_sell']
        }
    
    def analyze_symbols(self, symbol_data: Dict[str, Dict[str, pd.DataFrame]], 
                        asset_type: str = 'both') -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Analyze multiple symbols across multiple timeframes and return buy/sell signals
        
        Args:
            symbol_data: Dictionary with structure {symbol: {timeframe: DataFrame}}
            asset_type: 'gainers' (use sell RSI), 'losers' (use buy RSI), or 'both'
        
        Returns:
            Dictionary with structure {timeframe: {symbol: {signal: str}}}
        """
        TIMEFRAMES = ["1h", "1d"]
        results = {}

        for timeframe in TIMEFRAMES:
            results[timeframe] = {}
        
            for symbol, timeframe_data in symbol_data.items():
                if timeframe in timeframe_data:
                    try:
                        data = timeframe_data[timeframe]
                        
                        # Generate signals for this symbol
                        signals = self.generate_signals(data)
                        
                        # Determine which RSI to use based on asset type
                        if asset_type == 'gainers':
                            # Top gainers: focus on sell signals with RSI sell
                            rsi_value = signals['rsi_sell'].iloc[-1]
                        elif asset_type == 'losers':
                            # Bottom losers: focus on buy signals with RSI buy
                            rsi_value = signals['rsi_buy'].iloc[-1]
                        else:
                            # Default: use both
                            rsi_value = signals['rsi_buy'].iloc[-1]
                        
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
                            "rsi": rsi_value,  # Single RSI based on asset type
                            "rsi_buy": float(signals['rsi_buy'].iloc[-1]),
                            "rsi_sell": float(signals['rsi_sell'].iloc[-1]),
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
            DataFrame with columns: Symbol, Timeframe, Signal, RSI_Buy, RSI_Sell, Trend
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
                if 'rsi_buy' in signal_data:
                    row['RSI_Buy'] = round(signal_data['rsi_buy'], 2)
                if 'rsi_sell' in signal_data:
                    row['RSI_Sell'] = round(signal_data['rsi_sell'], 2)
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
    print("Pivot Trend and RSI Indicator - FIXED Multi-Symbol Screener")
    print("=" * 60)
    
    # Create indicator instance with separate RSI lengths
    indicator = PivotRSIIndicator(
        pivot_period=4,
        pivot_num=3,
        rsi_buy_length=6,
        rsi_sell_length=14,
        rsi_buy_level=40,
        rsi_sell_level=60,
        buy_rsi_range=0,
        sell_rsi_range=0,
        skip_candles=0
    )
    
    print("Indicator initialized with FIXED bar-by-bar execution:")
    print(f"Pivot period: {indicator.prd}")
    print(f"Pivot num: {indicator.pnum}")
    print(f"RSI buy length: {indicator.rsi_buy_length}")
    print(f"RSI sell length: {indicator.rsi_sell_length}")
    print(f"RSI buy level: {indicator.rsi_buy_level}")
    print(f"RSI sell level: {indicator.rsi_sell_level}")
    
    print("\nFIXES APPLIED:")
    print("✓ Pivot detection now matches Pine Script realtime behavior")
    print("✓ Bar-by-bar execution with proper array state management")
    print("✓ Trend calculation aligned with Pine Script logic")
    print("✓ No more lookahead bias or timing misalignment")
    
    print("\nTo use this indicator with Binance bot:")
    print("1. Import: from pivot_rsi_indicator import PivotRSIIndicator")
    print("2. Initialize: indicator = PivotRSIIndicator(rsi_buy_length=6, rsi_sell_length=14)")
    print("3. Analyze: results = indicator.analyze_symbols(all_data)")
    print("4. Trend values should now match Pine Script exactly!")