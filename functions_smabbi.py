import pandas as pd
import pandas_ta as ta
import logging
import ccxt # Added for exchange interaction functions
import time # Potentially needed if we re-add delays

# --- Indicator Calculations ---

def calculate_bollinger_bands(df, window=20, std_dev=2):
    """Calculates Bollinger Bands."""
    if len(df) < window:
        logging.warning(f"Not enough data ({len(df)}) for Bollinger Bands window {window}")
        return df
    try:
        # Use pandas_ta for convenience
        bbands = df.ta.bbands(length=window, std=std_dev)
        # Rename columns to expected names (lowercase)
        df['sma20'] = bbands[f'BBM_{window}_{float(std_dev)}']
        df['stddev'] = df['close'].rolling(window=window).std() # Recalculate stddev if needed elsewhere
        df['upper_band'] = bbands[f'BBU_{window}_{float(std_dev)}']
        df['lower_band'] = bbands[f'BBL_{window}_{float(std_dev)}']
        return df
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}", exc_info=True)
        return df

def calculate_adx(df, window=14):
    """Calculates Average Directional Index (ADX)."""
    if len(df) < window * 2: # ADX typically needs more data
        logging.warning(f"Not enough data ({len(df)}) for ADX window {window}")
        return df
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=window)
        df[f'adx_{window}'] = adx_df[f'ADX_{window}'] # Use lowercase
        return df
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}", exc_info=True)
        return df

def calculate_rsi(df, window=14):
    """Calculates Relative Strength Index (RSI)."""
    if len(df) < window:
        logging.warning(f"Not enough data ({len(df)}) for RSI window {window}")
        return df
    try:
        df[f'rsi_{window}'] = ta.rsi(df['close'], length=window) # Use lowercase
        return df
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}", exc_info=True)
        return df

def calculate_ichimoku(df, tenkan=9, kijun=26, senkou_b_period=52, senkou_shift=26, chikou_shift=-26):
    """Calculates Ichimoku Cloud components using standard definitions."""
    # Ensure enough data for the longest lookback/shift
    required_data = max(tenkan, kijun, senkou_b_period) + senkou_shift
    if len(df) < required_data:
        logging.warning(f"Not enough data ({len(df)}) for Ichimoku calculation (required: {required_data})")
        return df

    try:
        # Tenkan-sen (Conversion Line): (Highest High + Lowest Low)/2 for the past 9 periods
        tenkan_high = df['high'].rolling(window=tenkan).max()
        tenkan_low = df['low'].rolling(window=tenkan).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line): (Highest High + Lowest Low)/2 for the past 26 periods
        kijun_high = df['high'].rolling(window=kijun).max()
        kijun_low = df['low'].rolling(window=kijun).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward 26 periods
        # Note: This value is plotted 26 periods in the future, relative to the *current* bar
        df['senkou_span_a_shifted'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(senkou_shift)

        # Senkou Span B (Leading Span B): (Highest High + Lowest Low)/2 for the past 52 periods, shifted forward 26 periods
        # Note: This value is plotted 26 periods in the future, relative to the *current* bar
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b_shifted'] = ((senkou_b_high + senkou_b_low) / 2).shift(senkou_shift)

        # Chikou Span (Lagging Span): Close shifted back 26 periods
        df['chikou_span'] = df['close'].shift(chikou_shift)

        # Also calculate ATR needed for Stop Loss
        df = calculate_atr(df) # Use default period 14

        return df
    except Exception as e:
        logging.error(f"Error calculating Ichimoku: {e}", exc_info=True)
        return df

def calculate_atr(df, period=14):
    """Calculates Average True Range (ATR)."""
    if len(df) < period:
        logging.warning(f"Not enough data ({len(df)}) for ATR period {period}")
        return df
    try:
        # Ensure columns are numeric and lowercase
        high = pd.to_numeric(df['high'])
        low = pd.to_numeric(df['low'])
        close = pd.to_numeric(df['close'])

        df[f'atr_{period}'] = ta.atr(high, low, close, length=period) # Use lowercase
        return df
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}", exc_info=True)
        return df

def calculate_ema(df, window=9):
    """Calculates Exponential Moving Average (EMA)."""
    if len(df) < window:
        logging.warning(f"Not enough data ({len(df)}) for EMA window {window}")
        return df
    try:
        df[f'ema_{window}'] = ta.ema(df['close'], length=window) # Use lowercase
        return df
    except Exception as e:
        logging.error(f"Error calculating EMA {window}: {e}", exc_info=True)
        return df

def calculate_all_indicators(df, bb_window=20, bb_stddev=2, adx_window=14, rsi_window=14, ichi_tenkan=9, ichi_kijun=26, ichi_senkou_b=52, atr_period=14, ema_windows=[9, 50]):
    """Calculates all required indicators for a strategy."""
    # Ensure column names are lowercase before calculations
    df.columns = map(str.lower, df.columns)

    # --- Calculate Indicators ---
    df = calculate_bollinger_bands(df, window=bb_window, std_dev=bb_stddev)
    df = calculate_adx(df, window=adx_window) # Keep ADX calc in case needed later
    df = calculate_rsi(df, window=rsi_window)
    df = calculate_ichimoku(df, tenkan=ichi_tenkan, kijun=ichi_kijun, senkou_b_period=ichi_senkou_b) # Also calls ATR inside
    df = calculate_atr(df, period=atr_period) # Recalculate ATR explicitly if needed with different period or outside Ichimoku

    # Calculate required EMAs
    for window in ema_windows:
        df = calculate_ema(df, window=window)

    # --- Rename columns for consistency (Optional but good practice) ---
    # Example renaming Bollinger Bands if needed, pandas_ta usually names them clearly
    # if f'BBM_{bb_window}_{float(bb_stddev)}' in df.columns:
    #     df.rename(columns={f'BBM_{bb_window}_{float(bb_stddev)}': 'middle_band'}, inplace=True)
    # if f'BBU_{bb_window}_{float(bb_stddev)}' in df.columns:
    #     df.rename(columns={f'BBU_{bb_window}_{float(bb_stddev)}': 'upper_band'}, inplace=True)
    # if f'BBL_{bb_window}_{float(bb_stddev)}' in df.columns:
    #     df.rename(columns={f'BBL_{bb_window}_{float(bb_stddev)}': 'lower_band'}, inplace=True)

    # Note: calculate_bollinger_bands already renames to sma20, upper_band, lower_band

    return df

# --- General Exchange Functions (Copied from functions_gpb.py) ---

def fetch_candles(exchange, symbol, timeframe, limit):
    """Fetches candles with basic error handling."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logging.warning(f"CCXT Error fetching {symbol} {timeframe} candles: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching candles: {e}", exc_info=True)
    return pd.DataFrame()

def get_position_bybit(exchange, symbol):
    """Fetches current position status for Bybit Linear Perpetual."""
    try:
        positions = exchange.fetch_positions(symbols=[symbol], params={'category': 'linear'})
        if not positions:
            return None, False, None # position_info, in_position, is_long
        pos = positions[0]
        size = float(pos['info'].get('size', 0))
        side = pos['info'].get('side', 'None').lower()
        if size > 0:
            is_long = (side == 'buy')
            return pos, True, is_long
        else:
            return pos, False, None
    except Exception as e:
        logging.error(f"Error fetching position for {symbol}: {e}")
        return None, False, None

def close_position_market(exchange, symbol, position_info):
    """Closes the current position using a market order (Bybit V5)."""
    if not position_info:
        logging.warning("close_position_market called but no position info.")
        return True
    size = float(position_info['info'].get('size', 0))
    side = position_info['info'].get('side', 'None').lower()
    if size <= 0: return True
    is_long = (side == 'buy')
    close_side = 'sell' if is_long else 'buy'
    logging.info(f"Attempting MARKET close: {close_side} {size} {symbol}")
    try:
        params = {'category': 'linear', 'reduceOnly': True}
        order = exchange.create_market_order(symbol, close_side, size, params=params)
        logging.info(f"Market close order placed: {order.get('id', 'N/A')}")
        return True
    except Exception as e:
        logging.error(f"Error during market close: {e}", exc_info=True)
        return False

def get_market_precisions(exchange, symbol):
    """Fetches amount and price precision for the symbol."""
    try:
        market = exchange.market(symbol)
        return market['precision']['amount'], market['precision']['price']
    except Exception as e:
        logging.error(f"Error fetching market precision for {symbol}: {e}. Using defaults.")
        return None, None 