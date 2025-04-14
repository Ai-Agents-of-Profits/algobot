import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas_ta as ta
import logging
import os

# --- Configuration ---
DATA_PATH = r'datas/COREUSDT_15m_2024.csv'  # Ensure this file exists locally
INITIAL_CASH = 10000.0
COMMISSION_RATE = 0.001  # 0.1%

BACKTEST_OUTPUT_DIR = "backtest"
RESULTS_FILENAME = os.path.join(BACKTEST_OUTPUT_DIR, "smabbi_tweaked_backtest_results.txt")
PLOT_FILENAME = os.path.join(BACKTEST_OUTPUT_DIR, "smabbi_tweaked_backtest_plot.html")

# Ensure the output directory exists
os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

# --- Indicator Calculation ---
def compute_indicators(df):
    logging.info("Computing indicators...")
    # Bollinger Bands using pandas_ta; we then rename the key columns.
    try:
        df.ta.bbands(length=20, std=2, append=True)
        df.rename(columns={
            'BBM_20_2.0': 'SMA20',
            'BBU_20_2.0': 'Upper_Band',
            'BBL_20_2.0': 'Lower_Band'
        }, inplace=True)
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")

    # ADX
    try:
        adx_df = df.ta.adx(length=14)
        df['ADX'] = adx_df['ADX_14']
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")

    # RSI
    try:
        df['RSI'] = df.ta.rsi(length=14)
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")

    # Ichimoku Cloud (simplified manual calculation)
    try:
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()

        df['Tenkan_Sen'] = (high_9 + low_9) / 2
        df['Kijun_Sen'] = (high_26 + low_26) / 2
        df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
        df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        df['Chikou_Span'] = df['Close'].shift(-26)
    except Exception as e:
        logging.error(f"Error calculating Ichimoku: {e}")

    # ATR for Stop Loss
    try:
        df['ATR'] = df.ta.atr(length=14)
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")

    # Drop extra columns from pandas_ta if they exist (avoid conflicts)
    cols_to_drop = ['BBB_20_2.0', 'BBP_20_2.0', 'DMP_14', 'DMN_14']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    logging.info("Indicator computation finished.")
    return df

# --- Load and Prepare Data ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info(f"Loading data from {DATA_PATH}...")
try:
    data = pd.read_csv(DATA_PATH, parse_dates=True, index_col='Date')
    # Rename columns to match backtesting.py expectations
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    logging.info(f"Data loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    logging.critical(f"Error: Data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    logging.critical(f"Error loading or processing data: {e}", exc_info=True)
    exit()

# Compute technical indicators
data = compute_indicators(data)

# Drop the initial rows that contain NaNs from indicator calculations
initial_nan_rows = data.isnull().any(axis=1).sum()
if initial_nan_rows > 0:
    logging.info(f"Dropping initial {initial_nan_rows} rows with NaNs.")
    data = data.iloc[initial_nan_rows:]

# --- Tweaked Strategy Definition ---
class TweakedSMA20BollingerADXIchimokuStrategy(Strategy):
    # Tweak the parameters:
    adx_threshold = 35           # Remains unchanged
    rsi_long_entry = 47          # Slightly higher than before to relax long entries
    rsi_short_entry = 53         # Slightly lower than before to relax short entries
    rsi_long_exit = 70
    rsi_short_exit = 30
    sl_atr_multiple = 1.8        # Tighter stop loss multiplier for quicker exits

    def init(self):
        # Map pre-computed indicators via self.I
        self.sma = self.I(lambda x: x, self.data.SMA20, name="SMA20")
        self.upper_band = self.I(lambda x: x, self.data.Upper_Band, name="Upper_Band")
        self.lower_band = self.I(lambda x: x, self.data.Lower_Band, name="Lower_Band")
        self.adx = self.I(lambda x: x, self.data.ADX, name="ADX")
        self.rsi = self.I(lambda x: x, self.data.RSI, name="RSI")
        self.kijun_sen = self.I(lambda x: x, self.data.Kijun_Sen, name="Kijun_Sen")
        self.chikou_span = self.I(lambda x: x, self.data.Chikou_Span, name="Chikou_Span")
        self.senkou_a = self.I(lambda x: x, self.data.Senkou_Span_A, name="Senkou_A")
        self.senkou_b = self.I(lambda x: x, self.data.Senkou_Span_B, name="Senkou_B")
        self.atr = self.I(lambda x: x, self.data.ATR, name="ATR")

        # State variables for stop management
        self.active_entry_price = None
        self.entry_atr_value = None
        self.active_stop_price = None

        logging.info(f"Strategy Initialized with ADX<{self.adx_threshold}, RSI long entry<{self.rsi_long_entry}, "
                     f"RSI short entry>{self.rsi_short_entry}, SL Multiple={self.sl_atr_multiple}")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # --- Stop Loss Check and Trailing Stop Update ---
        if self.position:
            if self.active_stop_price is not None:
                if self.position.is_long and price <= self.active_stop_price:
                    logging.info(f"Long SL hit: Price {price:.4f} <= Stop {self.active_stop_price:.4f}")
                    self.position.close()
                    self.active_entry_price = None
                    self.entry_atr_value = None
                    self.active_stop_price = None
                    return
                elif self.position.is_short and price >= self.active_stop_price:
                    logging.info(f"Short SL hit: Price {price:.4f} >= Stop {self.active_stop_price:.4f}")
                    self.position.close()
                    self.active_entry_price = None
                    self.entry_atr_value = None
                    self.active_stop_price = None
                    return

            # Update trailing stop for long positions
            if self.position.is_long and price > self.active_entry_price:
                new_stop = price - (self.sl_atr_multiple * atr_val)
                if new_stop > self.active_stop_price:
                    logging.info(f"Trailing long stop moved from {self.active_stop_price:.4f} to {new_stop:.4f}")
                    self.active_stop_price = new_stop
            # Update trailing stop for short positions
            if self.position.is_short and price < self.active_entry_price:
                new_stop = price + (self.sl_atr_multiple * atr_val)
                if new_stop < self.active_stop_price:
                    logging.info(f"Trailing short stop moved from {self.active_stop_price:.4f} to {new_stop:.4f}")
                    self.active_stop_price = new_stop

        # --- Exit Logic ---
        if self.position:
            if self.position.is_long and (crossover(self.sma, self.data.Close) or self.rsi[-1] > self.rsi_long_exit):
                logging.info(f"Exiting long position: Price {price:.4f}, RSI {self.rsi[-1]:.2f}")
                self.position.close()
                self.active_entry_price = None
                self.entry_atr_value = None
                self.active_stop_price = None
                return
            if self.position.is_short and (crossover(self.data.Close, self.sma) or self.rsi[-1] < self.rsi_short_exit):
                logging.info(f"Exiting short position: Price {price:.4f}, RSI {self.rsi[-1]:.2f}")
                self.position.close()
                self.active_entry_price = None
                self.entry_atr_value = None
                self.active_stop_price = None
                return

        # Only consider entries if not in a position
        if self.position:
            return

        # --- Define Ichimoku Cloud Components ---
        cloud_upper = np.nan
        cloud_lower = np.nan
        if not np.isnan(self.senkou_a[-1]) and not np.isnan(self.senkou_b[-1]):
            cloud_upper = max(self.senkou_a[-1], self.senkou_b[-1])
            cloud_lower = min(self.senkou_a[-1], self.senkou_b[-1])

        # --- Entry Logic ---
        if len(self.data.Close) < 2:
            return

        long_signal = (self.data.Close[-2] < self.lower_band[-2] and
                       self.data.Close[-1] > self.lower_band[-1] and
                       self.adx[-1] < self.adx_threshold and
                       self.rsi[-1] < self.rsi_long_entry)

        short_signal = (self.data.Close[-2] > self.upper_band[-2] and
                        self.data.Close[-1] < self.upper_band[-1] and
                        self.adx[-1] < self.adx_threshold and
                        self.rsi[-1] > self.rsi_short_entry)

        # --- Ichimoku Confirmation using a 20-bar look-back ---
        ichimoku_confirms_long = False
        ichimoku_confirms_short = False
        if len(self.data.Close) > 20:
            if not np.isnan(self.kijun_sen[-1]) and not np.isnan(cloud_lower) and not np.isnan(self.chikou_span[-1]):
                ichimoku_confirms_long = (self.kijun_sen[-1] > cloud_lower and
                                          self.chikou_span[-1] > self.data.Close[-20])
            if not np.isnan(self.kijun_sen[-1]) and not np.isnan(cloud_upper) and not np.isnan(self.chikou_span[-1]):
                ichimoku_confirms_short = (self.kijun_sen[-1] < cloud_upper and
                                           self.chikou_span[-1] < self.data.Close[-20])

        # --- Execute Trade ---
        if long_signal and ichimoku_confirms_long:
            logging.info(f"Entering long position at price {price:.4f}")
            self.buy(size=0.95)
            self.active_entry_price = price
            self.entry_atr_value = atr_val
            self.active_stop_price = price - (self.sl_atr_multiple * atr_val)
        elif short_signal and ichimoku_confirms_short:
            logging.info(f"Entering short position at price {price:.4f}")
            self.sell(size=0.95)
            self.active_entry_price = price
            self.entry_atr_value = atr_val
            self.active_stop_price = price + (self.sl_atr_multiple * atr_val)

# --- Run Backtest ---
logging.info("Setting up Backtest...")
bt = Backtest(data, TweakedSMA20BollingerADXIchimokuStrategy, cash=INITIAL_CASH, commission=COMMISSION_RATE)

logging.info("Running Backtest...")
stats = bt.run()
logging.info("Backtest finished.")

# --- Print and Save Results ---
print("\n--- Tweaked Backtest Results ---")
print(stats)

try:
    with open(RESULTS_FILENAME, "w") as f:
        f.write("Backtest Results for TweakedSMA20BollingerADXIchimokuStrategy\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Initial Cash: {INITIAL_CASH}\n")
        f.write(f"Commission: {COMMISSION_RATE}\n\n")
        f.write(stats.to_string())
    logging.info(f"Results saved to {RESULTS_FILENAME}")
except Exception as e:
    logging.error(f"Error saving results to file: {e}")

# --- Plot Results ---
logging.info("Plotting results...")
try:
    bt.plot(filename=PLOT_FILENAME, open_browser=False)
    logging.info(f"Plotting complete. Check the generated HTML file: {PLOT_FILENAME}")
except Exception as e:
    logging.warning(f"Could not generate plot: {e}")
