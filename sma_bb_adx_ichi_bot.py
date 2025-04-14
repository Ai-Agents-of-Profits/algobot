# SMA20 Bollinger ADX Ichimoku Bot
import ccxt
import os
import time
import schedule
import logging
import pandas as pd
from dotenv import load_dotenv
import math # For rounding
import colorama
from colorama import Fore, Back, Style
from datetime import datetime

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# --- Local Modules ---
# Using own state manager and functions now
from state_manager_smabbi import initialize_state, get_state, set_state, reset_state
from functions_smabbi import fetch_candles, get_position_bybit, close_position_market, get_market_precisions, calculate_all_indicators

# --- Setup ---
load_dotenv()
# Configure logging with a custom formatter for better visual display
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Silence verbose ccxt DEBUG logs
ccxt_logger = logging.getLogger('ccxt')
ccxt_logger.setLevel(logging.WARNING)

# --- Configuration ---
SYMBOL = 'CORE/USDT:USDT' # Adjust to your desired symbol
TIMEFRAME = '15m'
ORDER_SIZE_USD = 25
FETCH_LIMIT = 200 # Increased from 100 for Ichimoku
SCHEDULE_INTERVAL_SECONDS = 30 # How often to check
USE_TESTNET = False # SET TO FALSE FOR LIVE

# Strategy Specific Params
ADX_WINDOW = 14
RSI_WINDOW = 14
BB_WINDOW = 20
BB_STDDEV = 2
# Ichimoku defaults (9, 26, 52, 26 shift, -26 shift) are used in functions_smabbi
ADX_THRESHOLD = 35  # Increased from 30 to match backtest
RSI_LONG_ENTRY_THRESHOLD = 47  # Increased from 40 to match backtest
RSI_SHORT_ENTRY_THRESHOLD = 53  # Decreased from 60 to match backtest
RSI_LONG_EXIT_THRESHOLD = 70
RSI_SHORT_EXIT_THRESHOLD = 30

# Stop Loss Config
sl_atr_period = 14
sl_atr_multiple = 1.8  # Decreased from 2.5 to match backtest

# Display fancy header
def print_header():
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'SMA20 BOLLINGER ADX ICHIMOKU STRATEGY BOT':^80}")
    print(f"{Fore.CYAN}{'-' * 80}")
    print(f"{Fore.GREEN}Symbol: {Fore.WHITE}{SYMBOL} | {Fore.GREEN}Timeframe: {Fore.WHITE}{TIMEFRAME} | {Fore.GREEN}Order Size: {Fore.WHITE}${ORDER_SIZE_USD} USD")
    print(f"{Fore.GREEN}ADX Threshold: {Fore.WHITE}{ADX_THRESHOLD} | {Fore.GREEN}RSI Entry (L/S): {Fore.WHITE}{RSI_LONG_ENTRY_THRESHOLD}/{RSI_SHORT_ENTRY_THRESHOLD} | {Fore.GREEN}SL Multiple: {Fore.WHITE}{sl_atr_multiple}×ATR")
    print(f"{Fore.CYAN}{'=' * 80}\n{Style.RESET_ALL}")

print_header()
logging.info(f"Strategy: SMA20+BB({BB_WINDOW},{BB_STDDEV})+ADX({ADX_WINDOW})< {ADX_THRESHOLD}+Ichimoku+RSI({RSI_WINDOW}) Entry<{RSI_LONG_ENTRY_THRESHOLD}/>{RSI_SHORT_ENTRY_THRESHOLD}, Exit<{RSI_SHORT_EXIT_THRESHOLD}/>{RSI_LONG_EXIT_THRESHOLD}")
logging.info(f"SL: {sl_atr_multiple} * ATR({sl_atr_period})")

# --- Exchange Setup --- (Reusing from gpb_bot.py)
logging.info("Connecting to Bybit...")
try:
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
        'options': {'defaultType': 'linear'}
    })
    if not os.getenv('BYBIT_API_KEY') or not os.getenv('BYBIT_API_SECRET'):
        raise ValueError("API Key/Secret missing in .env")

    if USE_TESTNET:
        logging.info("Using Bybit Testnet")
        exchange.set_sandbox_mode(True)
    else:
        logging.info("Using Bybit Mainnet")

    exchange.load_markets()
    logging.info(f"Connected to Bybit ({'Testnet' if USE_TESTNET else 'Mainnet'}).")

    AMOUNT_PRECISION, PRICE_PRECISION = get_market_precisions(exchange, SYMBOL)
    if AMOUNT_PRECISION is None or PRICE_PRECISION is None:
        raise ValueError("Could not fetch market precision.")

except Exception as e:
    logging.critical(f"Exchange setup failed: {e}", exc_info=True)
    exit()

# Initialize State
initialize_state() # Reusing the GPB state manager

# --- Main Bot Logic ---
def bot_logic():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{Fore.CYAN}[{now}] {Style.BRIGHT}Running SMABBI Cycle [{TIMEFRAME}] {Style.RESET_ALL}")
    logging.info(f"--- Running SMABBI Cycle [{TIMEFRAME}] ---")
    state = get_state()
    try:
        # --- Sync with Exchange --- #
        exch_pos_info, exch_in_pos, exch_is_long = get_position_bybit(exchange, SYMBOL)

        # State reconciliation (same as gpb_bot)
        if state.get('active_trade', False) and not exch_in_pos:
            logging.warning("Bot active but no exchange position found. Resetting state.")
            reset_state()
            state = get_state()
        elif not state.get('active_trade', False) and exch_in_pos:
            logging.error("Exchange position found, but bot inactive. Manual intervention needed. Bot exiting cycle.")
            return

        # --- Get Data & Indicators ---
        df = fetch_candles(exchange, SYMBOL, TIMEFRAME, limit=FETCH_LIMIT)
        if df.empty or len(df) < FETCH_LIMIT: # Ensure sufficient data
            logging.warning(f"Insufficient candle data ({len(df)}). Skipping.")
            return

        # Calculate all required indicators
        df = calculate_all_indicators(df)

        # Check if essential indicators were calculated successfully (using lowercase names)
        all_required_cols = ['sma20', 'upper_band', 'lower_band', f'adx_{ADX_WINDOW}', f'rsi_{RSI_WINDOW}',
                         'tenkan_sen', 'kijun_sen', 'senkou_span_a_shifted', 'senkou_span_b_shifted', 'chikou_span', f'atr_{sl_atr_period}']
        if not all(col in df.columns for col in all_required_cols):
            logging.error(f"Indicator calculation failed. Columns missing: {[col for col in all_required_cols if col not in df.columns]}. Skipping.")
            return

        # Check for NaNs in the *latest* row for non-shifted indicators
        non_shifted_cols = ['sma20', 'upper_band', 'lower_band', f'adx_{ADX_WINDOW}', f'rsi_{RSI_WINDOW}',
                          'tenkan_sen', 'kijun_sen', f'atr_{sl_atr_period}'] # Exclude senkou spans AND chikou_span
        latest_indicators = df[[col for col in non_shifted_cols if col in df.columns]].iloc[-1]
        if latest_indicators.isnull().any():
             logging.warning(f"Latest indicator values contain NaNs: {latest_indicators[latest_indicators.isnull()].index.tolist()}. Skipping cycle.")
             return

        # --- Current & Previous Candle Data --- (using .iloc[-1] for latest, .iloc[-2] for previous)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_close = latest['close']
        current_atr = latest[f'atr_{sl_atr_period}']
        if pd.isna(current_atr) or current_atr <= 0:
            logging.warning(f"Invalid ATR value ({current_atr}). Skipping cycle.")
            return

        # --- EXIT LOGIC --- #
        if state.get('active_trade', False):
            close_reason = None
            is_long = state['position_side'] == 'long'
            stop_loss_price = state.get('stop_loss_price')
            entry_price = state.get('entry_price')

            # Update trailing stop if price moved favorably
            if stop_loss_price is not None and entry_price is not None:
                new_stop = None
                if is_long and current_close > entry_price:
                    # For long positions, move stop up when price rises above entry
                    new_stop = current_close - (sl_atr_multiple * current_atr)
                    if new_stop > stop_loss_price:
                        stop_loss_price = new_stop
                        print(f"{Fore.YELLOW}Trailing stop updated: {stop_loss_price:.4f}")
                        logging.info(f"Trailing stop updated: {stop_loss_price:.4f}")
                        # Update state with new stop loss
                        state['stop_loss_price'] = stop_loss_price
                        set_state(state)
                elif not is_long and current_close < entry_price:
                    # For short positions, move stop down when price falls below entry
                    new_stop = current_close + (sl_atr_multiple * current_atr)
                    if new_stop < stop_loss_price:
                        stop_loss_price = new_stop
                        print(f"{Fore.YELLOW}Trailing stop updated: {stop_loss_price:.4f}")
                        logging.info(f"Trailing stop updated: {stop_loss_price:.4f}")
                        # Update state with new stop loss
                        state['stop_loss_price'] = stop_loss_price
                        set_state(state)

            # 1. Check Stop Loss First
            if stop_loss_price is not None:
                if (is_long and current_close <= stop_loss_price) or (not is_long and current_close >= stop_loss_price):
                    close_reason = f"STOP LOSS Hit! Price={current_close:.4f}, SL={stop_loss_price:.4f}"

            # 2. Check Strategy Exit Conditions (if SL not hit)
            # Note: The strategy example uses crossover, which is harder in live trading. 
            #       We'll check if the condition is met on the latest closed bar.
            if close_reason is None:
                sma20 = latest['sma20']
                rsi = latest[f'rsi_{RSI_WINDOW}']

                if is_long:
                    # Exit if price closes below SMA20 OR RSI > threshold
                    if current_close < sma20:
                         close_reason = f"Exit: Close < sma20 ({current_close:.4f} < {sma20:.4f})"
                    elif rsi > RSI_LONG_EXIT_THRESHOLD:
                         close_reason = f"Exit: RSI > {RSI_LONG_EXIT_THRESHOLD} ({rsi:.2f})"
                else: # is_short
                    # Exit if price closes above SMA20 OR RSI < threshold
                    if current_close > sma20:
                        close_reason = f"Exit: Close > sma20 ({current_close:.4f} > {sma20:.4f})"
                    elif rsi < RSI_SHORT_EXIT_THRESHOLD:
                        close_reason = f"Exit: RSI < {RSI_SHORT_EXIT_THRESHOLD} ({rsi:.2f})"

            # Execute Close if reason found
            if close_reason:
                print(f"\n{Fore.RED}{Style.BRIGHT}EXIT SIGNAL: {close_reason}. Closing {state['position_side']} position.{Style.RESET_ALL}")
                logging.info(f"EXIT SIGNAL: {close_reason}. Closing {state['position_side']} position.")
                success = close_position_market(exchange, SYMBOL, exch_pos_info)
                if success: 
                    reset_state()
                    print(f"{Fore.MAGENTA}Position closed successfully. State reset.{Style.RESET_ALL}")
                else: 
                    logging.error("Market close FAILED after exit signal. State not reset.")
                    print(f"{Fore.RED}Market close FAILED after exit signal. State not reset.{Style.RESET_ALL}")
                return
            else:
                # Print position details
                profit_pct = ((current_close / entry_price - 1) * 100) if is_long else ((entry_price / current_close - 1) * 100)
                profit_color = Fore.GREEN if profit_pct > 0 else Fore.RED
                print(f"{Fore.CYAN}Active {Fore.GREEN if is_long else Fore.RED}{state['position_side'].upper()} position: "
                      f"Entry={entry_price:.4f}, Current={current_close:.4f}, "
                      f"SL={stop_loss_price:.4f}, P/L: {profit_color}{profit_pct:.2f}%")
                logging.info("Holding position. No exit signal.")
        # --- ENTRY LOGIC --- #
        elif not state.get('active_trade', False):
            # Get indicator values for latest and previous bar (using lowercase)
            adx = latest[f'adx_{ADX_WINDOW}']
            rsi = latest[f'rsi_{RSI_WINDOW}']
            upper_band = latest['upper_band']; lower_band = latest['lower_band']
            prev_close = prev['close']
            prev_upper_band = prev['upper_band']; prev_lower_band = prev['lower_band']

            # Ichimoku values needed for confirmation
            chikou_span = latest['chikou_span'] # Compare this value to price 26 bars ago
            # For cloud confirmation, we need the cloud values from 26 bars ago to compare with the current price
            # However, the backtest logic used Kijun vs Cloud (which isn't standard)
            # Let's stick closer to standard Ichimoku: Price vs Cloud and Chikou vs Price
            # We need Senkou A/B values calculated 26 bars ago
            senkou_a_past = df['senkou_span_a_shifted'].iloc[-1 - 26] if len(df) > 26 else None
            senkou_b_past = df['senkou_span_b_shifted'].iloc[-1 - 26] if len(df) > 26 else None

            cloud_upper_past = None
            cloud_lower_past = None
            if senkou_a_past is not None and senkou_b_past is not None:
                cloud_upper_past = max(senkou_a_past, senkou_b_past)
                cloud_lower_past = min(senkou_a_past, senkou_b_past)

            enter_long = False
            enter_short = False

            # Primary Entry Conditions (Bollinger + ADX + RSI)
            primary_long_signal = (prev_close < prev_lower_band and
                                   current_close > lower_band and
                                   adx < ADX_THRESHOLD and
                                   rsi < RSI_LONG_ENTRY_THRESHOLD)

            primary_short_signal = (prev_close > prev_upper_band and
                                    current_close < upper_band and
                                    adx < ADX_THRESHOLD and
                                    rsi > RSI_SHORT_ENTRY_THRESHOLD)

            # Ichimoku Confirmation (Standard: Price vs Cloud, Chikou vs Price)
            chikou_compare_price = df['close'].iloc[-1 - 26] if len(df) > 26 else None # Price 26 bars ago

            # --- Logging Conditions --- #
            log_long_conditions = {
                "BB_Cross_Up": prev_close < prev_lower_band and current_close > lower_band,
                "ADX_Low": adx < ADX_THRESHOLD,
                "RSI_OK": rsi < RSI_LONG_ENTRY_THRESHOLD,
                "Ichimoku_OK": False # Default
            }
            log_short_conditions = {
                "BB_Cross_Down": prev_close > prev_upper_band and current_close < upper_band,
                "ADX_Low": adx < ADX_THRESHOLD,
                "RSI_OK": rsi > RSI_SHORT_ENTRY_THRESHOLD,
                "Ichimoku_OK": False # Default
            }
            # --- End Logging Conditions --- #

            ichimoku_confirms_long = False
            if cloud_lower_past is not None and chikou_compare_price is not None:
                price_above_cloud = current_close > cloud_upper_past # Check if current price is above the relevant past cloud
                chikou_above_price = not pd.isna(chikou_span) and chikou_span > chikou_compare_price
                ichimoku_confirms_long = price_above_cloud and chikou_above_price
                log_long_conditions["Ichimoku_OK"] = ichimoku_confirms_long # Update log dict

            ichimoku_confirms_short = False
            if cloud_upper_past is not None and chikou_compare_price is not None:
                price_below_cloud = current_close < cloud_lower_past # Check if current price is below the relevant past cloud
                chikou_below_price = not pd.isna(chikou_span) and chikou_span < chikou_compare_price
                ichimoku_confirms_short = price_below_cloud and chikou_below_price
                log_short_conditions["Ichimoku_OK"] = ichimoku_confirms_short # Update log dict

            # Log the conditions before the final check
            condition_color = lambda x: Fore.GREEN if x else Fore.RED
            print(f"\n{Fore.CYAN}Current Market: {SYMBOL} @ {current_close:.4f} | RSI: {rsi:.2f} | ADX: {adx:.2f}")
            print(f"{Fore.BLUE}Long Entry Conditions: "
                  f"{condition_color(log_long_conditions['BB_Cross_Up'])}BB✓ "
                  f"{condition_color(log_long_conditions['ADX_Low'])}ADX✓ "
                  f"{condition_color(log_long_conditions['RSI_OK'])}RSI✓ "
                  f"{condition_color(log_long_conditions['Ichimoku_OK'])}ICHI✓")
            
            print(f"{Fore.MAGENTA}Short Entry Conditions: "
                  f"{condition_color(log_short_conditions['BB_Cross_Down'])}BB✓ "
                  f"{condition_color(log_short_conditions['ADX_Low'])}ADX✓ "
                  f"{condition_color(log_short_conditions['RSI_OK'])}RSI✓ "
                  f"{condition_color(log_short_conditions['Ichimoku_OK'])}ICHI✓")
            
            logging.debug(f"Long Entry Check: {log_long_conditions}")
            logging.debug(f"Short Entry Check: {log_short_conditions}")

            if primary_long_signal and ichimoku_confirms_long:
                 enter_long = True
                 print(f"\n{Fore.GREEN}{Style.BRIGHT}🔵 LONG ENTRY SIGNAL: BB + ADX + RSI + Ichimoku Confirmed.{Style.RESET_ALL}")
                 logging.info("LONG ENTRY SIGNAL: BB + ADX + RSI + Ichimoku Confirmed.")
            elif primary_short_signal and ichimoku_confirms_short:
                 enter_short = True
                 print(f"\n{Fore.RED}{Style.BRIGHT}🔴 SHORT ENTRY SIGNAL: BB + ADX + RSI + Ichimoku Confirmed.{Style.RESET_ALL}")
                 logging.info("SHORT ENTRY SIGNAL: BB + ADX + RSI + Ichimoku Confirmed.")

            # --- Execute Entry ---
            if enter_long or enter_short:
                side = 'buy' if enter_long else 'sell'
                try:
                    # Calculate Amount (Fixed USD)
                    amount = ORDER_SIZE_USD / current_close
                    # Round amount based on market precision
                    amount_str = '{:.{prec}f}'.format(amount, prec=AMOUNT_PRECISION)
                    logging.info(f"Calculated amount: {amount}, Rounded amount: {amount_str}")
                    amount_float = float(amount_str)
                    if amount_float <= 0:
                         logging.error(f"Calculated amount {amount_float} invalid. Skipping entry.")
                         return

                    logging.info(f"Attempting {side.upper()} entry: {amount_str} {SYMBOL.split(':')[0]} @ Market")
                    params = {'category': 'linear'}
                    order = exchange.create_market_order(SYMBOL, side, amount_float, params=params)
                    logging.info(f"Entry order placed: {order.get('id', 'N/A')}")
                    print(f"{Fore.GREEN}Entry order placed: {order.get('id', 'N/A')}")

                    # --- Set State and SL --- #
                    # Use latest available close as approx entry price
                    entry_price_approx = current_close
                    # Calculate Stop Loss Price
                    sl_distance = current_atr * sl_atr_multiple
                    sl_price = None
                    if side == 'buy':
                         sl_price = entry_price_approx - sl_distance
                    else: # sell
                         sl_price = entry_price_approx + sl_distance

                    sl_price_str = '{:.{prec}f}'.format(sl_price, prec=PRICE_PRECISION)
                    logging.info(f"Calculated SL price: {sl_price}, Rounded SL: {sl_price_str}")
                    sl_price_float = float(sl_price_str)

                    new_state = {
                        "active_trade": True,
                        "position_side": 'long' if enter_long else 'short',
                        "entry_price": entry_price_approx,
                        "stop_loss_price": sl_price_float # Store SL price
                    }
                    set_state(new_state)
                    print(f"{Fore.YELLOW}Stop loss set at: {sl_price_float:.4f}")
                    logging.info(f"State updated: Active=True, Side={new_state['position_side']}, Entry={entry_price_approx:.4f}, SL={sl_price_float:.4f}")

                    time.sleep(5) # Pause after order

                except ccxt.InsufficientFunds as e:
                    error_msg = f"Insufficient funds for entry: {e}"
                    logging.error(error_msg)
                    print(f"{Fore.RED}{Style.BRIGHT}{error_msg}{Style.RESET_ALL}")
                except ccxt.ExchangeError as e:
                    error_msg = f"Exchange error on entry: {e}"
                    logging.error(error_msg)
                    print(f"{Fore.RED}{Style.BRIGHT}{error_msg}{Style.RESET_ALL}")
                except Exception as e:
                    error_msg = f"Unexpected error on entry: {e}"
                    logging.error(error_msg, exc_info=True)
                    print(f"{Fore.RED}{Style.BRIGHT}{error_msg}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}No entry conditions met.{Style.RESET_ALL}")
                logging.info("No entry conditions met.")

    except ccxt.NetworkError as e:
        error_msg = f"Network Error in bot cycle: {e}"
        logging.warning(error_msg)
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
    except ccxt.ExchangeError as e:
        error_msg = f"Exchange Error in bot cycle: {e}"
        logging.warning(error_msg)
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
    except Exception as e:
        error_msg = f"Unexpected Error in bot_logic: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"{Fore.RED}{Style.BRIGHT}{error_msg}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}[{datetime.now().strftime('%H:%M:%S')}] Cycle completed {Style.RESET_ALL}")
    logging.info(f"--- SMABBI Cycle End ---\\n")

# --- Schedule Execution --- (Reusing from gpb_bot.py)
print(f"\n{Fore.GREEN}{Style.BRIGHT}Starting SMA20 Bollinger ADX Ichimoku Bot{Style.RESET_ALL}")
print(f"{Fore.CYAN}Checking conditions every {SCHEDULE_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.{Style.RESET_ALL}\n")
logging.info("Starting SMA20 Bollinger ADX Ichimoku Bot")
schedule.every(SCHEDULE_INTERVAL_SECONDS).seconds.do(bot_logic)

# Run once immediately at start
bot_logic()

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
        break
    except Exception as e:
        logging.critical(f"MAIN LOOP ERROR: {e}", exc_info=True)
        logging.info("Sleeping 60s...")
        time.sleep(60)

logging.info("Bot finished.") 