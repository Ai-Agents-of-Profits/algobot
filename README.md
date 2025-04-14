# SMABBI Trading Bot

## Strategy Overview
The SMABBI (SMA20 Bollinger Bands ADX Ichimoku) strategy combines multiple technical indicators to identify high-probability trading setups on Bybit futures.

### Core Indicators
- **SMA20**: 20-period Simple Moving Average
- **Bollinger Bands**: Volatility bands (20,2)
- **ADX**: Average Directional Index for trend strength
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **Ichimoku Cloud**: Japanese charting system for trend confirmation

### Optimized Parameters
- Timeframe: 15m
- ADX Threshold: 35 (less restrictive)
- RSI Long Entry: 47 (relaxed from original 40)
- RSI Short Entry: 53 (relaxed from original 60)
- RSI Long Exit: 70 (unchanged)
- RSI Short Exit: 30 (unchanged)
- Stop Loss Multiple: 1.8 × ATR (tighter than original 2.5)
- Order Size: $25 USD

### Performance Metrics
- 564% return with 80.6% win rate and 10.6 profit factor (COREUSDT 2024 data)

## Setup Instructions

### Requirements
1. Python 3.8+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Configuration
1. Create a `.env` file in the project directory with your Bybit API credentials:
```
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

## Bot Structure
- **sma_bb_adx_ichi_bot.py**: Main bot file with trading logic
- **state_manager_smabbi.py**: Manages trade state persistence
- **functions_smabbi.py**: Technical indicator calculations and helper functions

## Usage

### Running the Bot
```bash
python sma_bb_adx_ichi_bot.py
```

### Configuration
Edit the constants at the top of `sma_bb_adx_ichi_bot.py` to adjust:
- Symbol (default: CORE/USDT:USDT)
- Timeframe (default: 15m)
- Order Size (default: $25 USD)
- Check Interval (default: 30 seconds)
- Strategy Parameters

### Testnet Mode
Set `USE_TESTNET = True` for paper trading before using real funds.

## Risk Management
- Dynamic ATR-based stop loss
- Trailing stop implementation
- Position sizing based on fixed USD amount

## Disclaimer
This bot is for educational purposes. Trading cryptocurrency futures involves substantial risk. Past performance is not indicative of future results.

## License
MIT
