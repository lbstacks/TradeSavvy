import os
from datetime import datetime, time, timedelta
import pytz
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters, ConversationHandler
from tradingview_ta import TA_Handler, Interval
import random
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load token from .env file
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")

# Define valid pairs for each market
FOREX_PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']
COMMODITY_PAIRS = ['XAU/USD', 'XAG/USD', 'WTI/USD', 'BCO/USD', 'NATGAS/USD', 'XCU/USD']
CRYPTO_PAIRS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT']
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA']

# Define conversation states
MARKET_CHOICE, INSTRUMENT_CHOICE, TIMEFRAME_CHOICE, ANOTHER_PAIR, MENU_CHOICE, TRACK_CHOICE = range(6)

# Define trading sessions
ASIAN_SESSION = (time(0, 0), time(9, 0))  # 00:00 - 09:00 UTC
LONDON_SESSION = (time(8, 0), time(17, 0))  # 08:00 - 17:00 UTC
NEW_YORK_SESSION = (time(13, 0), time(22, 0))  # 13:00 - 22:00 UTC

def get_current_session():
    now = datetime.now(pytz.UTC)
    current_time = now.time()
    weekday = now.weekday()

    # Check if it's weekend
    if weekday >= 5:  # 5 is Saturday, 6 is Sunday
        return "Weekend - Markets Closed"

    if time(21, 0) <= current_time or current_time < time(2, 0):
        active = "Asian"
        next_session = "London opens in " + str(time(8, 0).hour - current_time.hour) + " hours"
    elif time(2, 0) <= current_time < time(8, 0):
        active = "Asian"
        next_session = "London"
    elif time(8, 0) <= current_time < time(12, 0):
        active = "London"
        next_session = "New York opens in " + str(time(13, 0).hour - current_time.hour) + " hours"
    elif time(12, 0) <= current_time < time(17, 0):
        active = "London and New York"
        next_session = "London closes in " + str(time(17, 0).hour - current_time.hour) + " hours"
    elif time(17, 0) <= current_time < time(21, 0):
        active = "New York"
        next_session = "Asian opens in " + str(time(21, 0).hour - current_time.hour) + " hours"

    return f"Active: {active}, Next: {next_session}"

def adjust_analysis_for_session(analysis_result, session):
    option = analysis_result['option']
    reason = analysis_result['reason']

    if session == "Asian":
        if option in ['Buy', 'Strong Buy']:
            reason += " Asian session typically has lower volatility, consider tighter stop-loss."
        elif option in ['Sell', 'Strong Sell']:
            reason += " Asian session might have less follow-through on trends, be cautious."
    elif session == "London":
        if option in ['Buy', 'Strong Buy', 'Sell', 'Strong Sell']:
            reason += " London session often sees increased volatility, especially at the open."
    elif session == "New York":
        if option in ['Buy', 'Strong Buy', 'Sell', 'Strong Sell']:
            reason += " New York session can see strong trend continuations or reversals based on US news."
    else:  # Off-Session
        reason += " Current time is between major sessions, expect potentially lower liquidity and unpredictable moves."

    analysis_result['reason'] = reason
    return analysis_result

def convert_to_tradingview_format(instrument, market):
    if market == 'FOREX':
        return instrument.replace("/", "")  # Remove the slash for forex pairs
    elif market == 'CRYPTO':
        return instrument  # Return the instrument as is for crypto
    elif market == 'STOCK':
        return instrument
    return instrument

def interpret_rsi(rsi):
    if rsi >= 70:
        return "Overbought [Strong Bullish]"
    elif rsi >= 60:
        return "Bullish [Moderate Uptrend]"
    elif rsi > 50:
        return "Slightly Bullish [Weak Uptrend]"
    elif rsi == 50:
        return "Neutral"
    elif rsi >= 40:
        return "Slightly Bearish [Weak Downtrend]"
    elif rsi >= 30:
        return "Bearish [Moderate Downtrend]"
    else:
        return "Oversold [Strong Bearish]"

def fetch_real_time_analysis(instrument, timeframe, market):
    interval_map = {
        '5m': Interval.INTERVAL_5_MINUTES,
        '15m': Interval.INTERVAL_15_MINUTES,
        '30m': Interval.INTERVAL_30_MINUTES,
        '1h': Interval.INTERVAL_1_HOUR,
        '4h': Interval.INTERVAL_4_HOURS,
        '1d': Interval.INTERVAL_1_DAY
    }
    
    exchange_map = {
        'FOREX': 'FX_IDC',
        'CRYPTO': 'BINANCE',
        'STOCK': 'NASDAQ'
    }
    
    screener_map = {
        'FOREX': 'forex',
        'CRYPTO': 'crypto',
        'STOCK': 'america'
    }
    
    symbol = convert_to_tradingview_format(instrument, market)
    
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange_map.get(market, 'NASDAQ'),
            screener=screener_map.get(market, 'america'),
            interval=interval_map.get(timeframe, Interval.INTERVAL_1_HOUR),
            timeout=None
        )
        
        analysis = handler.get_analysis()
        recommendation = analysis.summary['RECOMMENDATION']
        
        indicators = analysis.indicators
        rsi = indicators.get('RSI', 0)
        rsi_interpretation = interpret_rsi(rsi)
        
        # Price Action Analysis
        open_price = indicators.get('open', 0)
        close_price = indicators.get('close', 0)
        high_price = indicators.get('high', 0)
        low_price = indicators.get('low', 0)
        
        price_action = analyze_price_action(open_price, close_price, high_price, low_price)
        
        # EMA, MA, and Bollinger Bands
        ema20 = indicators.get('EMA20', 0)
        sma20 = indicators.get('SMA20', 0)
        bb_upper = indicators.get('BB.upper', 0)
        bb_lower = indicators.get('BB.lower', 0)
        
        # MACD
        macd_line = indicators.get('MACD.macd', 0)
        macd_signal = indicators.get('MACD.signal', 0)
        
        # Volume OBV
        obv = indicators.get('OBV', 0)
        
        # Fibonacci Retracement Levels
        fib_levels = calculate_fibonacci_levels(high_price, low_price)
        
        # Analyze indicators
        ma_signal = analyze_ma(close_price, ema20, sma20)
        bb_signal = analyze_bb(close_price, bb_upper, bb_lower)
        macd_signal = analyze_macd(macd_line, macd_signal)
        volume_signal = analyze_volume(obv)
        fib_signal = analyze_fibonacci(close_price, fib_levels)
        
        if recommendation == 'STRONG_BUY':
            option = 'Strong Buy'
            tp_percent = (0.25 + 0.50) / 2  # Midpoint of 0.25 and 0.50
        elif recommendation == 'BUY':
            option = 'Buy'
            tp_percent = (0.20 + 0.25) / 2  # Midpoint of 0.20 and 0.25
        elif recommendation == 'STRONG_SELL':
            option = 'Strong Sell'
            tp_percent = (0.25 + 0.50) / 2  # Midpoint of 0.25 and 0.50
        elif recommendation == 'SELL':
            option = 'Sell'
            tp_percent = (0.20 + 0.25) / 2  # Midpoint of 0.20 and 0.25
        else:
            option = 'Neutral'
            tp_percent = (0.05 + 0.19) / 2  # Midpoint of 0.05 and 0.19

        # Calculate take profit based on the midpoint percentage
        if option in ['Strong Buy', 'Buy']:
            tp_value = close_price * (1 + tp_percent)
        elif option in ['Strong Sell', 'Sell']:
            tp_value = close_price * (1 - tp_percent)
        else:
            tp_value = close_price * (1 + tp_percent)

        # Use TradingView's built-in stop loss
        sl_value = indicators.get('Recommend.Stop_Loss', close_price * 0.98)  # Default to 2% if not available

        # Format the reason
        reason = f"{option} recommendation based on technical analysis:\n"
        reason += f"               RSI: {rsi:.2f} [{rsi_interpretation}]\n"
        reason += f"               Price Action: {price_action}\n"
        reason += f"               MA: {ma_signal}\n"
        reason += f"               Bollinger Bands: {bb_signal}\n"
        reason += f"               MACD: {macd_signal}\n"
        reason += f"               Volume: {volume_signal}\n"
        reason += f"               OBV: {obv}\n"
        reason += f"               Fibonacci: {fib_signal}\n"
        
        return {
            'option': option,
            'entry_point': close_price,
            'take_profit': (tp_value, f"{tp_percent*100:.1f}%"),
            'stop_loss': (sl_value, f"{abs(sl_value - close_price) / close_price * 100:.1f}%"),
            'reason': reason
        }
    except Exception as e:
        print(f"Error fetching analysis for {symbol}: {e}")
        return {
            'option': 'Error',
            'entry_point': 'N/A',
            'take_profit': ('N/A', 'N/A'),
            'stop_loss': ('N/A', 'N/A'),
            'reason': f"Unable to fetch analysis for {symbol}: {str(e)}"
        }

def analyze_ma(close_price, ema20, sma20):
    if close_price > ema20 and close_price > sma20:
        return "Bullish"
    elif close_price < ema20 and close_price < sma20:
        return "Bearish"
    else:
        return "Neutral"

def analyze_bb(close_price, bb_upper, bb_lower):
    if close_price > bb_upper:
        return "Overbought"
    elif close_price < bb_lower:
        return "Oversold"
    else:
        return "Within bands"

def analyze_price_action(open_price, close_price, high_price, low_price):
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price
    
    if total_range == 0:
        return "Doji (Indecision)"
    
    body_to_range_ratio = body_size / total_range
    
    if body_to_range_ratio > 0.6:
        if close_price > open_price:
            return "Bullish Trend"
        else:
            return "Bearish Trend"
    elif body_to_range_ratio < 0.3:
        return "Indecision"
    else:
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price
        if upper_wick > 2 * body_size and lower_wick < body_size / 2:
            return "Potential Reversal (Bearish)"
        elif lower_wick > 2 * body_size and upper_wick < body_size / 2:
            return "Potential Reversal (Bullish)"
        else:
            return "Neutral"

def analyze_macd(macd_line, signal_line):
    if macd_line > signal_line:
        return "Bullish"
    elif macd_line < signal_line:
        return "Bearish"
    else:
        return "Neutral"

def analyze_volume(obv):
    # This is a simplified analysis. In reality, you'd want to compare the current OBV
    # to its moving average or look at its trend over time.
    if obv > 0:
        return "Bullish"
    elif obv < 0:
        return "Bearish"
    else:
        return "Neutral"

def calculate_fibonacci_levels(high, low):
    diff = high - low
    return {
        '0': low,
        '23.6': low + 0.236 * diff,
        '38.2': low + 0.382 * diff,
        '50': low + 0.5 * diff,
        '61.8': low + 0.618 * diff,
        '100': high
    }

def analyze_fibonacci(price, levels):
    for key in ['23.6', '38.2', '50', '61.8']:
        if abs(price - levels[key]) / price < 0.01:  # Within 1% of a Fib level
            return f"Near {key}% retracement level"
    return "Not near any key Fibonacci levels"

def save_user_data(user_id, data):
    """Save user data to a file."""
    with open(f"user_data_{user_id}.pkl", "wb") as file:
        pickle.dump(data, file)

def load_user_data(user_id):
    """Load user data from a file."""
    if os.path.exists(f"user_data_{user_id}.pkl"):
        with open(f"user_data_{user_id}.pkl", "rb") as file:
            return pickle.load(file)
    return {'tracked_pairs': []}

def add_to_tracked_pairs(context, instrument, entry_point):
    user_id = context.user_data.get('user_id', None)
    if not user_id:
        return

    # Load existing data
    user_data = load_user_data(user_id)
    tracked_pairs = user_data.get('tracked_pairs', [])

    current_time = datetime.now(pytz.UTC)
    tracked_pairs.append({
        'instrument': instrument,
        'entry_point': entry_point,
        'date': current_time.strftime("%Y-%m-%d"),
        'time': current_time.strftime("%H:%M:%S UTC"),
        'timestamp': current_time
    })

    # Remove entries older than 6 months
    six_months_ago = current_time - timedelta(days=180)
    tracked_pairs = [
        pair for pair in tracked_pairs
        if pair['timestamp'] > six_months_ago
    ]

    # Save updated data
    user_data['tracked_pairs'] = tracked_pairs
    save_user_data(user_id, user_data)

def group_tracked_pairs(tracked_pairs):
    grouped = defaultdict(lambda: defaultdict(list))
    current_time = datetime.now(pytz.UTC)
    
    for pair in tracked_pairs:
        timestamp = datetime.strptime(f"{pair['date']} {pair['time']}", "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=pytz.UTC)
        time_diff = current_time - timestamp
        
        if time_diff < timedelta(days=7):
            key = 'Daily'
            subkey = timestamp.strftime('%A')  # Day of the week
        elif time_diff < timedelta(days=30):
            key = 'Weekly'
            subkey = f"Week {timestamp.isocalendar()[1]}"  # Week number
        else:
            key = 'Monthly'
            subkey = timestamp.strftime('%B %Y')  # Month name and year
        
        grouped[key][subkey].append(pair)
    
    return grouped

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # Send a welcome message with an icon and prompt to use /menu
    welcome_message = (
        "Welcome to TradeSavvy! 📈\n\n\n"
        "Let's Get Started: /menu"
    )
    await update.message.reply_text(welcome_message)

async def handle_market_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    market = query.data.upper()
    
    if market == 'BACK_TO_MENU':
        return await show_menu(update, context)
    
    context.user_data['market'] = market

    if not is_market_open(market):
        closure_message = get_market_closure_message(market)
        await query.edit_message_text(f"{closure_message}\nYou can still perform analysis, but be aware that the data may not be current.")
    
    if market == 'FOREX':
        forex_message = ("You have selected the 💱 FOREX market. Please enter your currency pair or commodity:\n"
                         "- Currency pairs (e.g., EUR/USD, GBP/USD)\n"
                         "- Commodities: XAU/USD (Gold), XAG/USD (Silver), BCO/USD (Brent Crude Oil)")
        await query.edit_message_text(text=forex_message)
    elif market == 'CRYPTO':
        await query.edit_message_text(text="You have selected the 🪙 CRYPTO market. Please enter your cryptocurrency pair (e.g., BTCUSDT, ETHUSDT).")
    elif market == 'STOCK':
        await query.edit_message_text(text="You have selected the 📈 STOCK market. Please enter your stock symbol (e.g., AAPL, GOOGL).")
    return INSTRUMENT_CHOICE

async def handle_instrument_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper()
    market = context.user_data.get('market', '')
    
    valid_instruments = {
        'FOREX': FOREX_PAIRS + COMMODITY_PAIRS,
        'CRYPTO': CRYPTO_PAIRS,
        'STOCK': STOCK_SYMBOLS
    }
    
    if text in valid_instruments.get(market, []):
        context.user_data['instrument'] = text
        
        # Track the analyzed pair
        if 'tracked_pairs' not in context.user_data:
            context.user_data['tracked_pairs'] = []
        
        # We'll add the entry point and timestamp later when we have the analysis result
        
        await update.message.reply_text(f"Valid instrument received: {text}. Now, please enter the timeframe (e.g., 5m, 15m, 30m, 1h, 4h, 1d).")
        return TIMEFRAME_CHOICE
    else:
        valid_options = ", ".join(valid_instruments.get(market, []))
        await update.message.reply_text(f"Invalid instrument. Please enter a valid option for {market}. Valid options are: {valid_options}")
        return INSTRUMENT_CHOICE

async def timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    timeframe = update.message.text.lower()
    valid_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    if timeframe in valid_timeframes:
        instrument = context.user_data.get('instrument', '')
        market = context.user_data.get('market', '')
        
        await update.message.reply_text(f"Starting analysis for {instrument} on {timeframe} timeframe in the {market} market...")
        
        # Fetch real-time analysis
        analysis_result = fetch_real_time_analysis(instrument, timeframe, market)
        
        if analysis_result['option'] == 'Error':
            await update.message.reply_text(f"Error: {analysis_result['reason']}")
        else:
            # Adjust analysis based on current trading session
            current_session = get_current_session()
            adjusted_result = adjust_analysis_for_session(analysis_result, current_session)
            
            # Get current time in UTC
            current_time = datetime.now(pytz.UTC)
            
            # Update the tracked pair with entry point and timestamp
            for pair in context.user_data['tracked_pairs']:
                if pair['instrument'] == instrument:
                    pair['entry_point'] = adjusted_result['entry_point']
                    pair['date'] = current_time.strftime("%Y-%m-%d")
                    pair['time'] = current_time.strftime("%H:%M:%S UTC")
                    break
            else:
                context.user_data['tracked_pairs'].append({
                    'instrument': instrument,
                    'entry_point': adjusted_result['entry_point'],
                    'date': current_time.strftime("%Y-%m-%d"),
                    'time': current_time.strftime("%H:%M:%S UTC")
                })
            
            # Format the analysis result
            result_message = (
                f"Current Trading Session: {current_session}\n"
                f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                f"Currency Pair: {instrument}\n"
                f"Option: {adjusted_result['option']}\n"
                f"Entry Point: {adjusted_result['entry_point']:.5f}\n"
                f"Take Profit: {adjusted_result['take_profit'][0]:.5f} ({adjusted_result['take_profit'][1]})\n"
                f"Stop Loss: {adjusted_result['stop_loss'][0]:.5f} ({adjusted_result['stop_loss'][1]})\n"
                f"Reason:\n{adjusted_result['reason']}"
            )
            
            await update.message.reply_text("Analysis complete. Here are your results:")
            await update.message.reply_text(result_message)
        
        # Ask if user wants to analyze another pair
        keyboard = [
            [InlineKeyboardButton("Yes", callback_data='yes_another_pair'),
             InlineKeyboardButton("No", callback_data='no_another_pair')],
            [InlineKeyboardButton("🏠 Menu", callback_data='show_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Would you like to analyze another pair?", reply_markup=reply_markup)
        return ANOTHER_PAIR
    else:
        await update.message.reply_text("Invalid timeframe. Please enter a valid timeframe (5m, 15m, 30m, 1h, 4h, 1d).")
        return TIMEFRAME_CHOICE

async def handle_another_pair_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'yes_another_pair':
        await query.edit_message_text("Please enter the new instrument you'd like to analyze:")
        return INSTRUMENT_CHOICE
    elif query.data == 'no_another_pair':
        await query.edit_message_text("Thank you for using TradeSavvy. Type /start to begin a new session or /menu to see other options.")
        return ConversationHandler.END
    elif query.data == 'show_menu':
        return await show_menu(update, context)

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /menu command"""
    # Clear any existing user data
    context.user_data.clear()
    return await show_menu(update, context)

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    keyboard = [
        [InlineKeyboardButton("🏛 MARKET", callback_data='market'),
         InlineKeyboardButton("📝 FEEDBACK", callback_data='feedback')],
        [InlineKeyboardButton("❓ HELP", callback_data='help'),
         InlineKeyboardButton("📊 TRACK", callback_data='track')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = "Welcome to TradeSavvy! Please select an option:"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(message_text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    return MENU_CHOICE

async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    if query.data == 'market':
        return await show_market_menu(update, context)
    elif query.data == 'feedback':
        return await show_feedback(update, context)
    elif query.data == 'help':
        return await show_help(update, context)
    elif query.data == 'track':
        return await show_grouped_pairs(update, context)
    elif query.data == 'back_to_menu':
        return await show_menu(update, context)
    elif query.data.startswith('view_'):
        parts = query.data.split('_')
        if len(parts) == 2:
            return await view_group_details(update, context)
        elif len(parts) == 3:
            return await view_subgroup_details(update, context)

async def show_market_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    keyboard = [
        [InlineKeyboardButton("💱 FOREX", callback_data='forex'),
         InlineKeyboardButton("🪙 CRYPTO", callback_data='crypto')],
        [InlineKeyboardButton("📈 STOCK", callback_data='stock'),
         InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("Please select a market:", reply_markup=reply_markup)
    return MARKET_CHOICE

async def show_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    keyboard = [
        [InlineKeyboardButton("📞 Contact", url=f"https://t.me/{ADMIN_USERNAME}")],
        [InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("To provide feedback or report an issue, please click on 'Contact'.", reply_markup=reply_markup)

async def show_tracked_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query:
        await query.answer()

    tracked_pairs = context.user_data.get('tracked_pairs', [])
    
    if not tracked_pairs:
        message = "You haven't analyzed any currency pairs yet."
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if query:
            await query.edit_message_text(message, reply_markup=reply_markup)
        else:
            await update.message.reply_text(message, reply_markup=reply_markup)
        return MENU_CHOICE

    keyboard = [
        [InlineKeyboardButton("Daily", callback_data='track_daily'),
         InlineKeyboardButton("Weekly", callback_data='track_weekly'),
         InlineKeyboardButton("Monthly", callback_data='track_monthly')],
        [InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message = "How would you like to view your tracked pairs?"
    
    if query:
        await query.edit_message_text(message, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message, reply_markup=reply_markup)
    return TRACK_CHOICE

async def show_grouped_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    context.user_data['user_id'] = user_id

    tracked_pairs = context.user_data.get('tracked_pairs', [])
    grouped_pairs = group_tracked_pairs(tracked_pairs)

    message = "Here are your analyzed pairs:\n\n"
    keyboard = []
    for key, subgroups in grouped_pairs.items():
        pair_count = sum(len(pairs) for pairs in subgroups.values())
        message += f"{key}: {pair_count} pair{'s' if pair_count != 1 else ''}\n"
        keyboard.append([InlineKeyboardButton(f"{key} ({pair_count})", callback_data=f'view_{key}')])

    keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_menu')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup)
    return TRACK_CHOICE

async def view_group_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, key = query.data.split('_', 1)
    tracked_pairs = context.user_data.get('tracked_pairs', [])
    grouped_pairs = group_tracked_pairs(tracked_pairs)

    message = f"Detailed analysis for {key}:\n\n"
    keyboard = []
    for subkey, pairs in grouped_pairs[key].items():
        pair_count = len(pairs)
        message += f"{subkey}: {pair_count} pair{'s' if pair_count != 1 else ''}\n"
        keyboard.append([InlineKeyboardButton(f"{subkey} ({pair_count})", callback_data=f'view_{key}_{subkey}')])

    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='track')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup)
    return TRACK_CHOICE

async def view_subgroup_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, key, subkey = query.data.split('_', 2)
    tracked_pairs = context.user_data.get('tracked_pairs', [])
    grouped_pairs = group_tracked_pairs(tracked_pairs)

    message = f"Detailed analysis for {key} - {subkey}:\n\n"
    for pair in grouped_pairs[key][subkey]:
        message += (f"Currency: {pair['instrument']}\n"
                    f"Entry Point: {pair['entry_point']:.5f}\n"
                    f"Date: {pair['date']}\n"
                    f"Time: {pair['time']}\n\n")

    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f'view_{key}')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup)
    return TRACK_CHOICE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operation cancelled. Use /start to begin again.")
    return ConversationHandler.END

def is_market_open(market):
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()
    current_time = now.time()

    if market == 'FOREX':
        # Forex market is open 24/5
        return weekday < 5 or (weekday == 4 and current_time < time(22, 0))
    elif market == 'CRYPTO':
        # Crypto market is open 24/7
        return True
    elif market == 'STOCK':
        # US Stock market hours (simplified)
        return weekday < 5 and time(13, 30) <= current_time <= time(20, 0)
    
    return False

def get_market_closure_message(market):
    if market == 'FOREX':
        return "The Forex market is currently closed. It opens at 22:00 UTC on Sunday and closes at 22:00 UTC on Friday."
    elif market == 'STOCK':
        return "The US Stock market is currently closed. Regular trading hours are from 9:30 AM to 4:00 PM Eastern Time, Monday to Friday."
    else:
        return ""

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = get_help_text()
    keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_menu')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode='MarkdownV2')

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await show_menu(update, context)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'show_menu':
        return await show_menu(update, context)
    elif query.data == 'market':
        return await show_market_menu(update, context)
    elif query.data == 'feedback':
        return await show_feedback(update, context)
    elif query.data == 'help':
        return await show_help(update, context)
    elif query.data == 'track':
        return await show_tracked_pairs(update, context)
    elif query.data == 'back_to_menu':
        return await show_menu(update, context)

async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Indicate the bot is "thinking" by sending a typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

    help_text = get_help_text()
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(help_text, reply_markup=reply_markup, parse_mode='MarkdownV2')

def get_help_text():
    return (
        "*Welcome to TradeSavvy*\n\n"
        "Your Telegram bot for analyzing currency pairs and optimizing your trades\\. \n"
        "Simply select the market and the currency pair of your choice, and TradeSavvy will provide signals "
        "for entry, take profit \\(TP\\), and stop loss \\(SL\\), based on advanced TradingView indicators\\.\n\n"
        "We utilize the following technical indicators to ensure accurate analysis:\n\n"
        "📊 *Price Action Strategy*: Focuses on past price movements to predict future trends\\.\n\n"
        "📈 *Relative Strength Index \\(RSI\\)*: Measures the speed and change of price movements to identify overbought or oversold conditions\\.\n\n"
        "〰️ *Moving Averages \\(MA\\)*: Helps to smooth price data to identify trends\\.\n\n"
        "🎯 *Bollinger Bands*: Indicates market volatility and price levels relative to historical performance\\.\n\n"
        "📉 *Moving Average Convergence Divergence \\(MACD\\)*: Determines trend direction and strength through the convergence and divergence of moving averages\\.\n\n"
        "📊 *Volume*: Measures the number of shares or contracts traded in a market, signaling the strength of a price move\\.\n\n"
        "📶 *On\\-Balance Volume \\(OBV\\)*: A momentum indicator that relates volume flow to price movements\\.\n\n"
        "🔢 *Fibonacci Retracement Levels*: Identifies potential reversal levels using a mathematical sequence\\.\n\n"
        "In addition, the bot provides updates on the current trading session to help you strategize your trades in the correct market context\\.\n\n"
        "⚠️ *Disclaimer*: The information provided by TradeSavvy is for educational purposes only\\. Always perform your due diligence "
        "and confirm trading signals before executing any trades\\. Trading carries risks, and it is your responsibility to ensure "
        "you are comfortable with those risks before engaging in live market transactions\\.\n\n"
        "*Use /start to begin again*\\.\n\n"
        "*Version 1\\.0\\.0*\n\n"
        "Copyright © 2024 TradeSavvy\\. All rights reserved\\.\n\n"
        "*Made with ❤️ by @lifofkojolb*"
    )

async def clear_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Send 50 blank messages to clear the chat
    for _ in range(50):
        await update.message.reply_text("\u2800")  # Unicode character for a blank space
    
    # Send a confirmation message
    await update.message.reply_text("Chat cleared! 🧹✨")

def main():
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('menu', show_menu)],
        states={
            MARKET_CHOICE: [CallbackQueryHandler(handle_market_choice)],
            INSTRUMENT_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_instrument_choice)],
            TIMEFRAME_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, timeframe)],
            ANOTHER_PAIR: [CallbackQueryHandler(handle_another_pair_choice)],
            MENU_CHOICE: [CallbackQueryHandler(handle_menu_choice)],
            TRACK_CHOICE: [CallbackQueryHandler(handle_menu_choice)],
        },
        fallbacks=[CommandHandler('menu', show_menu), CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(CommandHandler('menu', show_menu))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('clear', clear_chat))

    application.run_polling()

if __name__ == '__main__':
    main()
