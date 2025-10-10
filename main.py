import json, os, shutil
from datetime import datetime, timedelta
import csv
from pathlib import Path
import pandas as pd
import yfinance as yf
from stock_holding import StockHolding
from stock_predictor import StockPredictor
from tui_utils import print_header, print_center, term_width
from typing import Optional
try:
    import plotext as plt
except ImportError:
    plt = None
from colorama import init, Fore, Style
try:
    import pyfiglet
except Exception:
    pyfiglet = None

APP_NAME = "Virtual Brokerage Account"

PRED_TARGET_TIMING = "close"            # Predictor timing toggle 

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
PRED_CSV = LOG_DIR / "predictions.csv"
USERS_FILE = "users.json"

RUNE_ART = r"""
              ,---------------------------,
              |  /---------------------\  |
              | |                       | |
              | |       Virtual         | |
              | |      Brokerage        | |
              | |       Account         | |
              | |                       | |
              |  \_____________________/  |
              |___________________________|
            ,---\_____     []     _______/------,
          /         /______________\           /|
        /___________________________________ /  | ___
        |                                   |   |    )
        |  _ _ _                 [-------]  |   |   (
        |  o o o                 [-------]  |  /    _)_
        |__________________________________ |/     /  /
    /-------------------------------------/|      ( )/
  /-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/ /
/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/ /
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def make_ascii_title(text: str) -> list[str]:
    """if pyfiglet:
        banner = pyfiglet.figlet_format(text, font="Standard")
        # split lines and trailing spaces
        return [ln.rstrip() for ln in banner.splitlines()]"""
    # fallback minimal title
    lines = [
        "╦  ╦┬┬─┐┌┬┐┬ ┬┌─┐┬    ╔╗ ┬─┐┌─┐┬┌─┌─┐┬─┐┌─┐┌─┐┌─┐",
        "╚╗╔╝│├┬┘ │ │ │├─┤│    ╠╩╗├┬┘│ │├┴┐├┤ ├┬┘├─┤│ ┬├┤ ",
        " ╚╝ ┴┴└─ ┴ └─┘┴ ┴┴─┘  ╚═╝┴└─└─┘┴ ┴└─┘┴└─┴ ┴└─┘└─┘",
        "",
        "      Stock Trading Simulator with ML & Real-Time Data"
    ]
    subtitle = [line.center(len(lines[0])) for line in text.splitlines()]
    return lines + [""] + subtitle

def render_welcome_screen():
    cols = term_width(120)
    title_lines = make_ascii_title("Virtual Brokerage")
    art_lines = [line.rstrip() for line in RUNE_ART.strip().splitlines()]

    #calculate widths
    title_width = max(len(ln) for ln in title_lines) if title_lines else 0
    art_width = max(len(ln) for ln in art_lines) if art_lines else 0
    gap = 4

    #Ensure it fits
    total_width = title_width + gap + art_width
    if total_width > cols:
        #Reduce the gap if needed
        gap = max(2, cols - title_width - art_width)
    
    max_lines = max(len(title_lines), len(art_lines))

    #Print header
    print("\n" + "=" * cols) 
    print()
    
    #Render side by side
    for i in range(max_lines):
        title_line = title_lines[i] if i < len(title_lines) else ""
        art_line = art_lines[i] if i < len(art_lines) else ""

        #Left justify title, right justify art
        title_part = title_line.ljust(title_width)

        # Right justify art (pad from the left to push it right)
        # Preserve the exact art content without truncation
        art_part = art_line.ljust(art_width)
        
        print(Fore.CYAN + Style.BRIGHT + title_part + Style.RESET_ALL + 
              " " * gap + 
              Fore.CYAN + Style.BRIGHT + art_part + Style.RESET_ALL)
    
    print()
    print("═" * cols)
    print()

#----------------------------------
# User Authentication System
#----------------------------------

def load_users():
    """Load user credentials from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
        return {}

def save_users(users_dict):
    """Save user credentials to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users_dict, f, indent=4)
        return True
    except Exception as e:
        print(Fore.RED + f"Error saving users: {e}" + Style.RESET_ALL)
        return False

def signup():
    """Handle new user registration"""
    users = load_users()

    print_header(Fore.CYAN + Syle.BRIGHT + "\n✦ Create New Account ✦" + Style.RESET_ALL)
    print()
    
    while True:
        username = input(Fore.CYAN + "➤ Choose a username (or 'back' to return): " + Style.RESET_ALL).strip()
        
        if username.lower() == 'back':
            return None
        
        if not username:
            print(Fore.YELLOW + "Username cannot be empty." + Style.RESET_ALL)
            continue
        
        if username in users:
            print(Fore.YELLOW + f"Username '{username}' already exists. Please choose another." + Style.RESET_ALL)
            continue
        
        if len(username) < 3:
            print(Fore.YELLOW + "Username must be at least 3 characters long." + Style.RESET_ALL)
            continue
        
        break
    
    while True:
        password = input(Fore.CYAN + "Choose a password (min 4 characters): " + Style.RESET_ALL).strip()
        
        if len(password) < 4:
            print(Fore.YELLOW + "Password must be at least 4 characters long." + Style.RESET_ALL)
            continue
        
        confirm = input(Fore.CYAN + "Confirm password: " + Style.RESET_ALL).strip()
        
        if password != confirm:
            print(Fore.YELLOW + "Passwords do not match. Please try again." + Style.RESET_ALL)
            continue
        
        break
    
    # Save new user
    users[username] = {
        "password": password,
        "created": datetime.now().isoformat(),
        "data_file": f"user_{username}_data.json"
    }
    
    if save_users(users):
        print(Fore.GREEN + Style.BRIGHT + f"\n✓ Account created successfully! Welcome, {username}!" + Style.RESET_ALL)
        return username, users[username]["data_file"]
    else:
        print(Fore.RED + "Failed to create account. Please try again." + Style.RESET_ALL)
        return None

def login():
    """Handle user login"""
    users = load_users()
    
    print_header(Fore.CYAN + Style.BRIGHT + "\n✦ Login ✦" + Style.RESET_ALL)
    print()
    
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        username = input(Fore.CYAN + "Username: " + Style.RESET_ALL).strip()
        password = input(Fore.CYAN + "Password: " + Style.RESET_ALL).strip()
        
        if username in users and users[username]["password"] == password:
            print(Fore.GREEN + Style.BRIGHT + f"\n✓ Welcome back, {username}!" + Style.RESET_ALL)
            return username, users[username]["data_file"]
        else:
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(Fore.YELLOW + f"Invalid credentials. {remaining} attempt(s) remaining." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Maximum login attempts exceeded." + Style.RESET_ALL)
                return None
    
    return None

def auth_menu():
    """Show authentication menu and handle user choice"""
    while True:
        print_center(Fore.CYAN + Style.BRIGHT + "══════════════════════════════════════════" + Style.RESET_ALL)
        print_center(Fore.CYAN + "1. Login" + Style.RESET_ALL)
        print_center(Fore.CYAN + "2. Sign Up (Create New Account)" + Style.RESET_ALL)
        print_center(Fore.CYAN + "3. Exit" + Style.RESET_ALL)
        print_center(Fore.CYAN + Style.BRIGHT + "══════════════════════════════════════════" + Style.RESET_ALL)
        print()
        
        choice = input(Fore.CYAN + "➤ Select an option (1-3): " + Style.RESET_ALL).strip()
        
        if choice == "1":
            result = login()
            if result:
                return result
        elif choice == "2":
            result = signup()
            if result:
                return result
        elif choice == "3":
            print(Fore.CYAN + "➤ Goodbye!" + Style.RESET_ALL)
            exit(0)
        else:
            print(Fore.YELLOW + "➤ Invalid option. Please choose 1, 2, or 3." + Style.RESET_ALL)


def _next_trading_day_index(df: pd.DataFrame, as_of_idx: pd.Timestamp) -> Optional[pd.Timestamp]:
    # df index is trading dates; get the next one after as_of_idx
    idx = df.index
    pos = idx.searchsorted(as_of_idx, side="right")
    if pos < len(idx):
        return idx[pos]
    return None

def log_prediction(symbol: str, 
                   timing: str, 
                   prob_up: float,
                   label: str,
                   decision: str,
                   stock_df: pd.DataFrame,
                   market_df: pd.DataFrame):
    """   Append prediction and try to backfill realized outcomes for prior rows.  """
    now_utc = datetime.utcnow().isoformat()
    today_idx = stock_df.index[-1]
    target_idx = _next_trading_day_index(stock_df, today_idx) if timing == "close" else today_idx
    today_close = float(stock_df["Close"].iloc[-1])
    market_close = float(market_df["Close"].iloc[-1])if market_df is not None and not market_df.empty else None

    # Read existing
    if PRED_CSV.exists():
        df_log = pd.read_csv(PRED_CSV)
    else:
        df_log = pd.DataFrame(columns=[
            "ts_utc","symbol","timing","today_date","target_date",
            "prob_up","label_pred","decision","today_close","realized_label",
            "realized_close","pnl_bps","market_close","market_ret"
        ])

    new_row = {
        "ts_utc": now_utc,
        "symbol": symbol,
        "timing": timing,
        "today_date": pd.to_datetime(str(today_idx)).date(),
        "target_date": pd.to_datetime(str(target_idx)).date() if target_idx else "",
        "prob_up": prob_up,
        "label_pred": label,
        "decision": decision,
        "today_close": today_close,
        "realized_label": "",
        "realized_close": "",
        "pnl_bps": "",
        "market_close": market_close,
        "market_ret": ""
    }
    df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)

    # Try to backfill realized for any rows whose target_date <= latest available
    latest_date = stock_df.index[-1].date()
    for i, row in df_log[df_log["realized_label"].eq("") & df_log["target_date"].ne("")].iterrows():
        tgt_date = pd.to_datetime(row["target_date"]).date()
        if tgt_date <= latest_date:
            # find close on tgt_date
            try:
                close_tgt = float(stock_df.loc[pd.Timestamp(tgt_date)]["Close"])
                close_ref = float(stock_df.loc[pd.Timestamp(row["today_date"])]["Close"])
                realized_label = "UP" if close_tgt > close_ref else "DOWN"
                pnl_bps = (close_tgt / close_ref - 1.0) * 10000.0
                df_log.at[i, "realized_close"] = close_tgt
                df_log.at[i, "realized_label"] = realized_label
                df_log.at[i, "pnl_bps"] = round(pnl_bps, 2)
                if market_df is not None and not market_df.empty:
                    mc_ref = float(market_df.loc[pd.Timestamp(row["today_date"])]["Close"])
                    mc_tgt = float(market_df.loc[pd.Timestamp(tgt_date)]["Close"])
                    mret = (mc_tgt / mc_ref - 1.0) * 100.0
                    df_log.at[i, "market_ret"] = round(mret, 3)
            except KeyError:
                # target date missing (holiday/newest day not in set yet); skip
                pass
            except Exception:
                pass

    df_log.to_csv(PRED_CSV, index=False)



# -------------------------
# Data helpers (yfinance)
# -------------------------

def fetch_ohlcv(symbol: str, period="3y", interval="1d") -> pd.DataFrame | None:
    try:
        t = yf.Ticker(symbol)
        h = t.history(period=period, interval=interval, auto_adjust=False)
        if h is None or h.empty:
            return None
        # Ensure expected columns exist
        need = {"Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(set(h.columns)):
            return None
        # Keep Adj Close if present for split/div adjustments in predictor
        return h.dropna(subset=["Close"])
    except Exception:
        return None

def fetch_price(symbol):
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="1d")
        if h is None or h.empty:
            return None
        return float(h["Close"].iloc[-1])
    except Exception:
        return None

def fetch_history(symbol, period="1y", interval="1d"):
    return fetch_ohlcv(symbol, period=period, interval=interval)

def sector_etf_for_symbol(symbol: str) -> str | None:
    # Lightweight mapping from sector name to SPDR ETF
    sector_to_etf = {
        "Technology": "XLK",
        "Information Technology": "XLK",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Health Care": "XLV",
        "Energy": "XLE",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
        "Utilities": "XLU",
    }
    try:
        info = yf.Ticker(symbol).info
        sector = info.get("sector")
        if sector and sector in sector_to_etf:
            return sector_to_etf[sector]
    except Exception:
        pass
    return None  

# -------------------------
# UI helpers
# -------------------------

def sparkline(values, width=42):
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    vmin = min(values); vmax = max(values)
    if vmax == vmin:
        return blocks[0] * min(width, len(values))
    data = values
    n = len(values)
    if n > width:
        step = n / width
        data = [values[int(i * step)] for i in range(width)]
    out = []
    for v in data:
        idx = int((v - vmin) / (vmax - vmin) * (len(blocks) - 1))
        out.append(blocks[idx])
    return "".join(out)

def market_index_rows():
    entries = []
    indices = [
        ("S&P 500", "^GSPC"),
        ("NASDAQ", "^IXIC"),
        ("NYSE", "^NYA"),
    ]
    for name, sym in indices:
        hist = fetch_history(sym, period="3mo", interval="1d")
        last = None; change_pct = None; sline = ""
        if hist is not None and not hist.empty:
            closes = [float(x) for x in hist["Close"].values.tolist()]
            sline = sparkline(closes, width=36)
            last = closes[-1]
            if len(closes) >= 2:
                prev = closes[-2]
                if prev:
                    change_pct = (last - prev) / prev * 100.0
        entries.append((name, sym, last, change_pct, sline))
    return entries

def show_research_dashboard():
    print_header(Fore.CYAN + Style.BRIGHT + "\n📊 Research - Market Overview" + Style.RESET_ALL)
    rows = market_index_rows()
    left_w = 36
    for name, sym, last, chg, sline in rows:
        last_str = f"${last:.2f}" if last is not None else "N/A"
        chg_color = Fore.GREEN if chg and chg >= 0 else Fore.RED
        chg_str = f"{chg:+.2f}%" if chg is not None else ""

        left = f"{name} ({sym}) {last_str:>12} {chg_str:>8}"
        left = left[:left_w].ljust(left_w)
        print(Fore.CYAN + left + " " + sline + Style.RESET_ALL)
    print(Fore.CYAN + f"⏱ Prediction timing: {PRED_TARGET_TIMING.upper()} - type 'toggle' to switch" + Style.RESET_ALL)
    print(Fore.CYAN + "\nType a ticker to view its chart and prediction or 'back' to return." + Style.RESET_ALL)

def show_symbol_chart(sym, hist):
    closes = [float(x) for x in hist["Close"].values.tolist()] if (hist is not None and not hist.empty) else []
    if closes:
        sline = sparkline(closes, width=60)
        print(Fore.CYAN + f"\n{sym} 1Y Sparkline: " + sline + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + f"\nNo close data available to render sparkline for {sym}." + Style.RESET_ALL)

    if plt and hist is not None and not hist.empty:
        try:
            cols, rows = shutil.get_terminal_size(fallback=(100, 30))
            plot_w = max(60, min(cols - 4, 140))
            plot_h = max(12, min(rows - 8, 32))
            plt.clf()
            plt.plotsize(plot_w, plot_h)
            dates = [d.strftime("%Y-%m-%d") for d in hist.index]
            prices = closes
            plt.title(f"{sym} Price - Last 1Y")
            plt.plot(dates, prices, marker="")
            if len(dates) > 12:
                step = max(1, len(dates) // 12)
                xticks = [dates[i] for i in range(0, len(dates), step)]
                plt.xticks(xticks)
            plt.show()
        except Exception:
            print(Fore.YELLOW + "(Chart rendering skipped due to a terminal or plotting error.)" + Style.RESET_ALL)
    else:
        if plt is None:
            print(Fore.RED + "(Install 'plotext' to enable full-size terminal charts.)" + Style.RESET_ALL)

# -------------------------
# Research scene (uses predictor)
# -------------------------

def research_scene(balance, portfolio, watchlist):
    global PRED_TARGET_TIMING
    while True:
        show_research_dashboard()
        sym = input(Fore.CYAN + "➤ Enter symbol (or 'toggle'/'back'): " + Style.RESET_ALL).strip()

        if not sym:
            continue

        if sym.lower() in ("back", "b"):
            break

        if sym.lower() == "toggle":
            PRED_TARGET_TIMING = "open" if PRED_TARGET_TIMING == "close" else "close"
            print(Fore.GREEN + f"✓ Switched prediction timing to {PRED_TARGET_TIMING.upper()}" + Style.RESET_ALL)
            continue

        sym = sym.upper()

        # Chart window (1y)
        hist_1y = fetch_history(sym, period="1y", interval="1d")

        if hist_1y is None or hist_1y.empty:
            print(Fore.RED + f"Could not retrieve data for {sym}." + Style.RESET_ALL)
            continue
        last_price = float(hist_1y["Close"].iloc[-1])
        print(Fore.CYAN + f"💵 Current Price of {sym}: ${last_price:,.2f}" + Style.RESET_ALL)
        show_symbol_chart(sym, hist_1y)

        # Predictor window (3y) with market + sector context
        stock_df = fetch_ohlcv(sym, period="3y", interval="1d")
        market_df = fetch_ohlcv("^GSPC", period="3y", interval="1d")
        sector_sym = sector_etf_for_symbol(sym)
        sector_df = fetch_ohlcv(sector_sym, period="3y", interval="1d") if sector_sym else None

        try:
            if stock_df is None or market_df is None or stock_df.empty or market_df.empty:
                raise ValueError("Insufficient history for model.")
            predictor = StockPredictor(
                target_timing=PRED_TARGET_TIMING,
                window_days=756,
                n_splits=4,
                confidence_threshold=0.55
            )
            metrics = predictor.fit(stock_df, market_df, sector_df)
            res = predictor.predict_next(stock_df, market_df, sector_df)

            direction_color = Fore.GREEN if res.label == "UP" else Fore.RED
            print(Fore.CYAN + f"\n🤖 ML Prediction: " + direction_color + f"{res.label}" + Fore.CYAN + f" (confidence: {res.confidence:.2f}, p_up: {res.prob_up:.2f})" + Style.RESET_ALL)
            print(Fore.CYAN + f"📋 Decision: {res.decision}. {res.reason}" + Style.RESET_ALL)
            print(Fore.CYAN + f"💡 Reason: {res.reason}" + Style.RESET_ALL)
            if "rolling_auc" in res.metrics:
                print(Fore.CYAN + f"📊 Model AUC: {res.metrics['rolling_auc']:.3f}" + Style.RESET_ALL)
            
            # Log prediction and attempt to backfill previous realized outcomes 
            log_prediction(sym, PRED_TARGET_TIMING, res.prob_up, res.label, res.descision, stock_df, market_df)
        except Exception as e:
            print(Fore.YELLOW + f"⚠ ML Prediction unavailable: {e}" + Style.RESET_ALL)

        input(Fore.CYAN + "➤ Press Enter to continue..." + Style.RESET_ALL)

# -------------------------
# App shell
# -------------------------

def main():
    init(autoreset=True)
    render_welcome_screen()

    #Authentication 
    auth_result = auth_menu()
    if not auth_result:
        return

    username, date_file = auth_result

    balance = 100000.0
    portfolio = []
    watchlist = []

    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                saved = json.load(f)
                balance = saved.get("balance", balance)
                for item in saved.get("portfolio", []):
                    sym = item["symbol"]; qty = item["quantity"]; avg = item.get("avg_price", 0.0)
                    portfolio.append(StockHolding(sym, qty, avg))
                watchlist = saved.get("watchlist", watchlist)
        except Exception:
            balance = 100000.0; portfolio = []; watchlist = []

    def save_data():
        data = {
            "balance": balance,
            "portfolio": [
                {"symbol": sh.symbol, "quantity": sh.quantity, "avg_price": sh.avg_price} for sh in portfolio
            ],
            "watchlist": watchlist
        }
        try:
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(Fore.RED + f"⚠ Warning: could not save data: {e}" + Style.RESET_ALL)

    def get_current_price(symbol):
        return fetch_price(symbol)

    try:
        while True:
            print_header(Fore.CYAN + Style.BRIGHT + "\n✦ Main Menu:" + Style.RESET_ALL)
            print(Fore.CYAN + "1. 💰 Account - View account balance and info" + Style.RESET_ALL)
            print(Fore.CYAN + "2. 📊 Portfolio - View or trade your holdings" + Style.RESET_ALL)
            print(Fore.CYAN + "3. 👁 Watchlist - Manage watchlist" + Style.RESET_ALL)
            print(Fore.CYAN + "4. 🔬 Research - Index overview, charts, ML" + Style.RESET_ALL)
            print(Fore.CYAN + "5. ❓ Help - Show available commands" + Style.RESET_ALL)
            print(Fore.CYAN + "6. 🚪 Exit" + Style.RESET_ALL)
            choice = input(Fore.CYAN + "➤ Enter a command or number: " + Style.RESET_ALL).strip().lower()
            if choice in ["6", "exit", "quit"]:
                print(Fore.GREEN + "💾 Saving data... Goodbye!" + Style.RESET_ALL)
                break
            elif choice in ["5", "help", "h"]:
                print_header(Fore.CYAN + Style.BRIGHT + "\n📖 Help" + Style.RESET_ALL)
                print(Fore.CYAN + "Available commands:" + Style.RESET_ALL)
                print(Fore.CYAN + "  • account (a) - View account information" + Style.RESET_ALL)
                print(Fore.CYAN + "  • portfolio (p) - Manage portfolio" + Style.RESET_ALL)
                print(Fore.CYAN + "  • watchlist (w) - Manage watchlist" + Style.RESET_ALL)
                print(Fore.CYAN + "  • research (r) - Market research & ML" + Style.RESET_ALL)
                print(Fore.CYAN + "  • exit - Save and quit" + Style.RESET_ALL)
                input(Fore.CYAN + "\n➤ Press Enter to continue..." + Style.RESET_ALL)

            elif choice in ["1", "account", "a"]:
                print(Fore.CYAN + Style.BRIGHT + "\n💰 Account Information:" + Style.RESET_ALL)
                total_portfolio_val = 0.0
                for sh in portfolio:
                    price = get_current_price(sh.symbol)
                    if price is not None:
                        total_portfolio_val += price * sh.quantity
                print(Fore.CYAN + f"💵 Cash Balance: ${balance:,.2f}" + Style.RESET_ALL)
                print(Fore.CYAN + f"📊 Portfolio Market Value: ${total_portfolio_val:.2f}" + Style.RESET_ALL)
                print(Fore.CYAN + f"💎 Total Net Value: ${(balance + total_portfolio_val):.2f}" + Style.RESET_ALL)
                input(Fore.CYAN + "➤ Press Enter to return to main menu..." + Style.RESET_ALL)
            elif choice in ["2", "portfolio", "p"]:
                while True:
                    print(Fore.CYAN + Style.BRIGHT + "\n📊 Your Portfolio Holdings:" + Style.RESET_ALL)
                    if portfolio:
                        print(Fore.CYAN + f"{'Symbol':<10}{'Quantity':>10}{'Avg Cost':>12}{'Curr Price':>12}{'Value':>15}" + Style.RESET_ALL)
                        total_val = 0.0
                        for sh in portfolio:
                            price = get_current_price(sh.symbol)
                            curr_price = price if price is not None else sh.avg_price
                            value = curr_price * sh.quantity
                            total_val += value

                            pnl = ((curr_price - sh.avg_price) / sh.avg_price * 100) if sh.avg_price > 0 else 0
                            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED

                            print(Fore.CYAN + f"{sh.symbol:<10}{sh.quantity:>10}{sh.avg_price:>12.2f}{curr_price:>12.2f}{value:>15.2f}" + pnl_color + f"{pnl:>11.2f}%" + Style.RESET_ALL)
                        print(Fore.CYAN + f"\nTotal Portfolio Value: " + Fore.GREEN + f"${total_val:,.2f}" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "(Portfolio is empty)" + Style.RESET_ALL)
                    
                    print()
                    action = input(Fore.CYAN + "➤ Command [buy <SYM> <QTY> / sell <SYM> <QTY> / back]: " + Style.RESET_ALL).strip().lower()
                    if not action:
                        continue
                    if action in ("back", "b"):
                        break
                    parts = action.split()
                    cmd = parts[0] if parts else ""
                    if cmd == "buy":
                        if len(parts) < 3:
                            print(Fore.YELLOW + "Usage: buy <symbol> <quantity>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        try:
                            qty = int(parts[2])
                        except ValueError:
                            print(Fore.RED + "Invalid quantity." + Style.RESET_ALL); continue
                        if qty <= 0:
                            print(Fore.RED + "Quantity must be positive." + Style.RESET_ALL); continue
                        price = get_current_price(sym)
                        if price is None:
                            print(Fore.RED + f"Symbol {sym} not found." + Style.RESET_ALL); continue
                        cost = price * qty
                        if cost > balance:
                            print(Fore.RED + "Insufficient funds to buy." + Style.RESET_ALL)
                        else:
                            balance -= cost
                            for sh in portfolio:
                                if sh.symbol == sym:
                                    sh.add_shares(qty, price); break
                            else:
                                portfolio.append(StockHolding(sym, qty, price))
                            print(Fore.GREEN + f"✓ Bought {qty} shares of {sym} at ${price:.2f} each, cost ${cost:.2f}." + Style.RESET_ALL)
                    elif cmd == "sell":
                        if len(parts) < 3:
                            print(Fore.YELLOW + "Usage: sell <symbol> <quantity>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        try:
                            qty = int(parts[2])
                        except ValueError:
                            print(Fore.RED + "Invalid quantity." + Style.RESET_ALL); continue
                        if qty <= 0:
                            print(Fore.RED + "Quantity must be positive." + Style.RESET_ALL); continue
                        found = False
                        for sh in portfolio:
                            if sh.symbol == sym:
                                found = True
                                if qty > sh.quantity:
                                    print(Fore.RED + "Not enough shares to sell." + Style.RESET_ALL)
                                else:
                                    price = get_current_price(sym)
                                    if price is None:
                                        print(Fore.RED + f"Could not retrieve price for {sym}. Sell aborted." + Style.RESET_ALL)
                                    else:
                                        revenue = price * qty
                                        balance += revenue
                                        all_sold = sh.remove_shares(qty)
                                        if all_sold:
                                            portfolio.remove(sh)
                                        print(Fore.GREEN + f"✓ Sold {qty} shares of {sym} at ${price:.2f} each, revenue ${revenue:.2f}." + Style.RESET_ALL)
                                break
                        if not found:
                            print(Fore.YELLOW + f"You do not own any shares of {sym}." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "Unknown command. Use 'buy', 'sell', or 'back'." + Style.RESET_ALL)
            elif choice in ["3", "watchlist", "w"]:
                while True:
                    print_header(Fore.CYAN + Style.BRIGHT + "\n👁 Your Watchlist" + Style.RESET_ALL)
                    
                    if watchlist:
                        print(Fore.CYAN + f"{'Symbol':<10}{'Price':>12}{'Change':>12}{'Trend':>40}" + Style.RESET_ALL)
                        print(Fore.CYAN + "─" * 74 + Style.RESET_ALL)
                        
                        for sym in watchlist:
                            price = get_current_price(sym)
                            change_str = ""
                            change_color = Fore.CYAN
                            
                            if price is not None:
                                try:
                                    hist2 = yf.Ticker(sym).history(period="5d")
                                except Exception:
                                    hist2 = None
                                
                                if hist2 is not None and len(hist2) >= 2:
                                    closes = hist2['Close'].values.tolist()
                                    if len(closes) >= 2:
                                        prev = closes[-2]
                                        last = closes[-1]
                                        if prev and last:
                                            change_pct = (last - prev) / prev * 100
                                            change_str = f"{change_pct:+.2f}%"
                                            change_color = Fore.GREEN if change_pct >= 0 else Fore.RED
                                    
                                    # Mini sparkline
                                    mini_spark = sparkline(closes, width=38) if len(closes) >= 2 else ""
                                else:
                                    mini_spark = ""
                            else:
                                mini_spark = ""
                            
                            price_str = f"${price:.2f}" if price is not None else "N/A"
                            print(Fore.CYAN + f"{sym:<10}{price_str:>12}" + 
                                  change_color + f"{change_str:>12}" + 
                                  Fore.GREEN + f"{mini_spark:>40}" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "(Watchlist is empty)" + Style.RESET_ALL)
                    
                    print()
                    action = input(Fore.CYAN + "➤ Command [add <SYM> / remove <SYM> / back]: " + Style.RESET_ALL).strip().lower()
                    
                    if not action:
                        continue
                    
                    if action in ("back", "b"):
                        break
                    
                    parts = action.split()
                    cmd = parts[0] if parts else ""
                    
                    if cmd == "add":
                        if len(parts) < 2:
                            print(Fore.YELLOW + "Usage: add <symbol>" + Style.RESET_ALL)
                            continue
                        
                        sym = parts[1].upper()
                        
                        if sym in watchlist:
                            print(Fore.YELLOW + f"{sym} is already in your watchlist." + Style.RESET_ALL)
                        else:
                            price = get_current_price(sym)
                            if price is None:
                                print(Fore.RED + f"Symbol {sym} not found or unavailable." + Style.RESET_ALL)
                            else:
                                watchlist.append(sym)
                                print(Fore.GREEN + f"✓ Added {sym} to watchlist." + Style.RESET_ALL)
                    
                    elif cmd == "remove":
                        if len(parts) < 2:
                            print(Fore.YELLOW + "Usage: remove <symbol>" + Style.RESET_ALL)
                            continue
                        
                        sym = parts[1].upper()
                        
                        if sym in watchlist:
                            watchlist.remove(sym)
                            print(Fore.GREEN + f"✓ Removed {sym} from watchlist." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"{sym} is not in your watchlist." + Style.RESET_ALL)
                    
                    else:
                        print(Fore.YELLOW + "Unknown command. Use 'add', 'remove', or 'back'." + Style.RESET_ALL)
            elif choice in ["4", "research", "r"]:
                research_scene(balance, portfolio, watchlist)
            else:
                print(Fore.YELLOW + "Unknown command. Type 'help' for options." + Style.RESET_ALL)
    except KeyboardInterrupt:
        print("\n" + Fore.CYAN + "➤ Interrupted. Exiting..." + Style.RESET_ALL)
    finally:
        save_data()

if __name__ == "__main__":
    main()