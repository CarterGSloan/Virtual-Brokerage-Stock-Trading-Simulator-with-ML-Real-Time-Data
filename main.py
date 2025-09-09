import json, os, shutil, time
from datetime import datetime
import yfinance as yf
from stock_holding import StockHolding
from stock_predictor import StockPredictor
from tui_utils import print_header
try:
    import plotext as plt
except ImportError:
    plt = None
from colorama import init, Fore, Style

def sparkline(values, width=42):
    """Return a one-line Unicode sparkline for the sequence of floats.
    Uses 8 levels:▁▂▃▄▅▆▇█ (works well in most terminals).
    """
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return blocks[0] * min(width, len(values))
    #Resample to desired width if needed
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
    try:
        t = yf.Ticker(symbol)
        h = t.history(period=period, interval=interval)
        #drop rows with missing close
        if h is not None and not h.empty:
            h = h.dropna(subset=["Close"])   #Ensure plotting works
        return h
    except Exception:
        return None
    
def market_index_rows():
    """Return tuples: (label, symbol, last_price, change_pct, spark) for three indices."""
    entries = []
    indices = [
        ("S&P 500", "^GSPC"), 
        ("NASDAQ", "^IXIC"), 
        ("NYSE", "^NYA"),
    ]
    for name, sym in indices:
        hist = fetch_history(sym, period="3mo", interval="1d")
        last = None
        change_pct = None
        sline = ""
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
    #Left pane header
    print_header(Fore.GREEN + Style.BRIGHT + "\nResearch - Market Overview" + Style.RESET_ALL)

    rows = market_index_rows()
    #layout: name/symbol/last/change left, sparkline right on the same line
    #compute pad so lines align in two columns
    left_w = 36
    for name, sym, last, chg, sline in rows:
        last_str = f"${last:.2f}" if last is not None else "N/A"
        chg_str = f"{chg:+.2f}%" if chg is not None else ""
        left = f"{name} ({sym}) {last_str:>12} {chg_str:>8}"
        left = left[:left_w].ljust(left_w)
        print(Fore.GREEN + left + " " + sline + Style.RESET_ALL)
    print(Fore.GREEN + "\nType a ticker to view its chart and prediction or 'back' to return." + Style.RESET_ALL)

def show_symbol_chart(sym, hist):
    #robust chart rendering: try plotext; always render a sparkline feedback
    closes = [float(x) for x in hist["Close"].values.tolist()] if (hist is not None and not hist.empty) else []
    #fallback sparkline(always) - printed first so user sees something even if plotext fails
    if closes:
        sline = sparkline(closes, width=60)
        print(Fore.GREEN + f"\n{sym} 1Y Sparkline: " + sline + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"\nNo close data available to render sparkline for {sym}." + Style.RESET_ALL)
    
    if plt and hist is not None and not hist.empty:
        try:
            #plotext sizing based on terminal
            cols, rows = shutil.get_terminal_size(fallback=(100, 30))
            #leave a little room for labels
            plot_w = max(60, min(cols - 4, 140))
            plot_h = max(12, min(rows - 8, 32))
            plt.clf()
            plt.plotsize(plot_w, plot_h)
            dates = [d.strftime("%Y-%m-%d") for d in hist.index]
            prices = closes
            plt.title(f"{sym} Price - Last 1Y")
            plt.plot(dates, prices, marker="")
            #reduce x ticks clutter
            if len(dates) > 12:
                step = len(dates) // 12
                xticks = [dates[i] for i in range(0, len(dates), step)]
                plt.xticks(xticks)
            plt.show()
        except Exception:
            print(Fore.GREEN + "(Chart rendering skipped due to a terminal or plotting error.)" + Style.RESET_ALL)
    else:
        if plt is None:
            print(Fore.GREEN + "(Install 'plotext' to enable full-size terminal charts.)" + Style.RESET_ALL)

def research_scene(balance, portfolio, watchlist):
    while True:
        show_research_dashboard()
        sym = input(Fore.GREEN + "> symbol: " + Style.RESET_ALL).strip()
        if not sym:
            continue
        if sym.lower() in ("back", "b"):
            break
        sym = sym.upper()
        hist = fetch_history(sym, period="1y", interval="1d")
        if hist is None or hist.empty:
            print(Fore.GREEN + f"Could not retieve data for {sym}." + Style.RESET_ALL)
            continue
        #current price
        last_price = float(hist["Close"].iloc[-1])
        print(Fore.GREEN + f"Current Price of {sym}: ${last_price:,.2f}" + Style.RESET_ALL)
        #show chart (sparkline + plotext if avaliable)
        show_symbol_chart(sym, hist)
        #ML prediction
        try:
            predictor = StockPredictor(hist)
            pred, reason = predictor.predict_next_day()
            print(Fore.GREEN + f"Predicted Next Day Move: {pred}" + Style.RESET_ALL)
            print(Fore.GREEN + f"Reasoning: {reason}" + Style.RESET_ALL)
        except Exception:
            print(Fore.GREEN + "ML Prediction unavailable (insufficient data or error)." + Style.RESET_ALL)
        input(Fore.GREEN + "Press Enter to continue..." + Style.RESET_ALL)

def main():
    #Initialize colorama for colored text ouput (green)
    init(autoreset=True)
    #welcome message
    print_header(Fore.GREEN + Style.BRIGHT + "WELCOME TO YOUR VIRTUAL BROKERAGE ACCOUNT MANAGER" + Style.RESET_ALL)
    print(Fore.GREEN + "Please login to continue." + Style.RESET_ALL)
    #Login prompt
    username = input(Fore.GREEN + "Username: " + Style.RESET_ALL)
    password = input(Fore.GREEN + "Password: " + Style.RESET_ALL)
    while not (username == "user" and password == "pass"):
        print(Fore.GREEN + "Invalid credentials. Please try again." + Style.RESET_ALL)
        username = input(Fore.GREEN + "Username: " + Style.RESET_ALL)
        password = input(Fore.GREEN + "Password: " + Style.RESET_ALL)
    print(Fore.GREEN + "Login successful!" + Style.RESET_ALL)
    #Load persistent data if avaliable
    data_file = "broker_data.json"
    balance = 100000.0
    portfolio = []
    watchlist = []
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                saved = json.load(f)
            balance = saved.get("balance", balance)
            #Reconstruct portfolio holdings
            for item in saved.get("portfolio", []):
                sym = item["symbol"]; qty = item["quantity"]; avg = item.get("avg_price", 0.0)
                portfolio.append(StockHolding(sym, qty, avg))
            watchlist = saved.get("watchlist", watchlist)
        except Exception as e:
            #In case of error, use defaults
            balance = 100000.0
            portfolio = []
            watchlist = []
    def save_data():
        """Save current balance, portfolio, and watchlist to file."""
        data = {
            "balance": balance, 
            "portfolio": [
                {"symbol":sh.symbol, "quantity": sh.quantity, "avg_price": sh.avg_price} for sh in portfolio
            ],
            "watchlist": watchlist
        }
        try:
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(Fore.GREEN + f"(Warning: could not save data: {e})" + Style.RESET_ALL)
    def get_current_price(symbol):
        """Helper to get the latest price for a stock ticker (returns None if not found)."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            price = hist['Close'].iloc[-1]
            return float(price) if price is not None else None
        except Exception:
            return None
    
    #Main menu loop
    try:
        while True:
            #Display menu
            print_header(Fore.GREEN + Style.BRIGHT + "\nMain Menu:" + Style.RESET_ALL)
            print(Fore.GREEN + "1. Account - View account balance and info" + Style.RESET_ALL)
            print(Fore.GREEN + "2. Portfolio - View or trade your holdings" + Style.RESET_ALL)
            print(Fore.GREEN + "3. Watchlist - Manage watchlist" + Style.RESET_ALL)
            print(Fore.GREEN + "4. Research - Index overview, charts, ML" + Style.RESET_ALL)
            print(Fore.GREEN + "5. Help - Show avaliable commands" + Style.RESET_ALL)
            print(Fore.GREEN + "6. Exit" + Style.RESET_ALL)
            choice = input(Fore.GREEN + "Enter a command or number: " + Style.RESET_ALL).strip().lower()
            if choice in ["6", "exit", "quit"]:
                print(Fore.GREEN + "Saving data and exiting... Goodbye!" + Style.RESET_ALL)
                break
            elif choice in ["5", "help", "h"]:
                print(Fore.GREEN + "Avaliable commands: account, portfolio, watchlist, research, exit" + Style.RESET_ALL)
                continue
            elif choice in ["1", "account", "a"]:
                #Account info
                print(Fore.GREEN + Style.BRIGHT + "\nAccount Information:" + Style.RESET_ALL)
                #Calculate current portfolio market value by fetching latest prices
                total_portfolio_val = 0.0
                for sh in portfolio:
                    price = get_current_price(sh.symbol)
                    if price is not None:
                        total_portfolio_val += price * sh.quantity
                print(Fore.GREEN + f"Cash Balance: ${balance:,.2f}" + Style.RESET_ALL)
                print(Fore.GREEN + f"Portfolio Market Value: ${total_portfolio_val:.2f}" + Style.RESET_ALL)
                print(Fore.GREEN + f"Total Net Value: ${ (balance + total_portfolio_val):.2f}" + Style.RESET_ALL)
                input(Fore.GREEN + "Press Enter to return to main menu..." + Style.RESET_ALL)
            elif choice in ["2", "portfolio", "p"]:
                #Portfolio management
                while True:
                    print(Fore.GREEN + Style.BRIGHT + "\nYour Portfolio Holdings:" + Style.RESET_ALL)
                    if portfolio:
                        # Table header
                        print(Fore.GREEN + f"{'Symbol':<10}{'Quantity':>10}{'Avg Cost':>12}{'Curr Price':>12}{'Value':>15}" + Style.RESET_ALL)
                        total_val = 0.0
                        for sh in portfolio:
                            price = get_current_price(sh.symbol)
                            curr_price = price if price is not None else sh.avg_price
                            value = curr_price * sh.quantity
                            total_val += value
                            print(Fore.GREEN + f"{sh.symbol:<10}{sh.quantity:>10}{sh.avg_price:>12.2f}{curr_price:>12.2f}{value:>15.2f}" + Style.RESET_ALL)
                        print(Fore.GREEN + f"\nTotal Portfolio Value: ${total_val:,.2f}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + "(Portfolio is empty)" + Style.RESET_ALL)
                    action = input(Fore.GREEN + "Enter command [buy <SYM> <QTY> / sell <SYM> <QTY> / back]: " + Style.RESET_ALL).strip().lower()
                    if action in ["", None]:
                        continue
                    if action == "back" or action == "b":
                        break
                    parts = action.split()
                    cmd = parts[0] if parts else ""
                    if cmd == "buy":
                        if len(parts) < 3:
                            print(Fore.GREEN + "Usage: buy <symbol> <quantity>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        qty_str = parts[2]
                        try:
                            qty = int(qty_str)
                        except ValueError:
                            print(Fore.GREEN + "Invalid quantity." + Style.RESET_ALL); continue
                        if qty <= 0:
                            print(Fore.GREEN + "Quantity must be positive." + Style.RESET_ALL); continue
                        price = get_current_price(sym)
                        if price is None:
                            print(Fore.GREEN + f"Symbol {sym} not found." + Style.RESET_ALL); continue
                        cost = price * qty
                        if cost > balance:
                            print(Fore.GREEN + "Insufficient funds to buy." + Style.RESET_ALL)
                        else:
                            balance -= cost
                            # Add to portfolio (or update existing holding)
                            for sh in portfolio:
                                if sh.symbol == sym:
                                    sh.add_shares(qty, price); break
                            else:
                                portfolio.append(StockHolding(sym, qty, price))
                            print(Fore.GREEN + f"Bought {qty} shares of {sym} at ${price:.2f} each, total cost ${cost:.2f}." + Style.RESET_ALL)
                    elif cmd == "sell":
                        if len(parts) < 3:
                            print(Fore.GREEN + "Usage: sell <symbol> <quantity>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        qty_str = parts[2]
                        try:
                            qty = int(qty_str)
                        except ValueError:
                            print(Fore.GREEN + "Invalid quantity." + Style.RESET_ALL); continue
                        if qty <= 0:
                            print(Fore.GREEN + "Quantity must be positive." + Style.RESET_ALL); continue
                        found = False
                        for sh in portfolio:
                            if sh.symbol == sym:
                                found = True
                                if qty > sh.quantity:
                                    print(Fore.GREEN + "Not enough shares to sell." + Style.RESET_ALL)
                                else:
                                    price = get_current_price(sym)
                                    if price is None:
                                        print(Fore.GREEN + f"Could not retrieve price for {sym}. Sell aborted." + Style.RESET_ALL)
                                    else:
                                        revenue = price * qty
                                        balance += revenue
                                        all_sold = sh.remove_shares(qty)
                                        if all_sold:
                                            portfolio.remove(sh)
                                        print(Fore.GREEN + f"Sold {qty} shares of {sym} at ${price:.2f} each, total revenue ${revenue:.2f}." + Style.RESET_ALL)
                                break
                        if not found:
                            print(Fore.GREEN + f"You do not own any shares of {sym}." + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + "Unknown command. Use 'buy', 'sell', or 'back'." + Style.RESET_ALL)
            elif choice in ["3", "watchlist", "w"]:
                # Watchlist management
                while True:
                    print(Fore.GREEN + Style.BRIGHT + "\nYour Watchlist:" + Style.RESET_ALL)
                    if watchlist:
                        print(Fore.GREEN + f"{'Symbol':<10}{'Price':>10}{'Change%':>12}" + Style.RESET_ALL)
                        for sym in watchlist:
                            price = get_current_price(sym)
                            change_str = ""
                            if price is not None:
                                # Fetch previous close for change%
                                try:
                                    ticker = yf.Ticker(sym)
                                    hist2 = ticker.history(period="2d")
                                except Exception:
                                    hist2 = None
                                if hist2 is not None and len(hist2) >= 2:
                                    prev = hist2['Close'].iloc[-2]
                                    last = hist2['Close'].iloc[-1]
                                    if prev and last:
                                        change_pct = (last - prev) / prev * 100
                                        change_str = f"{change_pct:+.2f}%"
                            price_str = f"${price:.2f}" if price is not None else "N/A"
                            print(Fore.GREEN + f"{sym:<10}{price_str:>10}{change_str:>12}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + "(Watchlist is empty)" + Style.RESET_ALL)
                    action = input(Fore.GREEN + "Enter command [add <SYM> / remove <SYM> / back]: " + Style.RESET_ALL).strip().lower()
                    if action in ["", None]:
                        continue
                    if action == "back" or action == "b":
                        break
                    parts = action.split()
                    cmd = parts[0] if parts else ""
                    if cmd == "add":
                        if len(parts) < 2:
                            print(Fore.GREEN + "Usage: add <symbol>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        if sym == "":
                            print(Fore.GREEN + "Please provide a symbol to add." + Style.RESET_ALL); continue
                        if sym in watchlist:
                            print(Fore.GREEN + f"{sym} is already in your watchlist." + Style.RESET_ALL)
                        else:
                            price = get_current_price(sym)
                            if price is None:
                                print(Fore.GREEN + f"Symbol {sym} not found." + Style.RESET_ALL)
                            else:
                                watchlist.append(sym)
                                print(Fore.GREEN + f"Added {sym} to watchlist." + Style.RESET_ALL)
                    elif cmd == "remove":
                        if len(parts) < 2:
                            print(Fore.GREEN + "Usage: remove <symbol>" + Style.RESET_ALL); continue
                        sym = parts[1].upper()
                        if sym in watchlist:
                            watchlist.remove(sym)
                            print(Fore.GREEN + f"Removed {sym} from watchlist." + Style.RESET_ALL)
                        else:
                            print(Fore.GREEN + f"{sym} is not in your watchlist." + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + "Unknown command. Use 'add', 'remove', or 'back'." + Style.RESET_ALL)
            elif choice in ["4", "research", "r"]:
                # Research and prediction
                research_scene(balance, portfolio, watchlist)

            else:
                print(Fore.GREEN + "Unknown command. Type 'help' for options." + Style.RESET_ALL)
    except KeyboardInterrupt:
        print("\n" + Fore.GREEN + "Interrupted. Exiting..." + Style.RESET_ALL)
    finally:
        save_data()

if __name__ == "__main__":
    main()