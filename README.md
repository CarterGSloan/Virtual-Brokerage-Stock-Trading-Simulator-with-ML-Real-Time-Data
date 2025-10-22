# Virtual Brokerage Account Manager (Python Terminal)

Python terminal application that simulates a brokerage account with watchlists, live pricing, and a machine learning (ML) research view. Fetches market data from Yahoo Finance (via 'yfinance'), trains a calibrated LightGBM model per symbol on recent OHLCV-derived features, and log predictions for later evaluation.

---

## Overview

- **Live data** via 'yfinance'. (Yahoo Finance).
- **Portfolio**: buys/sells, average cost, live valuation.
- **Research view**: 1 year ASCII chart + **next-day UP/DOWN** prediction with probability & confidence.
- **ML pipeline**: LightGBM + expanding TimeSeriesSplit + **isotonic calibration** (OOF-based).
- **Rich metrics**: AUC, log-loss, accuracy, precision/recall/F1, win-rate, profit factor (OOF).
- **Feature flags** via 'FeatureConfig' (MACD histogram, BB position, ROC(10), volatility regime).
- **Prediction logging** to logs/predictions.csv' (auto backfills realized labels when data arrives).
- **Auth**: simple username/password stored in 'users.json' with per-user data file.
- **Green retro TUI** with Colorama; optional full-width terminal charts via 'plotext'.

---

## Tech stack

- Python 3.10+
- **Data**: 'yfinance', 'pandas', 'numpy'
- **ML**: 'lightgbm', 'scikit-learn'
- **TUI**: 'colorama', 'plotext' (optional)
- **Persistence**: JSON files ('users.json', per-user 'user_<name>_data.json'), CSV logs

---

## Project structure
virtual-brokerage/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  main.py
  stock_holding.py
  stock_predictor.py
  tui_utils.py
  user.json # created on first run
  logs/
  predictions.csv # created on first prediction

---

- **'main.py'**: app shell, scenes (account/portfolio/watchlist/research), auth, prediction logging.
- **'stock_holding.py'**: holding model (avg price, share mutations), safe sell semantics.
- **'stock_predictor.py'**: feature engineering + LightGBM model, OOF metrics, calibration, features flags.
- **'tui_utils.py'**: terminal helpers (centering, headers, width).

---

## Installation and first run

Clone or download the repository, then open a terminal in the project root. Use a virtual environment to isolate dependencies.

### Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py

If activation is blocked, allow scripts for this user, then activate again:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

### macOS or Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py

Deactivate the environment when finished:

deactivate

---

## requirements.txt

If your checkout does not include it, create a file named requirements.txt with the lines below, then install with pip install -r requirements.txt.

yfinance
pandas
numpy
scikit-learn
plotext
colorama

---

## Usage

Start the app and log in with user and pass. In each scene, type commands as shown, or type back to return.

Main menu scenes: account, portfolio, watchlist, research, help, exit

Portfolio scene

buy TICKER QUANTITY
sell TICKER QUANTITY
back

Watchlist scene

add TICKER
remove TICKER
back

Research scene

- Enter a ticker to view the current price, a one year ASCII chart, and the ML prediction (UP/DOWN), probability, confidence, calibrated AUC context

- Type 'toggle' to switch prediction target timing between CLOSE and OPEN

- Type back to return

---

##  Data pipeline and modeling

### Data
- Daily OHLCV fetched via yfinance.Ticker(symbol).history(...).
- Trading day alignment across stock/market/sector series to avoid gaps.
- Prefer Adj Close when available (split/dividend aware).

### Features (stationary, lag-safe)
- returns: s_ret, intraday % (s_intra), overnight % (s_overnight)
- moving average differential: s_ma_diff (fast vs slow)
- volatility: s_vol_5, s_vol_20, ATR(14)
- momentum: RSI(14)
- context: market returns & MA diff, sector returns & MS diff, relative strength, volume z-score

---

## Training
- Expanding TimeSeriesSplit (default: 4 folds)
- Model: LightGBM binary classifier with early stopping.
- Isotonic calibration fitted on stacked out-of-fold (OOF) probabilities to produce well-calibrated results.

---

## Metrics
- **OOF (global)**: AUC, log-loss, accuracy, precision, recall, F1, win-rate, profit factor.
- **Last fold (calibrated)**: same set for the final validation slice.
- **Rolling AUC** reported at predict time for quick context.

---

## Prediction Logging
- Each prediction appends a row to logs/predictions.csv, then tried to backfill realized labels and PnL when the target date's bar becomes available. Columns include: ts_utc, symbol, timing, today_date, target_date, prob_up, label_pred, decision, today_close, realized_label, realized_close, pnl_bps, market_close, market_ret

---

## Authentication & Persistence
- users.json stores usesrs and references per-user data file user_<username>_data.json.
- Per-user data keeps: cash balance, portfolio (symbol, quantity, average price), watchlist tickers
- Logs live in logs/predictions.csv
- **Passwords are stores in plain JSON for demo purposed - DO NOT reuse real credentials.**

---

## Notes on accuracy and scope

- This is a simulator for education and portfolios; does not place real trades
- Data availability/latency depends on Yahoo Finance
- Models aims to demonstrate a disciplined time-series ML pipeline, not a garunteed edge.

---

## Troubleshooting
- **Windows plotting**: If full chart fails, you still get sparklines. Install/verify plotext.
- **Pylance**: datetime.utcnow depracated: Code uses datetime.now(timezone.utc) and writes z ISO strings.
- **Import errors**: pip install -r requirements.txt inside the activated venv.
- **LightGBM build issues**: Use prebuilt wheels (the listed minimum version typically ships wheels on major platforms).

---

## Roadmap
- Walk-forward backtest command (export PnL curve)
- Permutation importance report per symbol
- Configurable sector mapping (or sector inference cache)
- SQLite backend for multi-user, multi-portfolio scenarios

---

## License

- MIT.