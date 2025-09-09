# Virtual Brokerage Account Manager (Python Terminal)

Python terminal simulator for a brokerage account with real-time market data, ML-backed research, and a retro green TUI. Built to showcase end-to-end software engineering skills for data-driven systems.

# Overview

- Real market data via Yahoo Finance using the yfinance Python package

- Stateful portfolio accounting with average cost tracking and live valuation

- Watchlist with current price and daily percent change

- Research scene that plots one year of - prices in the terminal and predicts next-day UP or DOWN with a short rationale

- Green monochrome terminal theme using Colorama

- JSON persistence so sessions are saved between runs

# Tech stack

- Python 3.10+

- yfinance for quotes and historical data

- pandas and numpy for data handling

- scikit-learn DecisionTreeClassifier for classification

- plotext for ASCII charts in the terminal

- colorama for green monochrome styling on Windows, macOS, and Linux

- JSON for lightweight persistence

# What this project demonstrates

- Data ingestion from a public finance API and clean separation of concerns

- Terminal UI scenes with commands for account, portfolio, watchlist, and research

- Persisted application state and reproducible behavior

- A small ML pipeline that trains per symbol on recent OHLCV features

- Cross platform CLI UX that runs well inside Visual Studio Code and standard terminals

# Project structure
virtual-brokerage/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  main.py
  stock_holding.py
  stock_predictor.py
  broker_data.json        # created on first run

- main.py routes scenes, handles I O, fetches data, and persists state

- stock_holding.py implements share mutations and average cost

- stock_predictor.py trains a decision tree on OHLCV features and returns a label with a short rationale

# Installation and first run

Clone or download the repository, then open a terminal in the project root. Use a virtual environment to isolate dependencies.

# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py

If activation is blocked, allow scripts for this user, then activate again:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
# macOS or Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py

Deactivate the environment when finished:

deactivate
# requirements.txt

If your checkout does not include it, create a file named requirements.txt with the lines below, then install with pip install -r requirements.txt.

yfinance
pandas
numpy
scikit-learn
plotext
colorama
# Usage

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

- Enter a ticker to view the current price, a one year ASCII chart, and the ML prediction

- Type back to return

# Data pipeline and modeling

- Market data is retrieved from Yahoo Finance through yfinance ticker objects and history calls

- The research scene loads about one year of daily bars per symbol

- Features per day: open, close, volume, intraday percent change

- The model is a scikit-learn DecisionTreeClassifier trained on recent history to classify next-day UP or DOWN

- A brief rationale is derived from recent trends to aid interpretability

# Terminal charts and theme

- plotext renders a line chart of daily closes directly in the terminal

- colorama ensures green text renders correctly on Windows, macOS, and Linux

# Persistence

- broker_data.json stores cash balance, portfolio holdings with average price, and watchlist symbols

- Human readable and easy to version

- Can be swapped for SQLite if needed

# Notes on accuracy and scope

- This is a simulator and does not place real trades

- Data is provided by Yahoo Finance via yfinance and is intended for research and education

- Availability and latency depend on the upstream service

# SEOs

- Python terminal application for finance

- Real time market data with yfinance

- CLI portfolio manager and watchlist

- ASCII terminal charts with plotext

- scikit-learn decision tree classifier

- Data ingestion and ML inference in a single app

- Systems engineering and data pipelines

- Aerospace and defense software engineering portfolio

# License

- MIT.