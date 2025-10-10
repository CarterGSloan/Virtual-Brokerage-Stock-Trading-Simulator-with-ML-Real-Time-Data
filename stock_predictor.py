from __future__ import annotations
import math
import numpy as np 
import pandas as pd 
from dataclasses import dataclass 
from typing import Optional, Tuple, Dict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
import lightgbm as lgb 
import warnings

# Utility & Feature operations

def _check_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing: 
        raise ValueError(f"Missing columns: {missing}")

def _align_trading_days(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Inner-join on index to drop non-overlapping and non-trading days.
    Assumes index is DateTimeIndex in ascending order.
    """
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    return tuple(d.loc[idx].sort_index() for d in dfs)

def _adf_close(df: pd.DataFrame) -> pd.Series:
    # Prefer Adj Close to adjust for splits & dividends. Fallback to Close. 
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

def _lag(s: pd.Series, k: int = 1) -> pd.Series:
    return s.shift(k)

def _true_range(df: pd.DataFrame) -> pd.Series:
    # Wilder's True Range
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    # Simple RSI implementation with Wilder smoothing
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up = pd.Series(up, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    down = pd.Series(down, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def build_features(
    stock: pd.DataFrame,
    market: pd.DataFrame,
    sector: Optional[pd.DataFrame] = None,
    target_timing: str = "close",   # "close" means predict next close using features known at today's close
    fast_ma: int = 5,
    slow_ma: int = 20
) -> pd.DataFrame:
    """
    All features are computed from lagged data only to avoid look-ahead.
    Required columns in each df: Open, High, Low, Close, Volume, optional Adj Close
    Index must be trading dates ascending.
    """
    _check_cols(stock, ["Open", "High", "Low", "Close", "Volume"])
    _check_cols(market, ["Open", "High", "Low", "Close", "Volume"])
    if sector is not None:
        _check_cols(sector, ["Open", "High", "Low", "Close", "Volume"])

    # Align by common trading days
    if sector is not None:
        stock, market, sector = _align_trading_days(stock, market, sector)
    else:
        stock, market = _align_trading_days(stock, market)

    s_close_adj = _adj_close(stock)
    m_close_adj = _adj_close(market)
    q_close_adj = _adj_close(sector) if sector is not None else None

    # Core stationary returns
    s_ret = s_close_adj.pct_change()
    s_intra = (stock["Close"] - stock["Open"]) / stock["Open"]
    s_overnight = (_lag(stock["Open"]) - _lag(stock["Close"])) / _lag(stock["Close"])

    # Rolling stats
    s_ma_fast = s_close_adj.rolling(fast_ma).mean()
    s_ma_slow = s_close_adj.rolling(slow_ma).mean()
    s_ma_diff = (s_ma_fast - s_ma_slow) / s_ma_slow
    s_vol_5 = s_ret.rolling(5).std()
    s_vol_20 = s_ret.rolling(20).std()
    s_tr = _true_range(stock)
    s_atr_14 = s_tr.rolling(14).mean()
    s_rsi_14 = _rsi(s_close_adj, 14)

    # Market and sector context
    m_ret = m_close_adj.pct_change()
    m_ma_fast = m_close_adj.rolling(fast_ma).mean()
    m_ma_slow = m_close_adj.rolling(slow_ma).mean()
    m_ma_diff = (m_ma_fast - m_ma_slow) / m_ma_slow
    rel_strength = s_ret - m_ret

    if sector is not None:
        q_ret = q_close_adj.pct_change()
        q_ma_fast = q_close_adj.rolling(fast_ma).mean()
        q_ma_slow = q_close_adj.rolling(slow_ma).mean()
        q_ma_diff = (q_ma_fast - q_ma_slow) / q_ma_slow
    else:
        q_ret = pd.Series(0.0, index=stock.index)
        q_ma_diff = pd.Series(0.0, index=stock.index)

    # Volume context
    vol_mean_20 = stock["Volume"].rolling(20).mean()
    vol_std_20 = stock["Volume"].rolling(20).std()
    vol_z = (stock["Volume"] - vol_mean_20) / vol_std_20

    # Assemble DataFrame
    df = pd.DataFrame({
        "s_ret": s_ret,
        "s_intra": s_intra,
        "s_overnight": s_overnight,
        "s_ma_diff": s_ma_diff,
        "s_vol_5": s_vol_5,
        "s_vol_20": s_vol_20,
        "s_atr_14": s_atr_14,
        "s_rsi_14": s_rsi_14,
        "m_ret": m_ret,
        "m_ma_diff": m_ma_diff,
        "q_ret": q_ret,
        "q_ma_diff": q_ma_diff,
        "rel_strength": rel_strength,
        "vol_z": vol_z,
    }, index=stock.index)

    # Define label based on target timing
    # Predict "tomorrow close > today close"
    if target_timing == "close":
        # Features must be known at today's close, so use lagged features X_t = f(data up to t)
        # Label uses next close
        future_close = s_close_adj.shift(-1)
        df["label"] = (future_close > s_close_adj).astype(int)
        # Lag all features by 0 because they already use only up-to-t info
    elif target_timing == "open":
        # Pre-open signal. Only use information known by yesterday close.
        # Shift features by 1 day so the model uses X_{t-1} to predict y_t (today up from yesterday close)
        future_close = s_close_adj        # today close
        df["label"] = (future_close > _lag(s_close_adj)).astype(int)
        df = df.shift(1)
    else:
        raise ValueError("target_timing must be 'open' or 'close'")

    # Drop rows with NaNs from rolling and shifting and remove the last label if it points to future not present
    df = df.dropna().copy()
    return df

@dataclass
class PredictResult:
    label: str
    prob_up: float
    confidence: float
    decision: str
    metrics: Dict[str, float]
    reason: str

class StockPredictor:
    """
    LightGBM classifier with expanding-window TimeSeriesSplit, early stopping, and isotonic calibration.
    Trains on last N days (rolling window) and predicts next-day direction. All features are stationary.
    """

    def __init__(
        self,
        target_timing: str = "close",    # "close" or "open"
        window_days: int = 756,          # about 3 years of trading days
        n_splits: int = 4,
        confidence_threshold: float = 0.55,
        use_scaler_in_calibrator: bool = True,
        random_state: int = 42
    ):
        self.target_timing = target_timing
        self.window_days = window_days
        self.n_splits = n_splits
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        # Model and calibrator
        self.lgb_params = dict(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=2000,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.scaler = StandardScaler() if use_scaler_in_calibrator else None
        self.features_: Optional[list] = None

    def _fit_calibrator(self, x_val: np.ndarray, p_val: np.ndarray, y_val: np.ndarray):
        """
        Isotonic regression on validation probabilities. Optional standardization.
        """
        if self.scaler is not None:
            x_fit = self.scaler.fit_transform(p_val.reshape(-1, 1))
        else:
            x_fit = p_val.reshape(-1, 1)
        # Isotonic expects 1d x. Use probability itself as x
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(x_fit.ravel(), y_val.astype(int))

    def _apply_calibrator(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p
        x = self.scaler.transform(p.reshape(-1, 1)) if self.scaler is not None else p.reshape(-1, 1)
        return self.calibrator.predict(x.ravel())

    def fit(
        self,
        stock_df: pd.DataFrame,
        market_df: pd.DataFrame,
        sector_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train with expanding TimeSeriesSplit and early stopping on the last fold.
        Returns validation metrics for the last fold.
        """
        df = build_features(stock_df, market_df, sector_df, target_timing=self.target_timing)
        if len(df) < max(200, self.window_days // 2):
            raise ValueError("Not enough rows after feature prep. Provide at least ~1 year of data.")

        # Rolling window
        df = df.iloc[-self.window_days:].copy() if len(df) > self.window_days else df
        y = df["label"].values.astype(int)
        X = df.drop(columns=["label"])
        self.features_ = list(X.columns)

        # Expanding TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        metrics = {}
        last_fold = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**self.lgb_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                )
            # Save last fold model and calibrator
            p_val = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
            metrics[f"fold_{fold}_auc"] = float(auc)
            last_fold = (model, X_val, y_val, p_val)

        # Use last fold for final model and calibration
        self.model, X_val, y_val, p_val = last_fold
        self._fit_calibrator(X_val.values, p_val, y_val)
        metrics["last_fold_auc"] = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else np.nan
        metrics["best_iteration"] = int(getattr(self.model, "best_iteration_", self.lgb_params["n_estimators"]))
        return metrics

    def predict_next(self, stock_df: pd.DataFrame, market_df: pd.DataFrame, sector_df: Optional[pd.DataFrame] = None) -> PredictResult:
        """
        Produce next-day direction with calibrated probability and a simple decision based on threshold.
        """
        if self.model is None or self.features_ is None:
            raise RuntimeError("Model not fit yet. Call fit(...) first.")

        df = build_features(stock_df, market_df, sector_df, target_timing=self.target_timing)
        df = df.iloc[-self.window_days:].copy() if len(df) > self.window_days else df

        X = df.drop(columns=["label"])
        y = df["label"].values.astype(int)

        # Use the last row as "today" features to predict "tomorrow"
        x_last = X.iloc[[-1]]
        raw_p = self.model.predict_proba(x_last)[:, 1]
        p_up = float(self._apply_calibrator(raw_p)[0])

        label = "UP" if p_up >= 0.5 else "DOWN"
        confidence = max(p_up, 1 - p_up)

        decision = "NO-TRADE"
        if confidence >= self.confidence_threshold:
            decision = "LONG" if label == "UP" else "SHORT"

        # Quick rolling validation metric for context
        p_val = self.model.predict_proba(X)[:, 1]
        auc_val = roc_auc_score(y, p_val) if len(np.unique(y)) > 1 else np.nan

        reason = f"Model prob_up={p_up:.2f}, threshold={self.confidence_threshold:.2f}, decision={decision}."

        return PredictResult(
            label=label,
            prob_up=p_up,
            confidence=confidence,
            decision=decision,
            metrics={"rolling_auc": float(auc_val)},
            reason=reason,
        )

    def retrain_daily(self, stock_df: pd.DataFrame, market_df: pd.DataFrame, sector_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Call this after market close with updated data. It refits on the last window_days.
        """
        return self.fit(stock_df, market_df, sector_df)