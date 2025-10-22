from __future__ import annotations
import logging
import math
import numpy as np 
import pandas as pd 
from dataclasses import dataclass 
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
import lightgbm as lgb 
import warnings

# Utility & Feature operations

def _check_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing: 
        raise ValueError(f"Missing columns: {missing}")

def _align_trading_days(*dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Inner-join on index to drop non-overlapping and non-trading days.
    Assumes index is DateTimeIndex in ascending order.
    """
    if not dfs:
        return tuple()
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    return tuple(d.loc[idx].sort_index() for d in dfs)

def _adj_close(df: pd.DataFrame) -> pd.Series:
    """Get adjusted close price, falling back to close if unavaliable."""
    return df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

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
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    down = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    up_ema = up.ewm(alpha=1 / window, adjust=False).mean()
    down_ema = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = _safe_divide(up_ema, down_ema.replace(0, np.nan), fill_value=np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

@dataclass
class FeatureConfig:
    fast_ma: int = 5
    slow_ma: int = 20
    rsi_window: int = 14
    atr_window: int = 14
    vol_short: int = 5
    vol_long: int = 20
    volume_z_window: int = 20
    add_advanced: bool = True # toggle extra indicators
    macd_hist: bool = True
    bb_position: bool = True
    roc10: bool = True
    vol_regime: bool = True

@dataclass
class ModelConfig:
    num_leaves: int = 31
    learning_rate: float = 0.03
    n_estimators: int = 2000
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 200
    random_state: int = 42
    n_jobs: int = -1




def build_features(
    stock: pd.DataFrame,
    market: pd.DataFrame,
    sector: Optional[pd.DataFrame] = None,
    target_timing: str = "close",   # "close" means predict next close using features known at today's close
    feat_cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Build predictive features from stock, marekt and optional sector data. 

    All features are computed from lagged data to avoid look-ahead bias. 
    Features are designed to be stationary (returns, ratios, z-scores) for 
    better model performance and generalization across market regimes.

    Args:
        stock: OHLCV data for the target stock. Must have ascending DateTimeIndex.
        market: OHLCV data for market index (e.g., S&P 500).
        sector: Optional OHLCV data for sector index. 
        target_timing: when prediction is made:
            - "close": Features known at the current day's close, predict tomorrow's direction
            - "open": Features known at yesterday's close, predict today's direction
        fast_ma: Window for fast moving average (default: 5 days)
        slow_ma: Window for slow moving average (default: 20 days)
    
    Returns:
        DataFrame with features and binary label (1=up, 0=down).
        Index aligned to stock's trading days, NaN rows dropped.

    Features generated:
        - s_ret: Stock daily return
        - s_intra: Intraday return (close vs open)
        - s_overnight: Overnight gap (open vs previous close)
        - s_ma_diff: Fast MA vs slow MA divergence
        - s_vol_5, s_vol_20: Rolling volatility
        - s_atr_14: Average True Range (Wilder's method)
        - s_rsi_14: Relative Strength Index
        - m_ret: Market return
        - m_ma_diff: Market MA divergence
        - q_ret: Sector return
        - q_ma_diff: Sector MA divergence
        - rel_strength: Stock return minus market return
        - vol_z: Volume z-score

    Raise:
        ValueError: If required columns missing or insufficient data

    Example:
        >>> stock_data = yf.download("AAPL", start="2020-01-01")
        >>> market_data = yf.download("SPY", start="2020-01-01")
        >>> features = build_features(stock_data, market_data)
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
    s_intra = _safe_divide(stock["Close"] - stock["Open"], stock["Open"])
    s_overnight = _safe_divide(stock["Open"] - _lag(stock["Close"]), _lag(stock["Close"]))

    # Rolling stats
    s_ma_fast = s_close_adj.rolling(feat_cfg.fast_ma).mean()
    s_ma_slow = s_close_adj.rolling(feat_cfg.slow_ma).mean()
    s_ma_diff = _safe_divide(s_ma_fast - s_ma_slow, s_ma_slow)
    s_vol_5 = s_ret.rolling(feat_cfg.vol_short).std()
    s_vol_20 = s_ret.rolling(feat_cfg.vol_long).std()
    s_tr = _true_range(stock)
    s_atr_14 = s_tr.rolling(feat_cfg.atr_window).mean()
    s_rsi_14 = _rsi(s_close_adj, feat_cfg.rsi_window)

    # Market and sector context
    m_ret = m_close_adj.pct_change()
    m_ma_fast = m_close_adj.rolling(feat_cfg.fast_ma).mean()
    m_ma_slow = m_close_adj.rolling(feat_cfg.slow_ma).mean()
    m_ma_diff = _safe_divide(m_ma_fast - m_ma_slow, m_ma_slow)
    rel_strength = s_ret - m_ret

    if sector is not None:
        q_ret = q_close_adj.pct_change()
        q_ma_fast = q_close_adj.rolling(feat_cfg.fast_ma).mean()
        q_ma_slow = q_close_adj.rolling(feat_cfg.slow_ma).mean()
        q_ma_diff = _safe_divide(q_ma_fast - q_ma_slow, q_ma_slow)
    else:
        q_ret = pd.Series(0.0, index=stock.index)
        q_ma_diff = pd.Series(0.0, index=stock.index)

    # Volume context
    vol_mean_20 = stock["Volume"].rolling(feat_cfg.volume_z_window).mean()
    vol_std_20 = stock["Volume"].rolling(feat_cfg.volume_z_window).std()
    vol_z = _safe_divide(stock["Volume"] - vol_mean_20, vol_std_20.replace(0, np.nan), fill_value=0)

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

    # Implement add_features
    if any([feat_cfg.macd_hist, feat_cfg.bb_position, feat_cfg.roc10, feat_cfg.vol_regime]):
        df = _add_features(df, stock, s_close_adj, feat_cfg)

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

def _add_features(df: pd.DataFrame,
                  stock: pd.DataFrame,
                  s_close_adj: pd.Series,
                  feat_cfg: FeatureConfig,
) -> pd.DataFrame:
    """
    Adds 4 recommended, stationary, lag-safe indicators.
    Only minimal derived signals are added (avoid non-stationary raw levels).
    """
    if feat_cfg.macd_hist:
        ema_12 = s_close_adj.ewm(span=12, adjust=False).mean()
        ema_26 = s_close_adj.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = (macd - macd_signal)
    
    if feat_cfg.bb_position:
        sma_20 = s_close_adj.rolling(20).mean()
        std_20 = s_close_adj.rolling(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        pos = _safe_divide(s_close_adj - bb_lower, bb_upper - bb_lower, fill_value=0.5)
        df["bb_position"] = pos.clip(lower=0.0, upper=1.0) # Keep bounded and stationary
    
    if feat_cfg.roc10:
        df["roc_10"] = s_close_adj.pct_change(10)
    
    if feat_cfg.vol_regime:
        # Relies on s_vol_20 precomputed in base features
        if "s_vol_20" in df.columns:
            df["vol_regime"] = (df["s_vol_20"] > df["s_vol_20"].rolling(60).mean()).astype(int)
        else:
            # Falls back to price ret vol if missing
            s_ret = s_close_adj.pct_change()
            vol_20 = s_ret.rolling(20).std()
            df["vol_regime"] = (vol_20 > vol_20.rolling(60).mean()).astype(int)
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
        random_state: int = 42,
        feat_cfg: FeatureConfig = FeatureConfig(),
        model_cfg: ModelConfig = ModelConfig(),
        verbose: bool = True,
    ):
        self.target_timing = target_timing
        self.window_days = window_days
        self.n_splits = n_splits
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.feat_cfg = feat_cfg
        self.model_cfg = model_cfg

        # Model and calibrator
        self.lgb_params = dict(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=model_cfg.num_leaves,
            learning_rate=model_cfg.learning_rate,
            n_estimators=model_cfg.n_estimators,
            subsample=model_cfg.subsample,
            colsample_bytree=model_cfg.colsample_bytree,
            min_child_samples=model_cfg.min_child_samples,
            reg_alpha=model_cfg.reg_alpha,
            reg_lambda=model_cfg.reg_lambda,
            random_state=model_cfg.random_state,
            n_jobs=model_cfg.n_jobs,
        )
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.scaler = StandardScaler() if use_scaler_in_calibrator else None
        self.features_: Optional[List[str]] = None
        self._early_stopping_rounds = model_cfg.early_stopping_rounds
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
        if verbose:
            self.logger.setLevel(logging.INFO)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, name: str, required_cols: list, min_rows: int = 100) -> None:
        """Validate input DataFrame meets requirements."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must have DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be sorted in ascending order")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if len(df) < min_rows:
            raise ValueError(f"{name} has only {len(df)} rows, need at least {min_rows}")
        
        #Check for excessive NaNs
        nan_pct = df[required_cols].isna().sum() / len(df) 
        if (nan_pct > 0.1).any():
            warnings.warn(f"{name} has >10% NaN values in some coumns: {nan_pct[nan_pct > 0.1].to_dict()}")

    def _compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss as skl_log_loss
        out: Dict[str, float] = {}
        if len(np.unique(y_true)) > 1:
            try:
                out["auc"] = float(roc_auc_score(y_true, y_pred_proba))
            except Exception:
                out["auc"] = np.nan
            try:
                out["log_loss"] = float(skl_log_loss(y_true, y_pred_proba, labels=[0, 1]))
            except Exception:
                out["log_loss"] = np.nan
        else:
            out["auc"] = np.nan
            out["log_loss"] = np.nan
        
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        # Trading-style summaries (directional correctness proxy)
        wins = (y_pred == y_true).astype(int)
        out["win_rate"] = float(np.mean(wins)) if len(wins) else np.nan

        # Profit factor on +-1 unit per prediction
        signed = wins * 2 - 1
        losses = signed[signed < 0]
        gains = signed[signed > 0]
        if losses.size == 0:
            out["profit_factor"] = float(np.inf) if gains.size > 0 else np.nan
        else:
            out["profit_factor"] = float(gains.sum() / abs(losses.sum()))
        return out

    def _fit_calibrator(self, all_val_probs: list[np.ndarray], all_val_labels: list[np.ndarray]) -> None:
        """Fit isotonic regression on all validation folds."""
        p_all = np.concatenate(all_val_probs)
        y_all = np.concatenate(all_val_labels).astype(int)
        x_fit = p_all.reshape(-1,1)
        if self.scaler is not None:
            x_fit = self.scaler.fit_transform(x_fit)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(x_fit.ravel(), y_all)

    def _apply_calibrator(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p
        x = p.reshape(-1,1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
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
        #Validate raw inputs early (clear error messages)
        self._validate_dataframe(stock_df, "stock_df", ["Open", "High", "Low", "Close", "Volume"])
        self._validate_dataframe(market_df, "market_df", ["Open", "High", "Low", "Close", "Volume"])
        if sector_df is not None:
            self._validate_dataframe(sector_df, "sector_df", ["Open", "High", "Low", "Close", "Volume"])

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
        self.logger.info(f"Starting training with {len(df)} samples, {self.n_splits} folds")

        metrics: Dict[str, float] = {}
        last_fold = None

        oof_probs: list[np.ndarray] = [] 
        oof_labels: list[np.ndarray] = []
        oof_preds: list[np.ndarray] = []



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
            y_hat = (p_val >= 0.5).astype(int)
            auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
            metrics[f"fold_{fold}_auc"] = float(auc)
            last_fold = (model, X_val, y_val, p_val)
            
            # Accumulate for OOF
            oof_probs.append(p_val)
            oof_labels.append(y_val)
            oof_preds.append(y_hat)

            best_iter = int(getattr(model, "best_iteration_", self.lgb_params["n_estimators"]))
            auc_disp = "nan" if np.isnan(auc) else f"{auc:.4f}"
            self.logger.info(f"Fold {fold}/{self.n_splits}: train={len(train_idx)}, val={len(val_idx)}, AUC={auc_disp}, best_iter={best_iter}")


        # Use last fold for final model and calibration
        self.model, X_val, y_val, p_val = last_fold
        self._fit_calibrator(oof_probs, oof_labels)
        p_val_cal = self._apply_calibrator(p_val)

        # Aggregate metrics
        oof_p = np.concatenate(oof_probs)
        oof_y = np.concatenate(oof_labels)
        oof_yhat = np.concatenate(oof_preds)

        # Global (OOF) metrics raw
        m_raw = self._compute_metrics(oof_y, oof_p, oof_yhat)
        for key, val in m_raw.items():
            metrics[f"oof_{key}"] = float(val)

        # Last fold calibrated metrics
        yhat_cal = (p_val_cal >= 0.5).astype(int)
        m_last_cal = self._compute_metrics(y_val, p_val_cal, yhat_cal)
        for key, val in m_last_cal.items():
            metrics[f"last_fold_cal_{key}"] = float(val)

        metrics["best_iteration"] = int(getattr(self.model, "best_iteration", self.lgb_params["n_estimators"]))
        metrics["n_features"] = len(self.features_)
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
        self.logger.info(f"Predict_next: label={label}, prob_up={p_up:.3f}, conf={confidence:.3f}, decision={decision}")

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