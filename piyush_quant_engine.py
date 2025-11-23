#!/usr/bin/env python3
"""
piyush_quant_engine.py
Single-file Full Quant Engine (ML + Orderflow + Regime + Backtest)

Usage examples:
    python piyush_quant_engine.py --mode generate_sample
    python piyush_quant_engine.py --mode ingest --input sample_data.csv
    python piyush_quant_engine.py --mode features --input data/sample_1m.parquet
    python piyush_quant_engine.py --mode train --input data/features.parquet
    python piyush_quant_engine.py --mode backtest --input data/features.parquet --model models/ensemble_lgb.pkl
    python piyush_quant_engine.py --mode full_run   # generate -> features -> train -> backtest

Notes:
- Customize paths or pass args. Designed for manual GitHub Actions run.
- Requires LightGBM installed for training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import math

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
from statsmodels.robust.scale import mad

# ---------- Basic configuration ----------
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
LOG_LEVEL = logging.INFO

for d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("piyush_quant_engine")

# ---------- Utility helpers ----------
def now_iso():
    return datetime.utcnow().isoformat()

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ---------- Sample data generator (for testing) ----------
def generate_sample_tick_data(symbol="SAMPLE", minutes=60*24*5, start=None):
    """Generate synthetic 1s trade-like ticks aggregated into 1m candles for testing."""
    if start is None:
        start = datetime.utcnow() - timedelta(days=7)
    idx = pd.date_range(start=start, periods=minutes, freq="1T")  # 1 minute bars
    # create synthetic random walk price
    np.random.seed(42)
    price = 10000 + np.cumsum(np.random.randn(len(idx)) * 10)
    high = price + np.abs(np.random.randn(len(idx)) * 3)
    low = price - np.abs(np.random.randn(len(idx)) * 3)
    openp = price + np.random.randn(len(idx))
    close = price + np.random.randn(len(idx))
    volume = np.abs(np.random.randn(len(idx)) * 100).astype(int) + 1
    df = pd.DataFrame({
        "ts": idx,
        "open": openp,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    df['symbol'] = symbol
    return df.reset_index(drop=True)

# ---------- Ingestion: file loader ----------
def load_candles_from_file(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(p, parse_dates=["ts"])
    else:
        raise ValueError("Unsupported file format; use parquet or csv")
    # ensure columns
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = ["ts","open","high","low","close","volume"]
    df = ensure_cols(df, required)
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values("ts").reset_index(drop=True)
    return df

# ---------- Orderflow features ----------
def infer_buy_sell_volume_from_candles(df):
    """
    If you don't have tick-level buy/sell volume, use heuristic:
    - If close > open => assume more aggressive buys in that candle
    - If close < open => assume more aggressive sells
    This is a heuristic fallback. If you have tick/trade/agg data include real buy_volume & sell_volume.
    """
    df = df.copy()
    df['buy_volume'] = np.where(df['close'] >= df['open'], df['volume']*0.6, df['volume']*0.4)
    df['sell_volume'] = df['volume'] - df['buy_volume']
    # add small noise
    df['buy_volume'] = df['buy_volume'] * (1 + np.random.randn(len(df))*0.01)
    df['sell_volume'] = df['volume'] - df['buy_volume']
    return df

def compute_cvd(df):
    df = df.copy()
    if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
        df = infer_buy_sell_volume_from_candles(df)
    df['signed_vol'] = df['buy_volume'] - df['sell_volume']
    df['cvd'] = df['signed_vol'].cumsum()
    return df

def aggregate_orderflow_features(df):
    df = df.copy()
    df = compute_cvd(df)
    df['imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['volume'] + 1e-9)
    df['cvd_diff'] = df['cvd'].diff().fillna(0)
    df['buy_pct'] = df['buy_volume'] / (df['volume'] + 1e-9)
    # quick footprint proxy: wick ratio classification
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['upper_wick'] - df['lower_wick']) / (df['high'] - df['low'] + 1e-9)
    return df

# ---------- Volume profile (rolling POC) ----------
def compute_vprofile_poc(df, bins=20, window=240):
    """
    Rolling approximate POC using weighted histogram over 'window' bars.
    Adds a 'poc' column (price at which volume density highest) & 'vprofile_entropy' as a proxy.
    """
    df = df.copy().reset_index(drop=True)
    poc_list = [np.nan]*len(df)
    ent_list = [np.nan]*len(df)
    price_mid = (df['high'] + df['low']) / 2.0
    for i in range(len(df)):
        if i < window:
            continue
        wnd = df.iloc[i-window:i]
        mids = (wnd['high'] + wnd['low']) / 2.0
        vols = wnd['volume'].values.astype(float)
        # bin
        try:
            hist, edges = np.histogram(mids, bins=bins, weights=vols)
            max_idx = np.argmax(hist)
            poc_price = (edges[max_idx] + edges[max_idx+1]) / 2.0
            poc_list[i] = poc_price
            p = hist / (hist.sum() + 1e-9)
            ent = -np.sum(p * np.log(p + 1e-12))
            ent_list[i] = ent
        except Exception:
            poc_list[i] = np.nan
            ent_list[i] = np.nan
    df['poc'] = poc_list
    df['vprofile_entropy'] = ent_list
    return df

# ---------- Zone detection (single TF + multi TF merge) ----------
def detect_zones_single_tf(candles: pd.DataFrame,
                           lookback=200,
                           width_pct=0.006,
                           min_touches=3,
                           window_check=20):
    """
    Naive zone detector:
      - Find narrow-range 'bases' within past window_check bars
      - Ensure the base has been 'touched' min_touches times in lookback
    Returns DataFrame of zones: low, high, width, touches, origin_idx, origin_ts
    """
    candles = candles.copy().reset_index(drop=True)
    zones = []
    n = len(candles)
    for i in range(window_check, n):
        window = candles.iloc[i-window_check:i]
        low = window['low'].min()
        high = window['high'].max()
        width = high - low
        if width <= 0:
            continue
        rel_width = width / (window['close'].mean() + 1e-9)
        if rel_width > width_pct:
            continue
        # count touches in last lookback
        look_start = max(0, i - lookback)
        look_wnd = candles.iloc[look_start:i]
        touches = ((look_wnd['low'] <= high) & (look_wnd['high'] >= low)).sum()
        if touches < min_touches:
            continue
        zone = {
            "low": float(low),
            "high": float(high),
            "width": float(width),
            "touches": int(touches),
            "origin_idx": i,
            "origin_ts": str(candles.iloc[i]['ts'])
        }
        zones.append(zone)
    zones_df = pd.DataFrame(zones)
    if zones_df.empty:
        return zones_df
    # merge overlapping/duplicate zones (simple greedy)
    zones_df = zones_df.sort_values("low").reset_index(drop=True)
    merged = []
    cur = zones_df.loc[0].to_dict()
    for r in zones_df.loc[1:].to_dict(orient="records"):
        if r['low'] <= cur['high']:  # overlap -> merge
            cur['high'] = max(cur['high'], r['high'])
            cur['low'] = min(cur['low'], r['low'])
            cur['width'] = cur['high'] - cur['low']
            cur['touches'] = max(cur['touches'], r['touches'])
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)
    return pd.DataFrame(merged)

def merge_zones_across_tfs(zones_by_tf: dict):
    """
    zones_by_tf: dict {tf_name: df_zones}
    Returns merged zones with multi_tf_strength = count of TF overlap and confluence list
    """
    all_zones = []
    # flatten with tf label
    for tf, df in zones_by_tf.items():
        if df is None or df.empty:
            continue
        for r in df.to_dict(orient="records"):
            r2 = r.copy()
            r2['tf'] = tf
            all_zones.append(r2)
    if not all_zones:
        return pd.DataFrame()
    af = pd.DataFrame(all_zones).sort_values("low").reset_index(drop=True)
    merged = []
    cur = af.loc[0].to_dict()
    cur['confluence'] = {cur['tf']}
    for r in af.loc[1:].to_dict(orient="records"):
        # overlap test with small slack
        if r['low'] <= cur['high'] * 1.001 + 1e-9:
            cur['high'] = max(cur['high'], r['high'])
            cur['low'] = min(cur['low'], r['low'])
            cur['width'] = cur['high'] - cur['low']
            cur['touches'] = max(cur.get('touches',0), r.get('touches',0))
            cur.setdefault('confluence', set()).add(r['tf'])
        else:
            cur['multi_tf_strength'] = len(cur.get('confluence',[]))
            cur['confluence_list'] = sorted(list(cur.get('confluence',[])))
            merged.append(cur)
            cur = r
            cur['confluence'] = {cur['tf']}
    cur['multi_tf_strength'] = len(cur.get('confluence',[]))
    cur['confluence_list'] = sorted(list(cur.get('confluence',[])))
    merged.append(cur)
    out = pd.DataFrame(merged).reset_index(drop=True)
    # tidy columns
    if 'tf' in out.columns:
        out = out.drop(columns=['tf'])
    return out

# ---------- Regime detection ----------
def hurst_exponent(ts, min_lag=2, max_lag=20):
    # use log-log slope method
    ts = np.array(ts, dtype=float)
    lags = range(min_lag, max_lag)
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0]*2.0
    return float(hurst)

def detect_regimes(close_series, n_regimes=3):
    X = close_series.diff().dropna().values.reshape(-1,1)
    if len(X) < n_regimes:
        return pd.Series([np.nan]*len(close_series), index=close_series.index)
    gm = GaussianMixture(n_components=n_regimes, random_state=42).fit(X)
    regimes = pd.Series(index=close_series.index, dtype=float)
    regimes.iloc[0] = np.nan
    regimes.iloc[1:] = gm.predict(X)
    return regimes

# ---------- Feature engineering & label creation ----------
def create_features_and_labels(candles: pd.DataFrame,
                               horizon_bars=5,
                               reversal_ret_threshold= -0.005):
    """
    Creates features used for ML training and binary labels for 'reversal' (example).
    horizon_bars: number of future bars to consider for label
    reversal_ret_threshold: e.g., negative return beyond threshold indicates reversal (adjust per asset)
    """
    df = candles.copy().reset_index(drop=True)
    required = ["ts","open","high","low","close","volume"]
    df = ensure_cols(df, required)
    # orderflow
    df = infer_buy_sell_volume_from_candles(df)
    df = aggregate_orderflow_features(df)
    # volume profile
    df = compute_vprofile_poc(df, window=240)
    # wick score proxy
    df['wick_score'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 1e-9)
    # regime
    df['hurst'] = df['close'].rolling(256, min_periods=32).apply(lambda x: hurst_exponent(x) if len(x)>32 else np.nan)
    regimes = detect_regimes(df['close'].fillna(method='ffill'), n_regimes=3)
    df['regime'] = regimes.values
    # smooth features
    df['vol_zscore'] = (df['volume'] - df['volume'].rolling(240,min_periods=1).mean()) / (df['volume'].rolling(240,min_periods=1).std()+1e-9)
    # label: example binary label for 'reversal' meaning next horizon has negative return below threshold
    df['future_close'] = df['close'].shift(-horizon_bars)
    df['ret_future'] = (df['future_close'] - df['close']) / (df['close'] + 1e-9)
    df['label_reversal'] = (df['ret_future'] <= reversal_ret_threshold).astype(int)
    # drop tail rows with NaN label
    return df

# ---------- Train ML (LightGBM ensemble with TimeSeriesSplit) ----------
def prepare_training_dataset(df, feature_cols=None, label_col='label_reversal'):
    df = df.copy().reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [
            'close','volume','imbalance','cvd_diff','poc','vprofile_entropy',
            'wick_score','touches','hurst','vol_zscore','buy_pct'
        ]
    # ensure columns exist
    df = ensure_cols(df, feature_cols + [label_col])
    train_df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    X = train_df[feature_cols]
    y = train_df[label_col]
    return X, y, train_df

def train_lightgbm_ensemble(df,
                            feature_cols=None,
                            label_col='label_reversal',
                            n_splits=5,
                            model_out_path=MODELS_DIR / "ensemble_lgb.pkl"):
    logger.info("Preparing dataset for training")
    X, y, train_df = prepare_training_dataset(df, feature_cols, label_col)
    tss = TimeSeriesSplit(n_splits=n_splits)
    models = []
    aucs = []
    fold = 0
    for train_idx, val_idx in tss.split(X):
        fold += 1
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logger.info(f"Training fold {fold}: train {len(X_train)} / val {len(X_val)}")
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        params = {
            'objective':'binary',
            'metric':'auc',
            'learning_rate':0.05,
            'num_leaves':64,
            'verbosity': -1,
            'seed': 42,
            'boosting_type': 'gbdt'
        }
        bst = lgb.train(params, lgb_train, valid_sets=[lgb_val], early_stopping_rounds=50,
                        num_boost_round=1000, verbose_eval=50)
        ypred = bst.predict(X_val, num_iteration=bst.best_iteration)
        auc = roc_auc_score(y_val, ypred) if len(np.unique(y_val))>1 else float('nan')
        logger.info(f"Fold {fold} AUC: {auc:.4f}")
        aucs.append(auc)
        models.append(bst)
    # Save ensemble (list of boosters)
    joblib.dump(models, str(model_out_path))
    logger.info(f"Saved ensemble to {model_out_path}")
    logger.info(f"Mean AUC across folds: {np.nanmean(aucs):.4f}")
    return models

# ---------- Predict using saved ensemble ----------
def predict_with_ensemble(df, model_path: str, feature_cols=None, out_col='p_reversal'):
    df = df.copy().reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [
            'close','volume','imbalance','cvd_diff','poc','vprofile_entropy',
            'wick_score','touches','hurst','vol_zscore','buy_pct'
        ]
    models = joblib.load(model_path)
    X = df[feature_cols].fillna(method='ffill').fillna(0)
    # average ensemble predictions
    preds = np.zeros((len(X),))
    for m in models:
        preds += m.predict(X, num_iteration=m.best_iteration)  # shape (n,)
    preds /= max(1, len(models))
    df[out_col] = preds
    return df

# ---------- Backtester (vectorized simplified) ----------
def compute_atr(df, period=14):
    df = df.copy()
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(period, min_periods=1).mean()
    return df

def backtest_strategy(df, pred_col='p_reversal', entry_threshold=0.7, sl_atr=1.5, tp_atr=2.0,
                      initial_capital=100000, risk_per_trade=0.001):
    """
    Very simplified backtest:
    - When p_reversal > threshold, go long assuming a reversal with SL = entry - sl_atr*ATR and TP = entry + tp_atr*ATR
    - Position sizing uses risk_per_trade fraction of capital (volatility normalized)
    - Uses candle-level execution assumptions (entry at next candle open)
    """
    df = df.copy().reset_index(drop=True)
    df = compute_atr(df)
    trades = []
    capital = initial_capital
    position = 0
    entry_price = None
    entry_idx = None
    for i in range(len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]  # we execute at next open (simulate latency)
        if position == 0:
            if row.get(pred_col, 0) >= entry_threshold:
                entry_price = nxt['open']
                atr = nxt['atr'] if not np.isnan(nxt['atr']) and nxt['atr'] > 0 else (nxt['high'] - nxt['low'])
                sl = entry_price - sl_atr * atr
                tp = entry_price + tp_atr * atr
                # size: risk_per_trade of capital / (entry - sl)
                risk_amount = capital * risk_per_trade
                if (entry_price - sl) <= 0:
                    units = 0
                else:
                    units = math.floor(risk_amount / max(1e-9, (entry_price - sl)))
                if units <= 0:
                    continue
                position = units
                entry_idx = i+1
                trades.append({
                    "entry_idx": entry_idx,
                    "entry_ts": str(nxt['ts']),
                    "entry_price": float(entry_price),
                    "units": units,
                    "sl": float(sl),
                    "tp": float(tp),
                    "closed": False
                })
        else:
            # check open-high-low-close of the candle for hitting SL or TP
            if len(trades) == 0:
                position = 0
                continue
            t = trades[-1]
            if t['closed']:
                position = 0
                continue
            # use current candle (i+1 since entry was at next open previously)
            cur = df.iloc[i+1]
            # check SL first then TP (pessimistic)
            exited = False
            if cur['low'] <= t['sl']:
                exit_price = t['sl']
                exited = True
            elif cur['high'] >= t['tp']:
                exit_price = t['tp']
                exited = True
            else:
                # if not hit, exit next candle open (small unrealized P&L)
                if i+2 < len(df):
                    exit_price = df.iloc[i+2]['open']
                    exited = True
                else:
                    exit_price = cur['close']
                    exited = True
            if exited:
                pnl = (exit_price - t['entry_price']) * t['units']
                capital = capital + pnl
                t['exit_ts'] = str(df.iloc[min(i+2, len(df)-1)]['ts'])
                t['exit_price'] = float(exit_price)
                t['pnl'] = float(pnl)
                t['closed'] = True
                t['capital_after'] = float(capital)
                position = 0
    trades_df = pd.DataFrame(trades)
    # metrics
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    wins = trades_df[trades_df['pnl'] > 0].shape[0] if not trades_df.empty else 0
    losses = trades_df[trades_df['pnl'] <= 0].shape[0] if not trades_df.empty else 0
    win_rate = wins / (wins + losses + 1e-9)
    num_trades = len(trades_df)
    return {
        "initial_capital": initial_capital,
        "final_capital": float(capital),
        "total_pnl": float(total_pnl),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "trades": trades_df
    }

# ---------- End-to-end pipeline helpers ----------
def run_generate_sample(out_path=None):
    df = generate_sample_tick_data(symbol="SAMPLE", minutes=60*24*10)
    if out_path is None:
        out_path = DATA_DIR / "sample_1m.parquet"
    else:
        out_path = Path(out_path)
    df.to_parquet(out_path, index=False)
    logger.info(f"Generated sample data -> {out_path}")
    return out_path

def run_ingest(input_path, out_path=None):
    df = load_candles_from_file(input_path)
    if out_path is None:
        out_path = DATA_DIR / Path(input_path).stem + ".parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Ingested data saved to {out_path}")
    return out_path

def run_features(input_path, out_path=None):
    df = load_candles_from_file(input_path)
    dff = create_features_and_labels(df)
    if out_path is None:
        out_path = DATA_DIR / "features.parquet"
    dff.to_parquet(out_path, index=False)
    logger.info(f"Features dataset written to {out_path}")
    return out_path

def run_train(features_path, model_out=None):
    df = load_candles_from_file(features_path)
    if model_out is None:
        model_out = MODELS_DIR / "ensemble_lgb.pkl"
    models = train_lightgbm_ensemble(df, model_out_path=model_out)
    return model_out

def run_predict(features_path, model_path, out_path=None):
    df = load_candles_from_file(features_path)
    dfp = predict_with_ensemble(df, model_path)
    if out_path is None:
        out_path = OUTPUT_DIR / "predictions.parquet"
    dfp.to_parquet(out_path, index=False)
    logger.info(f"Predictions saved to {out_path}")
    return out_path

def run_backtest(features_path, model_path=None, out_dir=OUTPUT_DIR):
    df = load_candles_from_file(features_path)
    if model_path and Path(model_path).exists():
        df = predict_with_ensemble(df, model_path)
    else:
        # create simple heuristic score using imbalance + wick for demo
        df['p_reversal'] = (df['imbalance'].fillna(0)*0.3 + (1 - np.abs(df['wick_ratio'].fillna(0)))*0.3 + (1/(1+np.exp(-df['vol_zscore'].fillna(0))))*0.4)
    # map to expected column name
    if 'p_reversal' not in df.columns:
        df['p_reversal'] = df.get('p_reversal', df.get('p_reversal', 0))
    res = backtest_strategy(df, pred_col='p_reversal', entry_threshold=0.7)
    # save trades
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trades_file = Path(out_dir) / f"trades_{ts}.parquet"
    if not res['trades'].empty:
        res['trades'].to_parquet(trades_file, index=False)
        logger.info(f"Saved trades: {trades_file}")
    else:
        logger.info("No trades executed")
    # summary
    summary_file = Path(out_dir) / f"backtest_summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "initial_capital": res['initial_capital'],
            "final_capital": res['final_capital'],
            "total_pnl": res['total_pnl'],
            "num_trades": res['num_trades'],
            "win_rate": res['win_rate']
        }, f, indent=2)
    logger.info(f"Backtest summary saved: {summary_file}")
    return res

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Single-file Quant Engine")
    p.add_argument("--mode", type=str, required=True,
                   choices=["generate_sample","ingest","features","train","predict","backtest","full_run"],
                   help="Mode to run")
    p.add_argument("--input", type=str, default=None, help="Input file path")
    p.add_argument("--model", type=str, default=str(MODELS_DIR / "ensemble_lgb.pkl"), help="Model path")
    p.add_argument("--out", type=str, default=None, help="Output path")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.mode == "generate_sample":
            out = run_generate_sample(args.out)
            logger.info(f"Sample generated: {out}")
        elif args.mode == "ingest":
            if not args.input:
                raise ValueError("ingest mode requires --input path to CSV/parquet")
            run_ingest(args.input, args.out)
        elif args.mode == "features":
            if not args.input:
                raise ValueError("features mode requires --input path to candle parquet/csv")
            run_features(args.input, args.out)
        elif args.mode == "train":
            if not args.input:
                raise ValueError("train mode requires --input features.parquet")
            run_train(args.input, Path(args.model))
        elif args.mode == "predict":
            if not args.input:
                raise ValueError("predict mode requires --input features.parquet")
            run_predict(args.input, args.model, args.out)
        elif args.mode == "backtest":
            if not args.input:
                raise ValueError("backtest mode requires --input features.parquet")
            res = run_backtest(args.input, model_path=args.model)
            logger.info(f"Backtest result: final capital {res['final_capital']}, trades {res['num_trades']}, winrate {res['win_rate']:.2f}")
        elif args.mode == "full_run":
            # generate sample -> features -> train -> predict -> backtest
            sample = run_generate_sample()
            features = run_features(sample)
            model = run_train(features)
            pred_out = run_predict(features, model)
            res = run_backtest(features, model_path=model)
            logger.info("Full run complete")
        else:
            logger.error("Unknown mode")
    except Exception as e:
        logger.exception("Error during execution")
        raise

if __name__ == "__main__":
    main()
