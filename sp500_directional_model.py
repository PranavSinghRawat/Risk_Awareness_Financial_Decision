"""
====================================================================================================
PROJECT: Risk-Aware & Explainable AI-Based Financial Decision Support for Stock Investment
STANDARD: Institutional Quantitative Asset Pricing & Financial Machine Learning
REFERENCE FRAMEWORK: Gu, Kelly, & Xiu (2020); López de Prado (2018); Rockafellar & Uryasev (2000)
====================================================================================================

Key Production Architecture:
1. Strict Information Causality & Non-Leaking Pipeline:
   - Walk-forward chronological 80/20 train/test partition (Zero random shuffling).
   - In-sample scaler fitting: StandardScaler fit ONLY on X_train, transforming X_test.
   - Causal alignment: Features at close(t) mapped to realization at close(t+1).
2. Multi-Pillar Stationary Feature Space:
   - Trend Pillar: Normalized 20/50 Exponential Moving Average (EMA) spread.
   - Momentum Pillar: Multi-lag log-returns, Overnight news gap, 14-day Wilder's RSI.
   - Volatility / Risk Pillar: 14-day Normalized Average True Range % (NATR%).
   - Institutional Flow Pillar: 20-day Volume Surge Ratio (RVOL).
3. Downside Risk & Uncertainty Quantification:
   - Historical & Parametric Value-at-Risk (VaR 95%) and Conditional VaR (CVaR / Expected Shortfall).
   - Epistemic Model Uncertainty: Random Forest inter-tree variance sigma_trees = std(p_b).
4. Tripartite Decision Engine (BUY / HOLD / AVOID):
   - AVOID (100% Cash): Activated during volatility spikes (NATR > threshold) or high model doubt.
   - BUY (Full Long): High Up-probability clearing hurdle rate with low downside tail risk.
   - HOLD: Neutral risk-reward regime preserving capital.
5. Friction-Adjusted Realistic Backtesting:
   - Conservative 8 bps (0.08%) transaction fee & slippage applied to every position change.
   - Portfolio metrics: Net CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown (MDD), Turnover drag.
6. Local Explainability Engine (XAI):
   - Instance-level feature attributions for today's forecast explaining WHY a decision was issued.
====================================================================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


from typing import Optional


# =====================================================================
# 1. DATA INGESTION & CANONICAL PREPROCESSING
# =====================================================================

def load_canonical_dxy_data(dxy_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads Foreign Exchange daily series (USD/INR exchange rate).
    Replaces previous US Dollar Index (DXY) dataset.
    """
    canonical_candidates = [
        "USD-INR.csv",
        os.path.join(os.path.dirname(__file__), "USD-INR.csv") if "__file__" in globals() else "USD-INR.csv"
    ]
    if dxy_path:
        canonical_candidates.insert(0, dxy_path)

    for c in canonical_candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            df.columns = [col.strip() for col in df.columns]
            rename_map = {'Close': 'dxy_close', 'close': 'dxy_close', 'Price': 'dxy_close', 'Adj Close': 'dxy_adj_close'}
            df = df.rename(columns=rename_map)
            if 'Date' in df.columns and 'dxy_close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['dxy_close'] = pd.to_numeric(df['dxy_close'], errors='coerce')
                df = df.dropna(subset=['Date', 'dxy_close']).sort_values('Date').reset_index(drop=True)
                print(f"      Loaded Foreign Exchange currency (USD/INR) daily bars from {c}: {len(df)} sessions ({df['Date'].min().date()} to {df['Date'].max().date()}).")
                return df[['Date', 'dxy_close']].sort_values('Date').reset_index(drop=True)

    return None


def load_canonical_vix_data(vix_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads Volatility Index daily series (India VIX).
    Replaces previous US CBOE VIX dataset.
    """
    candidates = [
        "India VIX Historical Data 101.csv",
        os.path.join(os.path.dirname(__file__), "India VIX Historical Data 101.csv") if "__file__" in globals() else "India VIX Historical Data 101.csv"
    ]
    if vix_path:
        candidates.insert(0, vix_path)

    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            df.columns = [col.strip() for col in df.columns]
            rename_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ['date', 'observation_date', 'time']:
                    rename_map[col] = 'Date'
                elif cl in ['close', 'price', 'vix_close', 'vixcls']:
                    rename_map[col] = 'vix_close'
            df = df.rename(columns=rename_map)
            if 'Date' in df.columns and 'vix_close' in df.columns:
                # India VIX uses DD-MM-YYYY format
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
                if df['Date'].isna().sum() > len(df) * 0.5:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                df['vix_close'] = pd.to_numeric(df['vix_close'], errors='coerce')
                df = df.dropna(subset=['Date', 'vix_close']).sort_values('Date').reset_index(drop=True)
                df['vix_close'] = df['vix_close'].ffill().bfill()
                print(f"      Loaded Volatility Index (India VIX) daily bars from {c}: {len(df)} sessions ({df['Date'].min().date()} to {df['Date'].max().date()}).")
                return df[['Date', 'vix_close']]

def load_canonical_benchmark_data(benchmark_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Loads NIFTY 50 broad market benchmark daily returns.
    Pre-computed from NIFTY 50 constituent stocks.
    """
    candidates = [
        "NIFTY50_benchmark.csv",
        os.path.join(os.path.dirname(__file__), "NIFTY50_benchmark.csv") if "__file__" in globals() else "NIFTY50_benchmark.csv",
    ]
    if benchmark_path:
        candidates.insert(0, benchmark_path)

    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['nifty_return'] = pd.to_numeric(df['nifty_return'], errors='coerce')
            df = df.dropna(subset=['Date', 'nifty_return']).sort_values('Date').reset_index(drop=True)
            print(f"      Loaded NIFTY 50 broad market benchmark from {c}: {len(df)} sessions ({df['Date'].min().date()} to {df['Date'].max().date()}).")
            return df[['Date', 'nifty_return']]

    return None


def load_canonical_data(csv_path: str, ticker: str = "AAPL") -> pd.DataFrame:
    """
    Loads price dataset, detects multi-index/yfinance formats, filters for single
    ticker if multi-asset, sorts chronologically, and cleans nulls.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")

    print(f"[1/6] Ingesting market data from: {csv_path}")

    # Inspect first lines to detect multi-level headers
    with open(csv_path, 'r', encoding='utf-8') as f:
        l1 = f.readline()
        l2 = f.readline()

    if "Ticker" in l2 or "Price" in l1:
        raw_df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df.index.name = "Date"
        df = raw_df.reset_index()
    else:
        df = pd.read_csv(csv_path)

    # Standardize column headers
    df.columns = [c.strip() for c in df.columns]

    # Multi-asset ticker filtering if present
    ticker_col = None
    for cand in ['Name', 'Symbol', 'Ticker', 'ticker', 'symbol', 'name']:
        if cand in df.columns:
            ticker_col = cand
            break

    if ticker_col is not None:
        initial_len = len(df)
        df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()].copy()
        print(f"      Filtered multi-asset dataset for '{ticker}' ({len(df)} of {initial_len} rows).")

    # Rename canonical OHLCV columns regardless of case
    rename_map = {}
    for col in df.columns:
        c_low = col.lower()
        if c_low == 'date':
            rename_map[col] = 'Date'
        elif c_low == 'close':
            rename_map[col] = 'Close'
        elif c_low == 'open':
            rename_map[col] = 'Open'
        elif c_low == 'high':
            rename_map[col] = 'High'
        elif c_low == 'low':
            rename_map[col] = 'Low'
        elif c_low == 'volume':
            rename_map[col] = 'Volume'
    df.rename(columns=rename_map, inplace=True)

    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Missing required canonical column: '{req}'. Present: {list(df.columns)}")

    # Sort strictly chronologically
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Cast numeric and forward/backward fill missing prices
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[num_cols] = df[num_cols].ffill().bfill()

    print(f"      Loaded {len(df)} trading bars from {df['Date'].min().date()} to {df['Date'].max().date()}.")
    return df


# =====================================================================
# 2. MULTI-PILLAR STATIONARY FEATURE ENGINEERING
# =====================================================================

def construct_stationary_features(
    df: pd.DataFrame, 
    dxy_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transforms non-stationary price levels into stationary, scale-invariant features
    spanning Trend, Momentum, Volatility (Risk), Institutional Flow, Cross-Asset Macro FX (DXY),
    Options-Implied Market Fear (CBOE VIX), and Broad Market Benchmark (NIFTY 50 Relative Alpha).
    """
    print("[2/6] Engineering stationary multi-pillar feature space...")
    data = df.copy()

    # Auto-load canonical US Dollar Index if available
    if dxy_df is None:
        dxy_df = load_canonical_dxy_data()

    has_dxy = False
    if dxy_df is not None and 'dxy_close' in dxy_df.columns:
        # Inner merge on calendar Date to strictly synchronize macro series
        merged = pd.merge(data, dxy_df[['Date', 'dxy_close']], on='Date', how='inner')
        if len(merged) > 200:
            data = merged.sort_values('Date').reset_index(drop=True)
            has_dxy = True

    # Auto-load canonical CBOE Volatility Index (VIX) if available
    if vix_df is None:
        vix_df = load_canonical_vix_data()

    has_vix = False
    if vix_df is not None and 'vix_close' in vix_df.columns:
        merged_vix = pd.merge(data, vix_df[['Date', 'vix_close']], on='Date', how='inner')
        if len(merged_vix) > 200:
            data = merged_vix.sort_values('Date').reset_index(drop=True)
            has_vix = True

    # Auto-load canonical NIFTY 50 Broad Market Benchmark if available
    if benchmark_df is None:
        benchmark_df = load_canonical_benchmark_data()

    has_benchmark = False
    if benchmark_df is not None and 'nifty_return' in benchmark_df.columns:
        merged_bench = pd.merge(data, benchmark_df[['Date', 'nifty_return']], on='Date', how='inner')
        if len(merged_bench) > 200:
            data = merged_bench.sort_values('Date').reset_index(drop=True)
            has_benchmark = True

    # --- Pillar 1: Momentum & Autoregressive Lags ---
    # Log-returns: r_t = ln(Close_t / Close_{t-1})
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['log_return_lag1'] = data['log_return'].shift(1)
    data['log_return_lag2'] = data['log_return'].shift(2)
    
    # Overnight gap return: captures off-market information flow
    data['gap_return'] = np.log(data['Open'] / data['Close'].shift(1))

    # --- Pillar 2: Trend Regime (Scale-Invariant) ---
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['trend_spread'] = ((data['EMA_20'] - data['EMA_50']) / data['EMA_50']) * 100.0

    # 14-day Wilder's Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    data['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # --- Pillar 3: Volatility & Tail Risk (Normalized ATR %) ---
    hl = data['High'] - data['Low']
    hc = (data['High'] - data['Close'].shift(1)).abs()
    lc = (data['Low'] - data['Close'].shift(1)).abs()
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    data['ATR_14'] = true_range.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    data['NATR_pct'] = (data['ATR_14'] / data['Close']) * 100.0

    # --- Pillar 4: Institutional Flow (Relative Volume RVOL) ---
    vol_sma20 = data['Volume'].rolling(window=20).mean()
    data['vol_ratio'] = data['Volume'] / (vol_sma20 + 1e-10)

    feature_cols = [
        'log_return',
        'log_return_lag1',
        'log_return_lag2',
        'gap_return',
        'trend_spread',
        'RSI_14',
        'NATR_pct',
        'vol_ratio'
    ]

    # --- Pillar 5: Cross-Asset Macro FX (US Dollar Index DXY) ---
    if has_dxy:
        # 1-day log return of DXY
        data['dxy_return'] = np.log(data['dxy_close'] / data['dxy_close'].shift(1))
        # 1-day lagged DXY return
        data['dxy_return_lag1'] = data['dxy_return'].shift(1)
        # Macro FX 20-day trend deviation %
        dxy_sma20 = data['dxy_close'].rolling(20).mean()
        data['dxy_trend'] = ((data['dxy_close'] - dxy_sma20) / (dxy_sma20 + 1e-10)) * 100.0
        # 10-day rolling annualized currency volatility
        data['dxy_vol10'] = data['dxy_return'].rolling(10).std() * np.sqrt(252) * 100.0

        dxy_features = ['dxy_return', 'dxy_return_lag1', 'dxy_trend', 'dxy_vol10']
        feature_cols.extend(dxy_features)
        print(f"      [Cross-Asset Integration] Merged US Dollar Index (DXY). Added 4 macro features: {dxy_features}")

    # --- Pillar 6: Options-Implied Market Fear (CBOE VIX) ---
    if has_vix:
        vix_sma20 = data['vix_close'].rolling(20).mean()
        vix_std20 = data['vix_close'].rolling(20).std()
        data['vix_zscore'] = (data['vix_close'] - vix_sma20) / (vix_std20 + 1e-10)
        # Binary Capitulation Spike: Z-score > 1.5 indicates extreme panic exhaustion / relief rally setup
        data['vix_spike'] = (data['vix_zscore'] > 1.5).astype(float)

        vix_features = ['vix_spike']
        feature_cols.extend(vix_features)
        print(f"      [Cross-Asset Integration] Merged CBOE Volatility Index (VIX). Added options fear spike feature: {vix_features}")

    # --- Pillar 7: Broad Market Benchmark & Relative Alpha ---
    if has_benchmark:
        data['rel_alpha'] = data['log_return'] - data['nifty_return']
        data['rel_alpha_lag1'] = data['rel_alpha'].shift(1)
        bench_features = ['nifty_return', 'rel_alpha', 'rel_alpha_lag1']
        feature_cols.extend(bench_features)
        print(f"      [Cross-Asset Integration] Merged NIFTY 50 Benchmark. Added 3 market alpha features: {bench_features}")

    # --- Pillar 8: Institutional Delivery Volume % ---
    if '%Deliverble' in data.columns:
        deliv = pd.to_numeric(data['%Deliverble'], errors='coerce')
        data['delivery_pct'] = deliv.fillna(deliv.rolling(20, min_periods=1).mean()).fillna(0.35)
        feature_cols.append('delivery_pct')
        print(f"      [Institutional Flow] Added delivery volume ratio feature: ['delivery_pct']")

    # --- Target Formulation ---
    # Binary next-day price direction: 1 if tomorrow's return > 0, else 0
    data['next_log_return'] = data['log_return'].shift(-1)
    data['target'] = (data['next_log_return'] > 0).astype('Int64')

    # Drop indicator warmup rows
    valid_data = data.dropna(subset=feature_cols).copy()

    # Terminal row (T) is the latest market close: its stationary features are fully observed,
    # but tomorrow's return (T+1) is unobserved in history. Used for live forward inference.
    inference_bar = valid_data.iloc[-1].copy()

    # Supervised dataset: all rows where tomorrow's return was realized (0 to T-1)
    clean_data = valid_data.iloc[:-1].copy()
    clean_data['target'] = clean_data['target'].astype(int)
    clean_data.attrs['inference_bar'] = inference_bar

    print(f"      Feature matrix: {clean_data.shape[0]} supervised observations across {len(feature_cols)} stationary features.")
    print(f"      Terminal inference bar: {inference_bar['Date'].date()} (Close: ${inference_bar['Close']:.2f})")
    return clean_data, feature_cols, inference_bar


# =====================================================================
# 3. DOWNSIDE RISK QUANTIFICATION (VaR & CVaR / Expected Shortfall)
# =====================================================================

def evaluate_downside_risk(returns_series: pd.Series, alpha: float = 0.95) -> dict:
    """
    Computes statistical downside risk metrics (Value-at-Risk & Expected Shortfall).
    Ref: Rockafellar & Uryasev (2000)
    """
    clean_r = returns_series.dropna()
    if len(clean_r) == 0:
        return {"VaR_95": 0.0, "CVaR_95": 0.0, "Annualized_Vol": 0.0}

    # Historical VaR (95% confidence one-day threshold loss)
    var_95 = clean_r.quantile(1.0 - alpha)
    # Expected Shortfall (CVaR): Average loss beyond VaR threshold
    cvar_tail = clean_r[clean_r <= var_95]
    cvar_95 = cvar_tail.mean() if len(cvar_tail) > 0 else var_95

    # Parametric Gaussian VaR
    mu = clean_r.mean()
    sigma = clean_r.std()
    param_var_95 = mu - (1.64485 * sigma)
    ann_vol = sigma * np.sqrt(252)

    return {
        "Historical_VaR_95_pct": round(float(var_95 * 100), 2),
        "Historical_CVaR_95_pct": round(float(cvar_95 * 100), 2),
        "Parametric_VaR_95_pct": round(float(param_var_95 * 100), 2),
        "Annualized_Vol_pct": round(float(ann_vol * 100), 2)
    }


# =====================================================================
# 4. CHRONOLOGICAL SPLIT, SCALING & UNCERTAINTY MODELING
# =====================================================================

def train_risk_aware_model(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    split_ratio: float = 0.80
):
    """
    Executes:
    1. Strict walk-forward chronological train/test split.
    2. Zero-leakage standard scaling (fit ONLY on X_train).
    3. Random Forest training with variance regularization.
    4. Epistemic uncertainty calculation via ensemble tree disagreement.
    """
    print("[3/6] Executing walk-forward chronological split & zero-leakage training...")
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Zero-leakage scaling: fit exclusively on train, transform test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])

    y_train = train_df['target'].values
    y_test = test_df['target'].values

    # Regularized Random Forest Classifier to prevent overfitting on daily market noise
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=4,
        min_samples_split=30,
        min_samples_leaf=30,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    # Calculate model predictions and Epistemic Uncertainty across all 100 trees
    # tree_predictions: shape (n_trees, n_test_samples)
    tree_probs = np.array([tree.predict_proba(X_test_scaled)[:, 1] for tree in rf.estimators_])
    mean_prob_up = np.mean(tree_probs, axis=0)
    # Inter-tree standard deviation measures model uncertainty/disagreement
    epistemic_uncertainty = np.std(tree_probs, axis=0)

    print(f"      Train Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} bars)")
    print(f"      Test Period:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} bars)")
    print(f"      Mean Model Uncertainty (sigma_trees): {epistemic_uncertainty.mean():.4f}")

    return {
        "model": rf,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "mean_prob_up": mean_prob_up,
        "epistemic_uncertainty": epistemic_uncertainty
    }


# =====================================================================
# 5. TRIPARTITE DECISION LOGIC & REALISTIC FRICTION BACKTEST
# =====================================================================

def determine_decision_action(
    prob_up: float,
    uncertainty: float,
    natr_pct: float,
    has_vix_spike: bool,
    vol_cap: float = 3.0,
    uncertainty_cap: float = 0.22,
    hurdle_prob: float = 0.52,
    lower_hurdle: float = 0.48
) -> dict:
    """
    Evaluates institutional tripartite decision logic (BUY / HOLD / PASS)
    under strict risk awareness and transaction friction hurdle rates.
    """
    is_vol_shock = (natr_pct > vol_cap) or has_vix_spike
    is_extreme_uncertainty = (uncertainty > uncertainty_cap)

    if is_vol_shock or is_extreme_uncertainty:
        action_code = "PASS"
        action_label = "HOLD CASH (SAFE)"
        if has_vix_spike:
            desc = "Macro CBOE VIX panic spike detected. Capital preserved in 100% Cash."
        elif natr_pct > vol_cap:
            desc = f"Extreme volatility surge ({natr_pct:.1f}% NATR > historical 90th percentile). Shielding capital in Cash."
        else:
            desc = f"Elevated model uncertainty ({uncertainty:.3f} > {uncertainty_cap}). Capital protected in Cash."
    elif prob_up >= hurdle_prob:
        action_code = "BUY"
        action_label = "BUY (INVEST)"
        desc = f"Strong upward conviction ({prob_up*100:.1f}%) exceeds hurdle rate ({hurdle_prob*100:.1f}%) under safe volatility regime."
    elif prob_up >= lower_hurdle:
        action_code = "HOLD"
        action_label = "HOLD (NEUTRAL)"
        desc = f"Directional probability ({prob_up*100:.1f}%) is in neutral deadband [{lower_hurdle*100:.0f}%-{hurdle_prob*100:.0f}%]. Preserving position to avoid transaction friction."
    else:
        action_code = "PASS"
        action_label = "HOLD CASH (SAFE)"
        desc = f"Bearish downward directional expectation ({(1.0-prob_up)*100:.1f}% Down). Capital preserved in 100% Cash."

    return {
        "action_code": action_code,
        "action_label": action_label,
        "description": desc,
        "is_vol_shock": is_vol_shock,
        "is_extreme_uncertainty": is_extreme_uncertainty
    }


def predict_next_session(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    inference_bar: pd.Series,
    feature_cols: list[str]
) -> dict:
    """
    Computes out-of-sample forward prediction and uncertainty for the terminal market bar
    to forecast the next upcoming trading session (T -> T+1).
    """
    sample_df = pd.DataFrame([inference_bar[feature_cols]])
    X_scaled = scaler.transform(sample_df)

    tree_probs = np.array([tree.predict_proba(X_scaled)[0, 1] for tree in model.estimators_])
    prob_up = float(np.mean(tree_probs))
    uncertainty = float(np.std(tree_probs))

    return {
        "prob_up": prob_up,
        "prob_down": float(1.0 - prob_up),
        "uncertainty": uncertainty,
        "direction": "UP" if prob_up >= 0.50 else "DOWN",
        "direction_code": 1 if prob_up >= 0.50 else 0,
        "confidence": round(max(prob_up, 1.0 - prob_up) * 100, 1),
        "date": str(inference_bar["Date"].date()),
        "close": float(inference_bar["Close"])
    }


def execute_tripartite_decision_backtest(
    model_bundle: dict,
    hurdle_prob: float = 0.52,
    vol_cap_natr: float = 3.0,
    uncertainty_cap: float = 0.22,
    cost_bps: float = 8.0
) -> dict:
    """
    Implements:
    - 3-Tier Institutional Decision Engine (BUY / HOLD / PASS in 100% Cash)
    - Dynamic Asset-Adaptive Volatility Guard (Adapts to any stock & future data)
    - Realistic Friction Backtest (8 bps transaction drag per rebalance)
    - Downside Risk & Capital Preservation Metrics
    """
    print("[4/6] Executing Risk-Aware Decision Engine & Backtest...")
    test_df = model_bundle["test_df"].copy()
    train_df = model_bundle["train_df"]
    probs = model_bundle["mean_prob_up"]
    uncertainty = model_bundle["epistemic_uncertainty"]
    natr = test_df["NATR_pct"].values
    n = len(test_df)

    # Dynamic Asset-Adaptive Volatility Threshold:
    # Uses 90th percentile of training volatility so it works automatically for any stock
    # (e.g. KO ~1.5%, AAPL ~2.4%, BBY ~4.1%, NVDA ~4.8%) and for any new data added.
    if "NATR_pct" in train_df.columns:
        train_natr_clean = train_df["NATR_pct"].dropna()
        dynamic_vol_cap = float(np.percentile(train_natr_clean, 90))
    else:
        dynamic_vol_cap = vol_cap_natr

    # Allow custom override if explicitly tightened, but default to stock-relative distribution
    effective_vol_cap = max(dynamic_vol_cap, 2.5)

    # Decision Engine:
    # 1 = ALLOCATE (LONG), 0 = PRESERVE (100% Cash)
    signals = np.zeros(n, dtype=int)
    reasons = []
    actions = []

    for i in range(n):
        p = probs[i]
        u = uncertainty[i]
        vol = natr[i]
        has_vix_spike = bool('vix_spike' in test_df.columns and test_df['vix_spike'].iloc[i] == 1)

        decision = determine_decision_action(
            prob_up=p,
            uncertainty=u,
            natr_pct=vol,
            has_vix_spike=has_vix_spike,
            vol_cap=effective_vol_cap,
            uncertainty_cap=uncertainty_cap,
            hurdle_prob=hurdle_prob,
            lower_hurdle=0.48
        )
        actions.append(decision["action_code"])
        reasons.append(decision["description"])

        if decision["action_code"] == "BUY":
            signals[i] = 1
        elif decision["action_code"] == "HOLD":
            # In neutral deadband, maintain previous day's position to prevent churn friction
            signals[i] = signals[i - 1] if i > 0 else 0
        else:  # PASS
            signals[i] = 0

    test_df['signal'] = signals
    test_df['action_code'] = actions
    test_df['decision_reason'] = reasons

    # --- Realistic Portfolio Backtesting Engine ---
    # Next-day realized return is observed for each row in supervised test_df
    if 'next_log_return' in test_df.columns:
        ret_vector = test_df['next_log_return'].values
    else:
        ret_vector = test_df['log_return'].shift(-1).fillna(0).values

    positions = signals
    n_bars = len(positions)

    # Deduct transaction friction on turnover (8 bps = 0.0008)
    friction_rate = cost_bps / 10000.0
    turnover = np.abs(np.diff(positions, prepend=0))
    cost_drag = turnover * friction_rate

    # Strategy net daily returns
    strategy_daily_ret = (positions * ret_vector) - cost_drag
    bnh_daily_ret = ret_vector

    # Cumulative wealth trajectories
    cum_strategy = np.cumprod(1.0 + strategy_daily_ret)
    cum_bnh = np.cumprod(1.0 + bnh_daily_ret)

    # Financial Performance Ratios
    mean_strat = np.mean(strategy_daily_ret)
    std_strat = np.std(strategy_daily_ret) + 1e-10
    sharpe_strat = (mean_strat / std_strat) * np.sqrt(252)

    mean_bnh = np.mean(bnh_daily_ret)
    std_bnh = np.std(bnh_daily_ret) + 1e-10
    sharpe_bnh = (mean_bnh / std_bnh) * np.sqrt(252)

    # Canonical Sortino Ratio (Downside deviation with target 0 over all N observations)
    # Ref: Sortino & Price (1994)
    downside_diff = np.minimum(strategy_daily_ret, 0.0)
    downside_dev = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
    sortino_strat = (mean_strat * 252) / (downside_dev + 1e-10)

    bnh_downside_diff = np.minimum(bnh_daily_ret, 0.0)
    bnh_downside_dev = np.sqrt(np.mean(bnh_downside_diff ** 2)) * np.sqrt(252)
    sortino_bnh = (mean_bnh * 252) / (bnh_downside_dev + 1e-10)

    # Maximum Drawdown (MDD)
    strat_peaks = np.maximum.accumulate(cum_strategy)
    strat_drawdowns = (cum_strategy - strat_peaks) / strat_peaks
    max_dd_strat = np.min(strat_drawdowns)

    bnh_peaks = np.maximum.accumulate(cum_bnh)
    bnh_drawdowns = (cum_bnh - bnh_peaks) / bnh_peaks
    max_dd_bnh = np.min(bnh_drawdowns)

    cagr_strat = (cum_strategy[-1] ** (252 / max(n_bars, 1)) - 1.0) * 100.0
    cagr_bnh = (cum_bnh[-1] ** (252 / max(n_bars, 1)) - 1.0) * 100.0

    # ML Classification metrics on raw probabilities
    binary_preds = (probs >= 0.50).astype(int)
    y_test = model_bundle["y_test"]
    acc = accuracy_score(y_test, binary_preds)
    prec = precision_score(y_test, binary_preds, zero_division=0)
    rec = recall_score(y_test, binary_preds, zero_division=0)
    f1 = f1_score(y_test, binary_preds, zero_division=0)

    # Downside Risk on Test Split
    downside_stats = evaluate_downside_risk(pd.Series(ret_vector))

    results = {
        "ml_metrics": {
            "Accuracy_pct": round(acc * 100, 2),
            "Precision_Up_pct": round(prec * 100, 2),
            "Recall_Up_pct": round(rec * 100, 2),
            "F1_Score": round(f1, 4),
            "Buy_Hold_Benchmark_Acc_pct": round(float(y_test.mean() * 100), 2)
        },
        "financial_metrics": {
            "Strategy_Net_CAGR_pct": round(float(cagr_strat), 2),
            "Buy_Hold_Benchmark_CAGR_pct": round(float(cagr_bnh), 2),
            "Strategy_Sharpe_Ratio": round(float(sharpe_strat), 3),
            "Buy_Hold_Benchmark_Sharpe": round(float(sharpe_bnh), 3),
            "Strategy_Sortino_Ratio": round(float(sortino_strat), 3),
            "Buy_Hold_Benchmark_Sortino": round(float(sortino_bnh), 3),
            "Strategy_Max_Drawdown_pct": round(float(max_dd_strat * 100), 2),
            "Buy_Hold_Max_Drawdown_pct": round(float(max_dd_bnh * 100), 2),
            "Drawdown_Reduction_pct": round(float((max_dd_bnh - max_dd_strat) * 100), 2),
            "Total_Friction_Drag_pct": round(float(np.sum(cost_drag) * 100), 2),
            "Cash_Avoid_Days": int((positions == 0).sum()),
            "Active_Long_Days": int((positions == 1).sum()),
            "Hold_Deadband_Days": int((test_df['action_code'] == 'HOLD').sum())
        },
        "downside_risk": downside_stats,
        "test_df": test_df,
        "cum_strategy": cum_strategy,
        "cum_bnh": cum_bnh
    }

    return results


# =====================================================================
# 6. SELF-CONTAINED LOCAL EXPLAINABILITY (XAI) ENGINE
# =====================================================================

def explain_latest_decision(
    model, 
    scaler, 
    latest_raw_features: pd.Series, 
    feature_cols: list[str],
    latest_prob: float,
    latest_uncertainty: float,
    latest_natr: float
):
    """
    Computes local feature attribution explaining WHY the system issued its
    current decision (BUY, HOLD, or AVOID).
    """
    # Standardize sample using training scaler
    sample_df = pd.DataFrame([latest_raw_features[feature_cols]])
    scaled_vals = scaler.transform(sample_df)[0]
    global_importances = model.feature_importances_

    # Local attribution: scaled feature z-score deviation multiplied by feature importance
    attributions = []
    for col, z_val, imp in zip(feature_cols, scaled_vals, global_importances):
        local_impact = z_val * imp
        attributions.append({
            "feature": col,
            "raw_value": round(float(latest_raw_features[col]), 4),
            "z_score": round(float(z_val), 2),
            "impact_score": round(float(local_impact), 4),
            "direction": "Pushes BUY" if local_impact > 0 else "Pushes CASH/AVOID"
        })

    attributions.sort(key=lambda x: abs(x["impact_score"]), reverse=True)

    print("\n" + "="*70)
    print("EXECUTIVE DECISION & EXPLAINABILITY AUDIT (LATEST SESSION)")
    print("="*70)
    ticker_sym = latest_raw_features.get('Symbol', 'RELIANCE') if 'Symbol' in latest_raw_features else 'RELIANCE'
    print(f"Asset / Benchmark            : {ticker_sym} (Close: ₹{latest_raw_features['Close']:.2f})")
    print(f"Bullish Probability (Up)     : {latest_prob * 100:6.2f}%")
    print(f"Model Consensus / Confidence : {round((1.0 - latest_uncertainty)*100, 1):6.1f}% (Tree Agreement)")
    print(f"Market Volatility (NATR-14)  : {latest_natr:6.2f}% ({'Spike' if latest_natr > 3.0 else 'Safe'})")
    
    if latest_natr > 3.0:
        action = "CASH (PASS) - Volatility Spike Shield"
    elif latest_uncertainty > 0.22:
        action = "CASH (PASS) - High Model Disagreement"
    elif latest_prob >= 0.52:
        action = "BUY (LONG) - High Conviction & Controlled Volatility"
    else:
        action = "HOLD (NEUTRAL) - Low Edge Deadband"

    print(f"Recommendation               : >>> {action} <<<")
    print("-" * 70)
    print("Primary Decision Drivers:")
    for i, att in enumerate(attributions[:4], 1):
        print(f"  {i}. {att['feature']:16s} = {att['raw_value']:8.4f} | {att['direction']} (Impact: {att['impact_score']:+0.4f})")
    print("="*70)


# =====================================================================
# 7. VISUAL AUDIT DASHBOARD GENERATION
# =====================================================================

def render_audit_plots(results: dict, ticker: str = "RELIANCE"):
    """
    Generates high-resolution performance and risk charts.
    """
    test_df = results["test_df"]
    cum_strat = results["cum_strategy"]
    cum_bnh = results["cum_bnh"]
    dates = test_df['Date']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Top Plot: Cumulative Net Wealth (Strategy vs. Buy-and-Hold)
    ax1.plot(dates, cum_strat, label=f"Risk-Aware Strategy (Net of 8 bps friction)", color='#10b981', linewidth=2.0)
    ax1.plot(dates, cum_bnh, label=f"Buy & Hold {ticker} Benchmark", color='#94a3b8', linewidth=1.4, linestyle='--')
    ax1.set_title(f"{ticker} Risk-Aware Out-of-Sample Performance (Friction-Adjusted Backtest)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Growth of ₹1.00", fontsize=11)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Bottom Plot: Asset Allocation State (Long = 1, Cash = 0)
    signals = test_df['signal'].values
    ax2.fill_between(dates, signals, step='pre', color='#10b981', alpha=0.3, label="Long Equity (+1)")
    ax2.fill_between(dates, 1 - signals, step='pre', color='#f43f5e', alpha=0.15, label="AVOID / Cash (0)")
    ax2.set_ylabel("Allocation", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Cash", "Long"])
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = "sp500_refactored_performance.png"
    plt.savefig(chart_path, dpi=300)
    print(f"[5/6] Saved friction-adjusted backtest chart to: {chart_path}")
    plt.close()


# =====================================================================
# 8. MAIN PRODUCTION PIPELINE
# =====================================================================

def run_production_pipeline(csv_path: str = None, ticker: str = "RELIANCE"):
    """
    Main runner orchestrating the entire audited quant pipeline.
    """
    if csv_path is None:
        candidates = [
            f"NIFTY50_Data/{ticker}.csv",
            f"archive (1)/{ticker}.csv",
            "NIFTY50_Data/RELIANCE.csv",
            "archive (1)/RELIANCE.csv",
            f"{ticker}.csv",
            "RELIANCE.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    if csv_path is None:
        raise FileNotFoundError("Could not auto-detect CSV dataset. Provide via --csv argument.")

    # 1. Ingestion
    raw_df = load_canonical_data(csv_path=csv_path, ticker=ticker)

    # 2. Stationary Features (Separating historical supervised data and terminal inference bar)
    features_df, feature_cols, inference_bar = construct_stationary_features(raw_df)

    # 3. Model Training & Uncertainty on strictly historical supervised data
    model_bundle = train_risk_aware_model(features_df, feature_cols, split_ratio=0.80)

    # 4. Tripartite Backtest & Risk Evaluation on held-out test split
    results = execute_tripartite_decision_backtest(model_bundle, cost_bps=8.0)

    # 5. Visual Rendering
    render_audit_plots(results, ticker=ticker)

    # 6. Live Forward Prediction & Explainability for Terminal Market Session (T -> T+1)
    fwd_pred = predict_next_session(
        model_bundle["model"],
        model_bundle["scaler"],
        inference_bar,
        feature_cols
    )
    explain_latest_decision(
        model_bundle["model"],
        model_bundle["scaler"],
        inference_bar,
        feature_cols,
        fwd_pred["prob_up"],
        fwd_pred["uncertainty"],
        float(inference_bar["NATR_pct"])
    )

    # Print Institutional Summary Audit Table
    ml = results["ml_metrics"]
    fin = results["financial_metrics"]
    risk = results["downside_risk"]

    print("\n" + "="*70)
    print("EXECUTIVE AUDIT & PERFORMANCE SUMMARY (OUT-OF-SAMPLE)")
    print("="*70)
    print(f"Model Directional Accuracy (OOS)       : {ml['Accuracy_pct']:6.2f}% (B&H Benchmark: {ml['Buy_Hold_Benchmark_Acc_pct']}%)")
    print(f"Model F1-Score                         : {ml['F1_Score']:6.4f}")
    print(f"Strategy Net CAGR (Net of 8 bps drag)  : {fin['Strategy_Net_CAGR_pct']:6.2f}% (B&H Benchmark: {fin['Buy_Hold_Benchmark_CAGR_pct']}%)")
    print(f"Annualized Sharpe Ratio (Net)          : {fin['Strategy_Sharpe_Ratio']:6.3f}")
    print(f"Annualized Sortino Ratio (Net)         : {fin['Strategy_Sortino_Ratio']:6.3f}")
    print(f"Strategy Maximum Drawdown              : {fin['Strategy_Max_Drawdown_pct']:6.2f}%")
    print(f"Buy & Hold Maximum Drawdown            : {fin['Buy_Hold_Max_Drawdown_pct']:6.2f}%")
    print(f"Drawdown Shield (Capital Preserved)    : {fin['Drawdown_Reduction_pct']:+6.2f}% Loss Avoidance")
    print(f"Cumulative Friction Drag Deducted      : {fin['Total_Friction_Drag_pct']:6.2f}%")
    print(f"Portfolio Exposure Allocation          : {fin['Active_Long_Days']} Days Long | {fin['Cash_Avoid_Days']} Days Cash (PASS)")
    print("-" * 70)
    print("Downside Tail Risk (Test Split):")
    print(f"  - 1-Day Historical VaR (95%)         : {risk['Historical_VaR_95_pct']:6.2f}%")
    print(f"  - 1-Day Expected Shortfall (CVaR 95%): {risk['Historical_CVaR_95_pct']:6.2f}%")
    print(f"  - Annualized Realized Volatility     : {risk['Annualized_Vol_pct']:6.2f}%")
    print("="*70)
    print("[6/6] Execution and audit completed successfully.\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audited Production Quant ML Decision System")
    parser.add_argument("--csv", type=str, default=None, help="Path to stock CSV dataset")
    parser.add_argument("--ticker", type=str, default="RELIANCE", help="Stock ticker symbol (default: RELIANCE)")
    args = parser.parse_args()

    run_production_pipeline(csv_path=args.csv, ticker=args.ticker)
