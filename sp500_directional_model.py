"""
Quantitative Financial Machine Learning Prototype: S&P 500 Directional Prediction
===================================================================================
Author: Quantitative Financial ML Engineer
Stack:  Python, pandas, numpy, scikit-learn, matplotlib

Key Objectives:
1. Robust Data Loading & Preprocessing (multi-ticker / single-ticker support).
2. Quant Feature Engineering:
   - Daily Log Returns: r_t = ln(Close_t / Close_{t-1})
   - Trend Direction: 20-day EMA, 50-day EMA, and normalized Trend Spread %
   - Momentum Lags: return_t, return_{t-1}, return_{t-2}
   - 14-day Wilder's RSI (Relative Strength Index)
   - Target Variable: Binary next-day price direction (1 if tomorrow's return > 0, else 0)
3. Strict Chronological Split (first 80% train, remaining 20% test, no shuffling).
4. Model Training & Evaluation:
   - Baseline Random Forest Classifier
   - Directional Accuracy, Precision, Recall, F1-Score, and Buy-and-Hold Benchmark
5. Visualizations:
   - Price chart with 20-day and 50-day EMA trend overlay and split boundary
   - Annotated Confusion Matrix heatmap
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. DATA LOADING & PREPROCESSING
# =====================================================================

def load_and_preprocess_data(csv_path: str, ticker: str = "AAPL") -> pd.DataFrame:
    """
    Loads stock data from a CSV file, handles multiple Kaggle/yfinance formats,
    filters for a single liquid ticker if multi-asset, sorts chronologically,
    and cleans missing values.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file.
    ticker : str
        Target ticker symbol to filter for if the CSV contains multiple assets.

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with standardized columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    print(f"[+] Loading dataset from: {csv_path}")

    # Inspect first few lines to detect multi-level headers (e.g. yfinance format)
    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line_1 = f.readline()
        header_line_2 = f.readline()

    if "Ticker" in header_line_2 or "Price" in header_line_1:
        # Multi-index header format (e.g. row 0 = Price type, row 1 = Ticker, row 2 = Date)
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten to top-level column names (Close, High, Low, Open, Volume)
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        df = df.reset_index()
    else:
        # Standard single-row header format
        df = pd.read_csv(csv_path)

    # Standardize column names (strip whitespace and lower-case comparison)
    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Check for ticker/name column to filter for a single liquid stock
    ticker_col = None
    for candidate in ['Name', 'Symbol', 'Ticker', 'ticker', 'symbol', 'name']:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if ticker_col is not None:
        initial_rows = len(df)
        df = df[df[ticker_col].astype(str).str.upper() == ticker.upper()].copy()
        print(f"[+] Multi-ticker dataset detected: filtered for '{ticker}' ({len(df)} rows from {initial_rows}).")

    # Rename canonical OHLCV columns regardless of case
    rename_dict = {}
    for col in df.columns:
        c_lower = col.lower()
        if c_lower == 'date':
            rename_dict[col] = 'Date'
        elif c_lower == 'close':
            rename_dict[col] = 'Close'
        elif c_lower == 'open':
            rename_dict[col] = 'Open'
        elif c_lower == 'high':
            rename_dict[col] = 'High'
        elif c_lower == 'low':
            rename_dict[col] = 'Low'
        elif c_lower == 'volume':
            rename_dict[col] = 'Volume'

    df.rename(columns=rename_dict, inplace=True)

    required_cols = ['Date', 'Close']
    for req in required_cols:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' is missing from the dataset. Found: {list(df.columns)}")

    # Parse dates and sort chronologically (strictly ascending)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Cast numeric columns
    numeric_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values: forward-fill then backward-fill (standard in time-series)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    print(f"[+] Preprocessed {len(df)} trading days for {ticker} from {df['Date'].min().date()} to {df['Date'].max().date()}.")
    return df


# =====================================================================
# 2. FEATURE ENGINEERING (CORE TRANSFORMATIONS & TREND)
# =====================================================================

def compute_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Computes core quant features and binary next-day directional target:
    - Daily Log Returns: r_t = ln(Close_t / Close_{t-1})
    - Trend Direction: 20-day EMA, 50-day EMA, and normalized Trend Spread %
    - Momentum Lags: return_t, return_{t-1}, return_{t-2}
    - 14-day Wilder's RSI (Relative Strength Index)
    - Target: Binary next-day price direction (1 if tomorrow's return > 0, else 0)

    Returns:
    --------
    tuple[pd.DataFrame, list[str]]:
        Cleaned dataframe without NaNs, and list of feature column names.
    """
    data = df.copy()

    # 1. Daily Log Returns: r_t = ln(Close_t / Close_{t-1})
    # Log returns are additive across time and exhibit better statistical properties for ML.
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # 2. Trend Direction: 20-day EMA, 50-day EMA, and normalized Trend Spread
    # EMA smoothing multiplier: alpha = 2 / (span + 1)
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # Normalized Trend Spread (%): scale-invariant distance between fast and slow moving averages
    data['trend_spread'] = ((data['EMA_20'] - data['EMA_50']) / data['EMA_50']) * 100.0

    # 3. Momentum Lags: return_t, return_{t-1}, return_{t-2}
    # Note: 'log_return' is return_t (available at close of day t)
    data['log_return_lag1'] = data['log_return'].shift(1)  # return_{t-1}
    data['log_return_lag2'] = data['log_return'].shift(2)  # return_{t-2}

    # 4. 14-day Relative Strength Index (RSI) using Wilder's exponential smoothing
    price_diff = data['Close'].diff()
    gain = price_diff.clip(lower=0)
    loss = -price_diff.clip(upper=0)
    
    # Wilder's smoothing corresponds to ewm with alpha = 1 / 14
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    
    # Relative Strength (RS) with zero-division safeguard
    rs = avg_gain / (avg_loss + 1e-10)
    data['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    # 5. Target Variable: Binary next-day price direction
    # 1 if tomorrow's return > 0 (Up day), else 0 (Down/Flat day)
    tomorrow_log_return = data['log_return'].shift(-1)
    data['target'] = (tomorrow_log_return > 0).astype(int)

    feature_cols = [
        'log_return',       # Momentum (t)
        'log_return_lag1',  # Momentum (t-1)
        'log_return_lag2',  # Momentum (t-2)
        'trend_spread',     # Normalized 20/50 EMA Trend Spread (%)
        'RSI_14'            # 14-day RSI
    ]

    # Drop NaNs:
    # - Warmup period for EMA/RSI and lags (initial rows)
    # - Last row has NaN for tomorrow's return (future unknown)
    # Drop the last row explicitly because tomorrow_log_return is NaN
    valid_rows = data.dropna(subset=feature_cols).iloc[:-1].copy()

    print(f"[+] Feature engineering complete. Remaining valid samples: {len(valid_rows)}")
    print(f"[+] Features utilized: {feature_cols}")
    return valid_rows, feature_cols


# =====================================================================
# 3. STRICT 80/20 CHRONOLOGICAL SPLIT
# =====================================================================

def chronological_train_test_split(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    split_ratio: float = 0.80
):
    """
    Performs a strict chronological train/test split.
    Guarantees zero random shuffling to prevent future look-ahead data leakage.
    """
    n_total = len(df)
    split_idx = int(n_total * split_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df['target']

    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print("\n" + "="*60)
    print("CHRONOLOGICAL TRAIN/TEST SPLIT (80/20)")
    print("="*60)
    print(f"Train Set: {len(train_df):5d} days | {train_df['Date'].min().date()} -> {train_df['Date'].max().date()} ({split_ratio*100:.0f}%)")
    print(f"Test Set:  {len(test_df):5d} days | {test_df['Date'].min().date()} -> {test_df['Date'].max().date()} ({(1-split_ratio)*100:.0f}%)")
    print("="*60)

    return train_df, test_df, X_train, y_train, X_test, y_test


# =====================================================================
# 4. MODEL TRAINING & ACCURACY TESTING
# =====================================================================

def train_and_evaluate_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    feature_cols: list[str]
):
    """
    Trains a baseline Random Forest Classifier and evaluates out-of-sample
    directional performance against quantitative benchmarks.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    # In financial time-series, moderate regularization (max_depth, min_samples_split)
    # is critical to mitigate overfitting on market noise.
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    print("\n[+] Training Random Forest Classifier on training set...")
    rf.fit(X_train, y_train)

    # Out-of-sample directional predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Buy-and-Hold Benchmark:
    # A naive long-only strategy always predicts direction = 1 (Up).
    # Its directional accuracy in the test set is simply the proportion of Up days.
    buy_and_hold_acc = y_test.mean()

    print("\n" + "="*60)
    print("OUT-OF-SAMPLE MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Directional Accuracy:         {acc * 100:6.2f}%")
    print(f"Precision (Up):               {prec * 100:6.2f}%")
    print(f"Recall (Up):                  {rec * 100:6.2f}%")
    print(f"F1-Score:                     {f1 * 100:6.2f}%")
    print("-" * 60)
    print(f"Buy-and-Hold Benchmark Acc:   {buy_and_hold_acc * 100:6.2f}% (Naive Always-Long)")
    excess_acc = (acc - buy_and_hold_acc) * 100
    print(f"Directional Alpha (Spread):   {excess_acc:+6.2f}% vs. Buy-and-Hold")
    print("="*60)

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)'], digits=4))

    # Feature Importance analysis
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Feature Importances:")
    for idx in sorted_idx:
        print(f"  - {feature_cols[idx]:18s}: {importances[idx]:.4f}")

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'buy_and_hold_acc': buy_and_hold_acc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    return rf, metrics


# =====================================================================
# 5. VISUALIZATIONS
# =====================================================================

def plot_price_and_ema_trend(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    ticker: str = "AAPL",
    save_path: str = "sp500_price_ema_trend.png"
):
    """
    Plot 1: Stock price with the 20-day and 50-day EMA trend overlay
    and clear chronological train/test split boundary.
    """
    full_df = pd.concat([train_df, test_df]).sort_values('Date').reset_index(drop=True)
    split_date = test_df['Date'].iloc[0]

    plt.figure(figsize=(14, 7), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot Close price
    plt.plot(full_df['Date'], full_df['Close'], label=f"{ticker} Close Price", color='#1f77b4', linewidth=1.2, alpha=0.9)

    # Plot EMAs
    plt.plot(full_df['Date'], full_df['EMA_20'], label="20-day EMA (Fast Trend)", color='#ff7f0e', linewidth=1.5, linestyle='--')
    plt.plot(full_df['Date'], full_df['EMA_50'], label="50-day EMA (Slow Trend)", color='#2ca02c', linewidth=1.5, linestyle='-.')

    # Split marker
    plt.axvline(x=split_date, color='#d62728', linestyle=':', linewidth=2, label=f"80/20 Chronological Split ({split_date.date()})")

    # Shaded test area
    plt.axvspan(split_date, full_df['Date'].iloc[-1], color='#d62728', alpha=0.07, label="Out-of-Sample Test Window (20%)")

    plt.title(f"{ticker} Daily Price Action & Trend Indicators (20 & 50 EMA)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=11, labelpad=8)
    plt.ylabel("Price (USD)", fontsize=11, labelpad=8)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"[+] Saved Price & EMA Trend chart to: {save_path}")
    plt.show(block=False)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray, 
    metrics: dict,
    ticker: str = "AAPL",
    save_path: str = "sp500_confusion_matrix.png"
):
    """
    Plot 2: Confusion Matrix heatmap showing correct vs. incorrect directional predictions
    with cell counts, percentages, and financial outcome labels.
    """
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    ax.grid(False)

    # Color mesh using clean sequential colormap
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Custom annotations for each quadrant
    cell_metadata = [
        [(tn, "True Negative (TN)", f"{tn/total*100:.1f}%\nCorrect 'Down'"),
         (fp, "False Positive (FP)", f"{fp/total*100:.1f}%\nType I Error ('Up' when Down)")],
        [(fn, "False Negative (FN)", f"{fn/total*100:.1f}%\nType II Error ('Down' when Up)"),
         (tp, "True Positive (TP)", f"{tp/total*100:.1f}%\nCorrect 'Up'")]
    ]

    for i in range(2):
        for j in range(2):
            count, title, subtext = cell_metadata[i][j]
            # Use white text for dark cells, black text for light cells
            text_color = "white" if count > (total * 0.3) else "#111827"
            ax.text(j, i, f"{title}\n{count} ({subtext})", 
                    ha="center", va="center", color=text_color, 
                    fontsize=11, fontweight='semibold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Down (0)', 'Pred Up (1)'], fontsize=11)
    ax.set_yticklabels(['Actual Down (0)', 'Actual Up (1)'], fontsize=11)
    ax.set_xlabel("Predicted Direction", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("Actual Direction", fontsize=12, labelpad=10, fontweight='bold')

    acc = metrics['accuracy'] * 100
    bnh = metrics['buy_and_hold_acc'] * 100
    plt.title(f"{ticker} Out-of-Sample Confusion Matrix\nDirectional Accuracy: {acc:.2f}% | Buy-and-Hold: {bnh:.2f}%", 
              fontsize=13, fontweight='bold', pad=18)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[+] Saved Confusion Matrix heatmap to: {save_path}")
    plt.show(block=False)


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def run_pipeline(csv_path: str = None, ticker: str = "AAPL"):
    """
    Executes the end-to-end quantitative ML pipeline.
    """
    # Auto-detect default Kaggle dataset path if not provided
    if csv_path is None:
        default_candidates = [
            f"SP500_Data_10Y/{ticker}.csv",
            "SP500_Data_10Y/AAPL.csv",
            "all_stocks_5yr.csv",
            "AAPL.csv"
        ]
        for candidate in default_candidates:
            if os.path.exists(candidate):
                csv_path = candidate
                break

    if csv_path is None:
        raise FileNotFoundError(
            "Could not locate an S&P 500 CSV file automatically. "
            "Please provide the path via arguments, e.g.: python sp500_directional_model.py path/to/dataset.csv"
        )

    # 1. Loading & Preprocessing
    df = load_and_preprocess_data(csv_path=csv_path, ticker=ticker)

    # 2. Feature Engineering & Target Construction
    features_df, feature_cols = compute_features_and_target(df)

    # 3. Strict 80/20 Chronological Split
    train_df, test_df, X_train, y_train, X_test, y_test = chronological_train_test_split(
        features_df, feature_cols, split_ratio=0.80
    )

    # 4. Model Training & Out-of-Sample Evaluation
    model, metrics = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, feature_cols
    )

    # 5. Visualizations
    plot_price_and_ema_trend(train_df, test_df, ticker=ticker, save_path="sp500_price_ema_trend.png")
    plot_confusion_matrix_heatmap(metrics['confusion_matrix'], metrics, ticker=ticker, save_path="sp500_confusion_matrix.png")

    print("\n[✓] Pipeline execution finished successfully.")
    return model, metrics, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Daily Directional ML Prototype")
    parser.add_argument("--csv", type=str, default=None, help="Path to S&P 500 CSV dataset")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to model (default: AAPL)")
    args = parser.parse_args()

    run_pipeline(csv_path=args.csv, ticker=args.ticker)
