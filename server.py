"""
Quantitative Financial ML FastAPI Backend Server
=================================================
Connects the S&P 500 Directional ML Pipeline to the React Frontend.
Provides endpoints for dynamic ticker modeling, chronological evaluation,
confusion matrix statistics, feature importances, and next-day forecasting.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import core quant modules from existing prototype
import sp500_directional_model as quant_engine

app = FastAPI(
    title="QuantML S&P 500 Directional Prediction Engine",
    description="Institutional-grade quantitative machine learning API for trend direction and directional accuracy.",
    version="1.0.0"
)

# Enable CORS for frontend Vite development server and production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "/Users/pranavrawat/Desktop/FinalYear/SP500_Data_10Y"


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "quant-ml-api", "version": "1.0.0"}


@app.get("/api/tickers")
def get_available_tickers():
    """
    Returns popular S&P 500 liquid tickers plus all available stocks in the dataset.
    """
    featured = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "AMD", "V"]
    available_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_tickers = sorted([os.path.splitext(os.path.basename(f))[0] for f in available_files])

    return {
        "featured": [t for t in featured if t in all_tickers],
        "all_tickers": all_tickers[:100],  # provide first 100 for fast selection
        "total_count": len(all_tickers)
    }


@app.get("/api/pipeline")
def run_quant_pipeline(
    ticker: str = Query("AAPL", description="S&P 500 Stock Ticker"),
    split_ratio: float = Query(0.80, ge=0.5, le=0.95, description="Chronological Train/Test Split Ratio"),
    n_estimators: int = Query(100, ge=10, le=500, description="Random Forest Trees"),
    max_depth: int = Query(5, ge=2, le=20, description="Random Forest Max Depth")
):
    """
    Executes the full quantitative ML pipeline:
    1. Loads dataset
    2. Calculates log returns, 20/50 EMAs, trend spread %, momentum lags, 14-day Wilder's RSI
    3. Executes strict 80/20 chronological split
    4. Trains Random Forest and evaluates out-of-sample directional accuracy
    5. Returns rich data for interactive charts, confusion matrix, and next-day forecast
    """
    ticker = ticker.upper()
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(csv_path):
        # Fallback to AAPL if requested ticker file isn't found
        csv_path = os.path.join(DATA_DIR, "AAPL.csv")
        ticker = "AAPL"

    try:
        # 1. Load data
        df = quant_engine.load_and_preprocess_data(csv_path=csv_path, ticker=ticker)

        # 2. Features and target
        features_df, feature_cols = quant_engine.compute_features_and_target(df)

        # Save the very latest row for Tomorrow's live prediction (before target dropping)
        # Note: In features_df, the last row had its target dropped because tomorrow hasn't happened yet.
        # Let's compute today's live feature vector from df
        df_live = df.copy()
        df_live['log_return'] = np.log(df_live['Close'] / df_live['Close'].shift(1))
        df_live['EMA_20'] = df_live['Close'].ewm(span=20, adjust=False).mean()
        df_live['EMA_50'] = df_live['Close'].ewm(span=50, adjust=False).mean()
        df_live['trend_spread'] = ((df_live['EMA_20'] - df_live['EMA_50']) / df_live['EMA_50']) * 100.0
        df_live['log_return_lag1'] = df_live['log_return'].shift(1)
        df_live['log_return_lag2'] = df_live['log_return'].shift(2)

        delta = df_live['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df_live['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

        latest_record = df_live.iloc[-1]
        latest_features = np.array([[
            latest_record['log_return'],
            latest_record['log_return_lag1'],
            latest_record['log_return_lag2'],
            latest_record['trend_spread'],
            latest_record['RSI_14']
        ]])

        # 3. Chronological split
        train_df, test_df, X_train, y_train, X_test, y_test = quant_engine.chronological_train_test_split(
            features_df, feature_cols, split_ratio=split_ratio
        )

        # 4. Train Model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Predictions on test set
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        total_test = len(y_test)

        buy_and_hold_acc = float(y_test.mean())
        alpha_spread = float((acc - buy_and_hold_acc) * 100.0)

        # Feature importances
        importances = rf.feature_importances_
        feature_importance_list = [
            {
                "feature": col,
                "importance": float(imp),
                "percentage": round(float(imp * 100), 2)
            }
            for col, imp in zip(feature_cols, importances)
        ]
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

        # Tomorrow's Forecast using latest live bar
        latest_df = pd.DataFrame(latest_features, columns=feature_cols)
        if not latest_df.isna().any().any():
            tomorrow_pred_dir = int(rf.predict(latest_df)[0])
            tomorrow_proba_up = float(rf.predict_proba(latest_df)[0][1])
            tomorrow_proba_down = float(1.0 - tomorrow_proba_up)
        else:
            tomorrow_pred_dir = 1
            tomorrow_proba_up = 0.5
            tomorrow_proba_down = 0.5

        # Format historical chart data
        # To ensure snappy frontend rendering, downsample the chart if large or return formatted records
        chart_df = pd.concat([train_df, test_df]).sort_values('Date').reset_index(drop=True)
        split_date_str = str(test_df['Date'].iloc[0].date())

        # Select last 800 days for crisp interactive visualization
        vis_df = chart_df.iloc[-800:].copy()
        chart_series = []
        for _, row in vis_df.iterrows():
            d_str = str(row['Date'].date())
            chart_series.append({
                "date": d_str,
                "close": round(float(row['Close']), 2),
                "ema20": round(float(row['EMA_20']), 2),
                "ema50": round(float(row['EMA_50']), 2),
                "trend_spread": round(float(row['trend_spread']), 2),
                "rsi": round(float(row['RSI_14']), 1),
                "is_test": d_str >= split_date_str
            })

        # Recent test predictions table (last 30 out-of-sample days)
        test_inspect = test_df.tail(30).copy()
        test_indices = [test_df.index.get_loc(idx) for idx in test_inspect.index]
        recent_preds = []
        for loc, (_, r) in zip(test_indices, test_inspect.iterrows()):
            p_dir = int(y_pred[loc])
            act_dir = int(y_test.iloc[loc])
            p_conf = float(y_proba[loc]) if p_dir == 1 else float(1.0 - y_proba[loc])
            recent_preds.append({
                "date": str(r['Date'].date()),
                "close": round(float(r['Close']), 2),
                "return_pct": round(float(np.exp(r['log_return']) - 1.0) * 100, 2),
                "actual_direction": act_dir,
                "predicted_direction": p_dir,
                "confidence": round(p_conf * 100, 1),
                "is_correct": bool(p_dir == act_dir)
            })

        return {
            "ticker": ticker,
            "metadata": {
                "total_samples": len(features_df),
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "split_ratio": split_ratio,
                "train_start": str(train_df['Date'].min().date()),
                "train_end": str(train_df['Date'].max().date()),
                "test_start": str(test_df['Date'].min().date()),
                "test_end": str(test_df['Date'].max().date()),
                "latest_date": str(latest_record['Date'].date()),
                "latest_close": round(float(latest_record['Close']), 2),
            },
            "metrics": {
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1": round(f1 * 100, 2),
                "buy_and_hold_acc": round(buy_and_hold_acc * 100, 2),
                "alpha_spread": round(alpha_spread, 2),
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "total": total_test,
                    "tn_pct": round(tn / total_test * 100, 1),
                    "fp_pct": round(fp / total_test * 100, 1),
                    "fn_pct": round(fn / total_test * 100, 1),
                    "tp_pct": round(tp / total_test * 100, 1),
                }
            },
            "feature_importances": feature_importance_list,
            "latest_forecast": {
                "date": str(latest_record['Date'].date()),
                "close": round(float(latest_record['Close']), 2),
                "rsi_14": round(float(latest_record['RSI_14']), 1),
                "trend_spread": round(float(latest_record['trend_spread']), 2),
                "predicted_direction": tomorrow_pred_dir,
                "confidence_up": round(tomorrow_proba_up * 100, 1),
                "confidence_down": round(tomorrow_proba_down * 100, 1),
            },
            "chart_series": chart_series,
            "recent_predictions": recent_preds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("[+] Starting QuantML API Server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
