"""
Quantitative Financial ML FastAPI Backend Server
=================================================
PRIMARY FOCUS: Next-Day Directional Price Prediction (UP / DOWN)
SECONDARY FOCUS: Risk-Aware Decision Action (BUY / HOLD / PASS)
"""

import os
import re
import time
import glob
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import core quant modules
import sp500_directional_model as quant_engine

app = FastAPI(
    title="QuantML Directional Prediction & Decision API",
    description="Primary Next-Day Directional Prediction with Secondary Risk-Aware Allocation.",
    version="3.1.0"
)

# Explicit CORS Origins for institutional UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NIFTY50_Data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive (1)")

# In-Memory Cache to prevent redundant 1.5s training on repeated requests
PIPELINE_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
CACHE_TTL = 300  # 5 minutes


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "quant-ml-api",
        "version": "3.1.0",
        "cache_entries": len(PIPELINE_CACHE)
    }


@app.get("/api/tickers")
def get_available_tickers():
    featured = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS", "LT", "BAJFINANCE"]
    available_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_tickers = sorted([
        os.path.splitext(os.path.basename(f))[0]
        for f in available_files
        if not os.path.splitext(os.path.basename(f))[0].endswith("_all") and not os.path.splitext(os.path.basename(f))[0].endswith("metadata")
    ])

    return {
        "featured": [t for t in featured if t in all_tickers],
        "all_tickers": all_tickers[:100],
        "total_count": len(all_tickers)
    }


@app.get("/api/pipeline")
def run_quant_pipeline(
    ticker: str = Query("RELIANCE", description="Stock Ticker Symbol"),
    split_ratio: float = Query(0.80, ge=0.5, le=0.95, description="Train/Test Split"),
    hurdle_prob: float = Query(0.52, ge=0.50, le=0.65, description="Buy Conviction Hurdle"),
    vol_cap_natr: float = Query(3.0, ge=1.5, le=6.0, description="Max Volatility Cap (NATR %)"),
    cost_bps: float = Query(8.0, ge=0.0, le=50.0, description="Transaction Friction (bps)")
):
    # 1. Input Sanitization: Strict regex check
    ticker_clean = ticker.strip().upper()
    if not re.match(r"^[A-Z0-9.-]{1,10}$", ticker_clean):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format: '{ticker}'. Must be 1-10 uppercase alphanumeric characters."
        )

    # 2. Dataset Existence Validation (NO silent fallback)
    csv_path = os.path.join(DATA_DIR, f"{ticker_clean}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"Stock ticker '{ticker_clean}' was not found in NIFTY 50 repository."
        )

    # 3. Cache Lookup
    cache_key = f"{ticker_clean}_{split_ratio}_{hurdle_prob}_{vol_cap_natr}_{cost_bps}"
    if cache_key in PIPELINE_CACHE:
        cached_result, cached_ts = PIPELINE_CACHE[cache_key]
        if time.time() - cached_ts < CACHE_TTL:
            return cached_result

    try:
        # 4. Ingestion & Minimum Data Size Validation
        raw_df = quant_engine.load_canonical_data(csv_path=csv_path, ticker=ticker_clean)
        if len(raw_df) < 200:
            raise HTTPException(
                status_code=422,
                detail=f"Stock ticker '{ticker_clean}' has insufficient historical data ({len(raw_df)} bars; minimum 200 required)."
            )

        # 5. Stationary Features & Terminal Inference Bar Separation
        features_df, feature_cols, inference_bar = quant_engine.construct_stationary_features(raw_df)

        # 6. Model Training on Historical Supervised Subset (Strict Chronological 80/20)
        model_bundle = quant_engine.train_risk_aware_model(features_df, feature_cols, split_ratio=split_ratio)

        # 7. Tripartite Backtest & Downside Risk on Held-Out Test Set
        backtest = quant_engine.execute_tripartite_decision_backtest(
            model_bundle,
            hurdle_prob=hurdle_prob,
            vol_cap_natr=vol_cap_natr,
            cost_bps=cost_bps
        )

        test_df = backtest["test_df"]
        y_test = model_bundle["y_test"]
        probs = model_bundle["mean_prob_up"]
        binary_preds = (probs >= 0.50).astype(int)

        # ML Classification Metrics on Held-out Out-of-Sample Period
        acc = accuracy_score(y_test, binary_preds)
        prec = precision_score(y_test, binary_preds, zero_division=0)
        rec = recall_score(y_test, binary_preds, zero_division=0)
        f1 = f1_score(y_test, binary_preds, zero_division=0)
        cm = confusion_matrix(y_test, binary_preds)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        total_test = len(y_test)
        bnh_acc = float(y_test.mean())

        # 8. PRIMARY: Live Forward Prediction for Terminal Market Session (T -> T+1)
        fwd_pred = quant_engine.predict_next_session(
            model=model_bundle["model"],
            scaler=model_bundle["scaler"],
            inference_bar=inference_bar,
            feature_cols=feature_cols
        )

        # 9. SECONDARY: Tripartite Risk-Aware Action Overlay (BUY / HOLD / PASS)
        train_df = model_bundle["train_df"]
        if "NATR_pct" in train_df.columns:
            asset_vol_cap = max(float(np.percentile(train_df["NATR_pct"].dropna(), 90)), 2.5)
        else:
            asset_vol_cap = 3.5

        has_vix_spike = bool('vix_spike' in inference_bar and inference_bar['vix_spike'] == 1)

        decision = quant_engine.determine_decision_action(
            prob_up=fwd_pred["prob_up"],
            uncertainty=fwd_pred["uncertainty"],
            natr_pct=float(inference_bar["NATR_pct"]),
            has_vix_spike=has_vix_spike,
            vol_cap=asset_vol_cap,
            uncertainty_cap=0.22,
            hurdle_prob=hurdle_prob,
            lower_hurdle=0.48
        )

        # 10. PRIMARY CHART: Price Action with 20-day & 50-day EMA Trend Overlay
        chart_df = pd.concat([features_df, pd.DataFrame([inference_bar])], ignore_index=True)
        split_date_str = str(test_df['Date'].iloc[0].date())
        vis_df = chart_df.iloc[-600:].copy()

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
                "natr": round(float(row['NATR_pct']), 2),
                "is_test": d_str >= split_date_str
            })

        # 11. Local Explainability (XAI) for Terminal Session
        sample_df = pd.DataFrame([inference_bar[feature_cols]])
        scaled_vals = model_bundle["scaler"].transform(sample_df)[0]
        importances = model_bundle["model"].feature_importances_

        xai_factors = []
        for col, z_val, imp in zip(feature_cols, scaled_vals, importances):
            impact = float(z_val * imp)
            xai_factors.append({
                "feature": col,
                "raw_value": round(float(inference_bar[col]), 4),
                "z_score": round(float(z_val), 2),
                "impact_score": round(impact, 4),
                "direction": "Pushes UP" if impact > 0 else "Pushes DOWN"
            })
        xai_factors.sort(key=lambda x: abs(x["impact_score"]), reverse=True)

        # 12. Historical Predictions Ledger (Last 25 Days of Held-out Test Set)
        recent_ledger = []
        tail_df = test_df.iloc[-25:].copy()
        for idx, row in tail_df.iterrows():
            loc = test_df.index.get_loc(idx)
            act_dir = int(y_test[loc])
            p_dir = int(binary_preds[loc])
            p_conf = round(float(probs[loc]) * 100, 1) if p_dir == 1 else round((1.0 - float(probs[loc])) * 100, 1)
            is_hit = bool(act_dir == p_dir)
            pos = int(row['signal'])
            act_code = row.get('action_code', 'BUY' if pos == 1 else 'PASS')

            if act_code == 'BUY':
                action_label = "BUY (INVEST)"
                if act_dir == 1:
                    outcome_status = "PROFIT GAINED"
                    outcome_desc = "Stock rose as predicted"
                else:
                    outcome_status = "MANAGED RISK"
                    outcome_desc = "Normal price fluctuation"
                primary_reason = "Strong Bullish Trend"
            elif act_code == 'HOLD':
                action_label = "HOLD (NEUTRAL)"
                outcome_status = "POSITION HELD"
                outcome_desc = "Neutral edge deadband; saved turnover friction"
                primary_reason = "Neutral Deadband [48%-52%]"
            else:  # PASS
                action_label = "HOLD CASH"
                if act_dir == 0:
                    outcome_status = "SAVED FROM LOSS"
                    outcome_desc = "Avoided stock drop by holding cash"
                    primary_reason = "Downside Drop Protected in Cash"
                else:
                    outcome_status = "MONEY PROTECTED"
                    outcome_desc = "Kept safe in cash during risk"
                    primary_reason = "Risk/Volatility Shield"

            recent_ledger.append({
                "date": str(row['Date'].date()),
                "close": round(float(row['Close']), 2),
                "return_pct": round(float(np.exp(row['log_return']) - 1.0) * 100, 2),
                "actual_direction": "UP" if act_dir == 1 else "DOWN",
                "predicted_direction": "UP" if p_dir == 1 else "DOWN",
                "is_hit": is_hit,
                "confidence": p_conf,
                "secondary_action": act_code,
                "action_label": action_label,
                "outcome_status": outcome_status,
                "outcome_desc": outcome_desc,
                "primary_driver": primary_reason
            })

        response_payload = {
            "ticker": ticker_clean,
            # PRIMARY OUTPUT: The Next-Day Price Prediction (Terminal Market Close -> T+1)
            "prediction": {
                "date": fwd_pred["date"],
                "close": round(fwd_pred["close"], 2),
                "direction": fwd_pred["direction"],
                "direction_code": fwd_pred["direction_code"],
                "prob_up": round(fwd_pred["prob_up"] * 100, 1),
                "prob_down": round(fwd_pred["prob_down"] * 100, 1),
                "confidence": fwd_pred["confidence"],
                "rsi_14": round(float(inference_bar["RSI_14"]), 1),
                "trend_spread": round(float(inference_bar["trend_spread"]), 2)
            },
            # SECONDARY OUTPUT: The Risk-Aware Action Overlay
            "secondary_action": {
                "action": decision["action_label"],
                "action_code": decision["action_code"],
                "description": decision["description"],
                "volatility_natr": round(float(inference_bar["NATR_pct"]), 2),
                "uncertainty": round(fwd_pred["uncertainty"], 4),
                "var_95_pct": backtest["downside_risk"]["Historical_VaR_95_pct"],
                "cvar_95_pct": backtest["downside_risk"]["Historical_CVaR_95_pct"]
            },
            # METRICS: Machine Learning Directional Accuracy & Confusion Matrix (Held-out Test Period)
            "metrics": {
                "accuracy": round(acc * 100, 2),
                "benchmark_accuracy": round(bnh_acc * 100, 2),
                "alpha_accuracy": round((acc - bnh_acc) * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1": round(f1, 4),
                "confusion_matrix": {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "total": total_test,
                    "tn_pct": round(tn / total_test * 100, 1),
                    "fp_pct": round(fp / total_test * 100, 1),
                    "fn_pct": round(fn / total_test * 100, 1),
                    "tp_pct": round(tp / total_test * 100, 1)
                },
                "financial": backtest["financial_metrics"]
            },
            "split_info": {
                "train_bars": len(train_df),
                "test_bars": len(test_df),
                "train_start": str(train_df['Date'].min().date()),
                "train_end": str(train_df['Date'].max().date()),
                "test_start": str(test_df['Date'].min().date()),
                "test_end": str(test_df['Date'].max().date()),
                "split_ratio": split_ratio
            },
            "dxy_integrated": any("dxy" in col for col in feature_cols),
            "vix_integrated": any("vix" in col for col in feature_cols),
            "feature_count": len(feature_cols),
            "explainability": xai_factors[:6],
            "chart_series": chart_series,
            "recent_ledger": recent_ledger
        }

        # Cache response
        if len(PIPELINE_CACHE) > 50:
            oldest_k = next(iter(PIPELINE_CACHE))
            PIPELINE_CACHE.pop(oldest_k, None)
        PIPELINE_CACHE[cache_key] = (response_payload, time.time())

        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("[+] Starting QuantML Primary Directional Prediction API on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

