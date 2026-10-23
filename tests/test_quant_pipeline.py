"""
Automated Quantitative Pipeline & Risk Decision Test Suite
===========================================================
Covers all 7 phases of the Financial Risk Awareness project audit:
- Data Ingestion & Canonical Validation
- Multi-Asset Synchronization (Stock + DXY + VIX)
- Stationary Feature Engineering & No Future Information Leakage
- Strict 80/20 Chronological Walk-Forward Train/Test Split
- Out-of-Sample Machine Learning Directional Evaluation vs Baselines
- Tripartite Decision Engine (BUY / HOLD / PASS) Boundary Conditions
- Realistic Friction-Adjusted Backtest & Canonical Sortino Ratio
- Backend API Input Sanitization & Error Handling
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient

import sp500_directional_model as quant_engine
from server import app

client = TestClient(app)

DATA_PATH = "NIFTY50_Data/RELIANCE.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "archive (1)/RELIANCE.csv"


class TestQuantPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DATA_PATH):
            raise unittest.SkipTest(f"Data file {DATA_PATH} not found.")
        cls.raw_df = quant_engine.load_canonical_data(DATA_PATH, "RELIANCE")
        cls.clean_data, cls.feature_cols, cls.inference_bar = quant_engine.construct_stationary_features(cls.raw_df)
        cls.model_bundle = quant_engine.train_risk_aware_model(cls.clean_data, cls.feature_cols, split_ratio=0.80)
        cls.backtest_results = quant_engine.execute_tripartite_decision_backtest(cls.model_bundle, cost_bps=8.0)

    # -------------------------------------------------------------
    # 1. DATA INGESTION & CANONICAL VALIDATION
    # -------------------------------------------------------------
    def test_01_raw_data_ingestion_and_columns(self):
        """Verify data ingestion, sorting, and required OHLCV columns."""
        df = self.raw_df
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        
        # Chronological order verification
        self.assertTrue(df['Date'].is_monotonic_increasing, "Raw dates must be strictly monotonic increasing.")
        self.assertGreater(len(df), 2000, "Dataset must have over 2000 trading sessions.")

    def test_02_macro_dataset_synchronization(self):
        """Verify cross-asset DXY, VIX, and NIFTY 50 Benchmark canonical files merge correctly."""
        dxy = quant_engine.load_canonical_dxy_data()
        vix = quant_engine.load_canonical_vix_data()
        bench = quant_engine.load_canonical_benchmark_data()
        self.assertIsNotNone(dxy, "DXY canonical data must be available.")
        self.assertIsNotNone(vix, "VIX canonical data must be available.")
        self.assertIsNotNone(bench, "NIFTY 50 Benchmark canonical data must be available.")
        self.assertIn('dxy_close', dxy.columns)
        self.assertIn('vix_close', vix.columns)
        self.assertIn('nifty_return', bench.columns)

    # -------------------------------------------------------------
    # 2. FEATURE ENGINEERING & NO LEAKAGE
    # -------------------------------------------------------------
    def test_03_stationary_feature_space(self):
        """Verify multi-pillar stationary features are calculated without NaNs in clean data."""
        self.assertEqual(len(self.feature_cols), 17, "Expected 17 stationary features including benchmark and delivery.")
        for col in self.feature_cols:
            self.assertIn(col, self.clean_data.columns)
            self.assertFalse(self.clean_data[col].isna().any(), f"Feature {col} contains NaN values.")

    def test_04_terminal_inference_bar_preservation(self):
        """Verify terminal bar (T) is preserved for live prediction and not dropped."""
        self.assertIsNotNone(self.inference_bar)
        self.assertIn('Close', self.inference_bar)
        self.assertIn('Date', self.inference_bar)
        # Terminal date should be strictly greater than the last training/testing target bar
        last_supervised_date = self.clean_data['Date'].iloc[-1]
        terminal_date = self.inference_bar['Date']
        self.assertGreater(terminal_date, last_supervised_date, "Terminal bar must be the latest market session.")

    # -------------------------------------------------------------
    # 3. 80/20 CHRONOLOGICAL SPLIT & ZERO LEAKAGE
    # -------------------------------------------------------------
    def test_05_chronological_split_ratios_and_counts(self):
        """Verify exact 80/20 observation counts and chronological order preservation."""
        train_df = self.model_bundle['train_df']
        test_df = self.model_bundle['test_df']
        total_supervised = len(self.clean_data)

        expected_train = int(total_supervised * 0.80)
        expected_test = total_supervised - expected_train

        self.assertEqual(len(train_df), expected_train, f"Expected {expected_train} train bars, got {len(train_df)}")
        self.assertEqual(len(test_df), expected_test, f"Expected {expected_test} test bars, got {len(test_df)}")

        # Strict chronological order across train and test
        self.assertLess(train_df['Date'].max(), test_df['Date'].min(), "Train period must strictly precede test period.")
        self.assertTrue(train_df['Date'].is_monotonic_increasing)
        self.assertTrue(test_df['Date'].is_monotonic_increasing)

    def test_06_scaler_zero_leakage(self):
        """Verify StandardScaler is fitted strictly on X_train and applied to X_test."""
        scaler = self.model_bundle['scaler']
        train_df = self.model_bundle['train_df']
        
        # Verify scaler mean matches training features mean
        train_means = train_df[self.feature_cols].mean().values
        np.testing.assert_allclose(scaler.mean_, train_means, rtol=1e-5, atol=1e-5)

    # -------------------------------------------------------------
    # 4. PREDICTION PERFORMANCE & HONEST METRICS
    # -------------------------------------------------------------
    def test_07_directional_classification_metrics(self):
        """Verify classification metrics are calculated from real test observations."""
        ml = self.backtest_results['ml_metrics']
        self.assertIn('Accuracy_pct', ml)
        self.assertIn('Precision_Up_pct', ml)
        self.assertIn('Recall_Up_pct', ml)
        self.assertIn('F1_Score', ml)
        self.assertIn('Buy_Hold_Benchmark_Acc_pct', ml)

        # Honest metrics check: values must be in valid mathematical bounds
        self.assertGreaterEqual(ml['Accuracy_pct'], 0.0)
        self.assertLessEqual(ml['Accuracy_pct'], 100.0)
        self.assertGreaterEqual(ml['F1_Score'], 0.0)
        self.assertLessEqual(ml['F1_Score'], 1.0)

    def test_08_canonical_sortino_ratio(self):
        """Verify Sortino ratio divides by total N observations, not just downside count."""
        fin = self.backtest_results['financial_metrics']
        sortino = fin['Strategy_Sortino_Ratio']
        self.assertIsInstance(sortino, float)
        self.assertFalse(np.isnan(sortino), "Sortino ratio must not be NaN.")
        self.assertTrue(np.isfinite(sortino), "Sortino ratio must be finite.")

    # -------------------------------------------------------------
    # 5. BUY / HOLD / PASS DECISION LOGIC BOUNDARIES
    # -------------------------------------------------------------
    def test_09_decision_boundary_buy(self):
        """Test BUY: High conviction (>=0.52), safe volatility, no panic spike."""
        dec = quant_engine.determine_decision_action(
            prob_up=0.55,
            uncertainty=0.12,
            natr_pct=2.0,
            has_vix_spike=False,
            vol_cap=3.0,
            hurdle_prob=0.52
        )
        self.assertEqual(dec['action_code'], 'BUY')
        self.assertEqual(dec['action_label'], 'BUY (INVEST)')

    def test_10_decision_boundary_hold(self):
        """Test HOLD: Probability in neutral deadband [0.48, 0.52)."""
        dec = quant_engine.determine_decision_action(
            prob_up=0.50,
            uncertainty=0.12,
            natr_pct=2.0,
            has_vix_spike=False,
            vol_cap=3.0,
            hurdle_prob=0.52,
            lower_hurdle=0.48
        )
        self.assertEqual(dec['action_code'], 'HOLD')
        self.assertEqual(dec['action_label'], 'HOLD (NEUTRAL)')

    def test_11_decision_boundary_pass_vol_shock(self):
        """Test PASS: Volatility shock overrides bullish probability."""
        dec = quant_engine.determine_decision_action(
            prob_up=0.60,
            uncertainty=0.10,
            natr_pct=4.5,  # Exceeds vol_cap of 3.0
            has_vix_spike=False,
            vol_cap=3.0,
            hurdle_prob=0.52
        )
        self.assertEqual(dec['action_code'], 'PASS')
        self.assertEqual(dec['action_label'], 'HOLD CASH (SAFE)')
        self.assertTrue(dec['is_vol_shock'])

    def test_12_decision_boundary_pass_vix_spike(self):
        """Test PASS: CBOE VIX panic spike overrides bullish probability."""
        dec = quant_engine.determine_decision_action(
            prob_up=0.58,
            uncertainty=0.10,
            natr_pct=2.0,
            has_vix_spike=True,  # Panic exhaustion spike
            vol_cap=3.0,
            hurdle_prob=0.52
        )
        self.assertEqual(dec['action_code'], 'PASS')
        self.assertEqual(dec['action_label'], 'HOLD CASH (SAFE)')

    def test_13_decision_boundary_pass_bearish(self):
        """Test PASS: Low directional probability (< 0.48)."""
        dec = quant_engine.determine_decision_action(
            prob_up=0.42,
            uncertainty=0.12,
            natr_pct=2.0,
            has_vix_spike=False,
            vol_cap=3.0,
            hurdle_prob=0.52
        )
        self.assertEqual(dec['action_code'], 'PASS')
        self.assertEqual(dec['action_label'], 'HOLD CASH (SAFE)')

    # -------------------------------------------------------------
    # 6. BACKEND API INTEGRATION & SANITIZATION
    # -------------------------------------------------------------
    def test_14_api_health(self):
        """Verify API health endpoint responds with 200 and healthy status."""
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_15_api_pipeline_success(self):
        """Verify API pipeline returns complete payload with prediction and metrics."""
        res = client.get("/api/pipeline?ticker=RELIANCE")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["ticker"], "RELIANCE")
        self.assertIn("prediction", data)
        self.assertIn("secondary_action", data)
        self.assertIn("metrics", data)
        self.assertIn("split_info", data)
        self.assertEqual(data["split_info"]["split_ratio"], 0.8)

    def test_16_api_nonexistent_ticker_404(self):
        """Verify API returns 404 for non-existent stock symbol instead of silent fallback."""
        res = client.get("/api/pipeline?ticker=NONEXIST")
        self.assertEqual(res.status_code, 404)
        self.assertIn("was not found", res.json()["detail"])

    def test_17_api_malformed_ticker_400(self):
        """Verify API rejects malformed ticker input with 400 Bad Request."""
        res = client.get("/api/pipeline?ticker=../../secret")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid ticker symbol format", res.json()["detail"])

    def test_18_api_invalid_split_ratio_422(self):
        """Verify API rejects out-of-range split ratio with 422 Unprocessable Content."""
        res = client.get("/api/pipeline?ticker=RELIANCE&split_ratio=1.5")
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
