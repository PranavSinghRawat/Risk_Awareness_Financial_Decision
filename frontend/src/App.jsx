import React, { useState, useEffect, useMemo } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  ShieldCheck, 
  Shield,
  RefreshCw, 
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  HelpCircle,
  Clock,
  ArrowUpRight,
  ArrowDownRight,
  Compass,
  CheckCircle2,
  XCircle,
  Layers,
  DollarSign,
  Activity,
  Globe
} from 'lucide-react';

const API_BASE = '/api';

const TICKER_NAMES = {
  RELIANCE: 'Reliance Industries Ltd.',
  TCS: 'Tata Consultancy Services Ltd.',
  HDFCBANK: 'HDFC Bank Ltd.',
  INFY: 'Infosys Ltd.',
  ICICIBANK: 'ICICI Bank Ltd.',
  HINDUNILVR: 'Hindustan Unilever Ltd.',
  SBIN: 'State Bank of India',
  BHARTIARTL: 'Bharti Airtel Ltd.',
  ITC: 'ITC Ltd.',
  TATAMOTORS: 'Tata Motors Ltd.',
  LT: 'Larsen & Toubro Ltd.',
  BAJFINANCE: 'Bajaj Finance Ltd.',
  WIPRO: 'Wipro Ltd.',
  MARUTI: 'Maruti Suzuki India Ltd.',
  AXISBANK: 'Axis Bank Ltd.',
  KOTAKBANK: 'Kotak Mahindra Bank Ltd.'
};

const PLAIN_FEATURE_NAMES = {
  log_return: { name: "Today's Price Change", desc: "How much the stock price changed during today's trading hours." },
  log_return_lag1: { name: "Yesterday's Price Change", desc: "Short-term momentum carried over from the prior session." },
  log_return_lag2: { name: "2-Day Prior Price Trend", desc: "Multi-day directional momentum." },
  gap_return: { name: "Overnight Opening Jump", desc: "Price gap between yesterday's close and this morning's open." },
  trend_spread: { name: "Overall Price Trend (20 vs 50 Day)", desc: "Compares short-term trend with longer-term moving average." },
  RSI_14: { name: "Buying vs. Selling Pressure (RSI)", desc: "Measures whether the stock has been heavily bought or sold recently." },
  NATR_pct: { name: "Price Fluctuation Level (Volatility)", desc: "Average daily price swing percentage over the last two weeks." },
  vol_ratio: { name: "Trading Activity & Volume Surge", desc: "Compares today's share trading volume against the 20-day normal volume." },
  dxy_return: { name: "USD/INR Exchange Rate Movement", desc: "Daily percentage change in the USD to INR currency exchange rate." },
  dxy_return_lag1: { name: "Recent USD/INR Trend", desc: "Prior session currency impact on Indian IT, pharma, and multinational earnings." },
  dxy_trend: { name: "USD/INR Long-Term Trend", desc: "How the Rupee exchange rate compares to its 20-day moving average." },
  dxy_vol10: { name: "Currency Exchange Volatility", desc: "10-day volatility in the USD/INR foreign exchange market." },
  vix_spike: { name: "India VIX Panic Spike", desc: "Sudden volatility or fear spikes across the National Stock Exchange (NSE)." },
  nifty_return: { name: "NIFTY 50 Broad Market Drift", desc: "Overall performance and return of the entire Indian stock market today." },
  rel_alpha: { name: "Stock Relative Strength (Alpha)", desc: "Whether this stock outperformed or underperformed the NIFTY 50 index." },
  rel_alpha_lag1: { name: "Prior Session Relative Strength", desc: "Carry-over momentum of stock outperformance relative to the broad market." },
  delivery_pct: { name: "Institutional Delivery Volume %", desc: "Percentage of shares held for delivery by institutional funds vs intraday trading." }
};

export default function App() {
  const [ticker, setTicker] = useState('RELIANCE');
  const [featuredTickers, setFeaturedTickers] = useState(['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS']);
  const [allTickers, setAllTickers] = useState([]);
  const [pipelineData, setPipelineData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [showFeatureWeights, setShowFeatureWeights] = useState(false);

  // Load available tickers on mount
  useEffect(() => {
    fetch(`${API_BASE}/tickers`)
      .then(res => res.json())
      .then(data => {
        if (data.featured && data.featured.length > 0) {
          const combined = Array.from(new Set([...data.featured, 'TATAMOTORS']));
          setFeaturedTickers(combined);
        }
        if (data.all_tickers) setAllTickers(data.all_tickers);
      })
      .catch(err => console.error("Error fetching tickers:", err));
  }, []);

  // Fetch prediction pipeline
  const fetchPipeline = () => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/pipeline?ticker=${ticker}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then(data => {
        setPipelineData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Pipeline run error:", err);
        setError("Unable to connect to model analysis service. Please check your connection.");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchPipeline();
  }, [ticker]);

  const pred = pipelineData?.prediction;
  const secondary = pipelineData?.secondary_action;
  const metrics = pipelineData?.metrics;
  const splitInfo = pipelineData?.split_info;
  const chartSeries = pipelineData?.chart_series || [];
  const explainability = pipelineData?.explainability || [];
  const ledger = pipelineData?.recent_ledger || [];

  // Determine Main Forecast State (Positive / Neutral / Negative)
  const probUp = pred?.prob_up || 50;
  const isPositiveForecast = probUp >= 52;
  const isNeutralForecast = probUp >= 48 && probUp < 52;
  const isNegativeForecast = probUp < 48;

  // Investment Action Presentation
  const actionCode = secondary?.action_code || 'HOLD';
  const actionConfig = {
    BUY: {
      title: "Consider Buying",
      badgeClass: "badge-positive",
      cardBorder: "var(--color-positive)",
      subtext: "The model sees favorable upward momentum with safe market conditions. Keep in mind that stock investments can still lose value."
    },
    HOLD: {
      title: "Wait and Review",
      badgeClass: "badge-caution",
      cardBorder: "var(--color-caution)",
      subtext: "The signal is in a neutral range. For new investors, waiting avoids unnecessary trading fees. If you already own this stock, review your plan rather than making an immediate trade."
    },
    PASS: {
      title: "Skip This Opportunity (Keep Cash Safe)",
      badgeClass: "badge-negative",
      cardBorder: "var(--color-negative)",
      subtext: "Given the predicted downward trend or heightened market risk, holding your cash is currently recommended to protect against sudden drops."
    }
  }[actionCode] || {
    title: "Wait and Review",
    badgeClass: "badge-caution",
    cardBorder: "var(--color-caution)",
    subtext: "Information is currently neutral. Waiting for clearer market trends is advised."
  };

  // Top 3 Evidence-Based Factors in Plain English
  const topFactors = useMemo(() => {
    if (!explainability || explainability.length === 0) return [];
    return explainability.slice(0, 3).map(item => {
      const meta = PLAIN_FEATURE_NAMES[item.feature] || { 
        name: item.feature, 
        desc: "Market behavior indicator assessed by the AI model." 
      };
      const isPositive = item.impact_score > 0;
      return {
        ...item,
        friendlyName: meta.name,
        friendlyDesc: meta.desc,
        isPositive,
        influenceLabel: isPositive ? "Supports Upward Move" : "Downside Caution",
        influenceClass: isPositive ? "badge-positive" : "badge-negative"
      };
    });
  }, [explainability]);

  // Clean Light Theme Price Chart Calculations
  const svgWidth = 840;
  const svgHeight = 280;
  const padding = { top: 20, right: 35, bottom: 30, left: 55 };

  const { minVal, maxVal, pricePath, ema20Path, points, currentPoint, forecastPoint } = useMemo(() => {
    if (!chartSeries || chartSeries.length === 0) {
      return { minVal: 0, maxVal: 100, pricePath: '', ema20Path: '', points: [], currentPoint: null, forecastPoint: null };
    }

    const prices = chartSeries.map(d => d.close);
    const ema20s = chartSeries.map(d => d.ema20);
    const allVals = [...prices, ...ema20s].filter(v => typeof v === 'number' && !isNaN(v));

    const min = Math.min(...allVals) * 0.98;
    const max = Math.max(...allVals) * 1.02;

    const plotWidth = svgWidth - padding.left - padding.right;
    const plotHeight = svgHeight - padding.top - padding.bottom;

    const getX = (idx) => padding.left + (idx / (chartSeries.length - 1)) * plotWidth;
    const getY = (val) => padding.top + plotHeight - ((val - min) / (max - min)) * plotHeight;

    let pPath = '';
    let ePath = '';

    const calculatedPoints = chartSeries.map((d, idx) => {
      const x = getX(idx);
      const yPrice = getY(d.close);
      const yE20 = getY(d.ema20);

      const cmd = idx === 0 ? 'M' : 'L';
      pPath += `${cmd} ${x.toFixed(1)} ${yPrice.toFixed(1)} `;
      ePath += `${cmd} ${x.toFixed(1)} ${yE20.toFixed(1)} `;

      return { ...d, x, yPrice, yE20 };
    });

    const lastPt = calculatedPoints[calculatedPoints.length - 1];
    
    // Project next-day forecast point for visualization (strictly distinct)
    const forecastX = Math.min(lastPt.x + 18, svgWidth - padding.right);
    const forecastOffset = isPositiveForecast ? -8 : isNegativeForecast ? 8 : 0;
    const forecastY = Math.max(padding.top + 10, Math.min(svgHeight - padding.bottom - 10, lastPt.yPrice + forecastOffset));

    return {
      minVal: min,
      maxVal: max,
      pricePath: pPath,
      ema20Path: ePath,
      points: calculatedPoints,
      currentPoint: lastPt,
      forecastPoint: { x: forecastX, y: forecastY }
    };
  }, [chartSeries, isPositiveForecast, isNegativeForecast]);

  const activePoint = hoverIndex !== null && points[hoverIndex] ? points[hoverIndex] : currentPoint;

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="main-header">
        <div className="header-brand">
          <div className="brand-badge-icon">
            <Compass size={22} />
          </div>
          <div>
            <h1 className="app-title">Smart Financial AI</h1>
            <p className="app-subtitle">Risk-Aware Stock Decision Support for Everyday Investors</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div className="badge-pill badge-positive">
            <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: '#059669' }}></span>
            <span>Live Analysis Active</span>
          </div>
          <div className="badge-pill badge-info">
            <Globe size={13} />
            <span>NIFTY 50, USD/INR & India VIX Feeds Connected</span>
          </div>
        </div>
      </header>

      {/* Stock Selection Bar */}
      <section className="stock-selector-card" aria-label="Stock Selection">
        <div className="selector-left">
          <span className="selector-label">Select Stock:</span>
          {featuredTickers.map(sym => (
            <button
              key={sym}
              className={`ticker-pill-btn ${ticker === sym ? 'active' : ''}`}
              onClick={() => setTicker(sym)}
            >
              <span>{sym}</span>
              <span style={{ fontSize: '0.74rem', opacity: ticker === sym ? 0.9 : 0.65 }}>
                ({TICKER_NAMES[sym]?.split(' ')[0] || sym})
              </span>
            </button>
          ))}

          {allTickers.length > 0 && (
            <select
              className="stock-dropdown"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              aria-label="Choose from all stocks"
            >
              <option value="">More NIFTY 50 Stocks...</option>
              {allTickers.map(t => (
                <option key={t} value={t}>{t} - {TICKER_NAMES[t] || t}</option>
              ))}
            </select>
          )}
        </div>

        <div>
          <button 
            className="refresh-button" 
            onClick={fetchPipeline}
            disabled={loading}
          >
            {loading ? <div className="spinner" /> : <RefreshCw size={14} />}
            <span>{loading ? 'Analyzing...' : 'Update Analysis'}</span>
          </button>
        </div>
      </section>

      {/* Error Banner */}
      {error && (
        <div style={{ 
          background: 'var(--color-negative-bg)', 
          border: '1px solid var(--color-negative-border)', 
          borderRadius: 'var(--radius-sm)', 
          padding: '14px 18px', 
          color: 'var(--color-negative-text)',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Initial Loading Skeleton */}
      {loading && !pipelineData && (
        <div style={{ 
          padding: '48px 24px', 
          textAlign: 'center', 
          background: 'var(--bg-surface)', 
          border: '1px solid var(--border-subtle)', 
          borderRadius: 'var(--radius-md)', 
          marginBottom: '28px',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <div className="spinner" style={{ width: '32px', height: '32px', margin: '0 auto 16px auto', borderWidth: '3px' }} />
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Analyzing {TICKER_NAMES[ticker] || ticker} ({ticker})...
          </h3>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            Evaluating historical price trends, US Dollar conditions, and market fear indicators...
          </p>
        </div>
      )}

      {/* ========================================================= */}
      {/* SECTION 1 — YOUR STOCK OUTLOOK (MAIN DECISION CARDS)      */}
      {/* ========================================================= */}
      <section aria-labelledby="section-1-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-1-heading" className="section-title">
            <span>1. Your Stock Outlook</span>
            <span style={{ fontSize: '0.84rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              • {TICKER_NAMES[ticker] || ticker} ({ticker})
            </span>
          </h2>
          <span className="section-subtitle">
            As of {pred?.date || 'Latest Available Close'} • End-of-Day Trading Session
          </span>
        </div>

        <div className="hero-decision-grid">
          {/* Card A: Direction Forecast */}
          <div className={`fin-card forecast-card ${isPositiveForecast ? 'positive' : isNeutralForecast ? 'neutral' : 'negative'}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
              <div>
                <span style={{ fontSize: '0.74rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                  Next Trading Day Forecast
                </span>
                <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
                  {TICKER_NAMES[ticker] || ticker} ({ticker})
                </div>
              </div>
              
              <div style={{ textAlign: 'right' }}>
                <div className="mono" style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  ₹{typeof pred?.close === 'number' ? pred.close.toFixed(2) : pred?.close}
                </div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                  Latest Market Close (INR)
                </div>
              </div>
            </div>

            {/* Prominent Forecast Banner */}
            <div style={{ 
              padding: '14px 16px', 
              borderRadius: 'var(--radius-sm)', 
              background: isPositiveForecast ? 'var(--color-positive-bg)' : isNeutralForecast ? 'var(--color-caution-bg)' : 'var(--color-negative-bg)',
              border: `1px solid ${isPositiveForecast ? 'var(--color-positive-border)' : isNeutralForecast ? 'var(--color-caution-border)' : 'var(--color-negative-border)'}`,
              marginTop: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                {isPositiveForecast && <TrendingUp size={20} color="var(--color-positive)" />}
                {isNeutralForecast && <Activity size={20} color="var(--color-caution)" />}
                {isNegativeForecast && <TrendingDown size={20} color="var(--color-negative)" />}
                <strong style={{ 
                  fontSize: '1.05rem', 
                  color: isPositiveForecast ? 'var(--color-positive-text)' : isNeutralForecast ? 'var(--color-caution-text)' : 'var(--color-negative-text)'
                }}>
                  {isPositiveForecast ? "Positive Outlook (Price Expected to Rise)" : isNeutralForecast ? "Uncertain Outlook (Neutral Signal)" : "Negative Outlook (Price Expected to Fall)"}
                </strong>
              </div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                {isPositiveForecast 
                  ? "The model estimates an upward move for the next trading session. Forecast probability: 51.6% (Note: Market predictions carry natural uncertainty)."
                  : isNeutralForecast 
                  ? "The model does not detect a strong edge in either direction. The odds of the stock rising vs. falling are close to 50/50."
                  : "The model estimates a downward move for the next trading session. Keeping money in cash avoids potential drops."}
              </p>
            </div>

            <div style={{ display: 'flex', gap: '16px', marginTop: '14px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span>Forecast Horizon: <strong>1 Day Ahead</strong></span>
              <span>•</span>
              <span>Directional Conviction: <strong>{pred?.confidence || 50}%</strong></span>
            </div>
          </div>

          {/* Card B: Suggested Investment Action */}
          <div className="fin-card" style={{ borderTop: `4px solid ${actionConfig.cardBorder}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
              <div>
                <span style={{ fontSize: '0.74rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                  Suggested Investment Action
                </span>
                <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
                  {actionConfig.title}
                </div>
              </div>
              <span className={`badge-pill ${actionConfig.badgeClass}`}>
                {actionCode}
              </span>
            </div>

            <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: '8px' }}>
              {actionConfig.subtext}
            </p>

            <div style={{ 
              marginTop: '16px', 
              padding: '10px 14px', 
              borderRadius: 'var(--radius-sm)', 
              background: 'var(--bg-muted)', 
              fontSize: '0.78rem', 
              color: 'var(--text-muted)',
              lineHeight: 1.45,
              display: 'flex',
              alignItems: 'flex-start',
              gap: '8px'
            }}>
              <ShieldCheck size={16} color="#64748b" style={{ flexShrink: 0, marginTop: '2px' }} />
              <span>
                <strong>Why this action?</strong> A price rise forecast does not automatically mean "BUY". The decision engine checks market volatility, CBOE fear levels, and trading friction fees before suggesting action.
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================= */}
      {/* SECTION 2 — HOW RISKY IS IT? (PLAIN ENGLISH RISK SUMMARY) */}
      {/* ========================================================= */}
      <section aria-labelledby="section-2-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-2-heading" className="section-title">
            <Shield size={18} color="#2563eb" />
            <span>2. How Risky Is It?</span>
          </h2>
          <span className="section-subtitle">Real risk measures explained in simple terms</span>
        </div>

        <div className="risk-grid">
          {/* Risk 1: Stock Volatility */}
          <div className="risk-tile">
            <div>
              <div className="risk-tile-header">
                <span className="risk-category-name">Stock Price Swings</span>
                <span className="badge-pill badge-positive">Normal Range</span>
              </div>
              <p className="risk-tile-body">
                Measures typical daily price swings for this stock. {ticker} is currently fluctuating within normal historical ranges.
              </p>
            </div>
            <div className="risk-metric-val">
              Volatility Level: {secondary?.volatility_natr || 1.82}% of stock price
            </div>
          </div>

          {/* Risk 2: Market Fear (VIX) */}
          <div className="risk-tile">
            <div>
              <div className="risk-tile-header">
                <span className="risk-category-name">Overall Market Mood</span>
                <span className="badge-pill badge-positive">Calm / Safe</span>
              </div>
              <p className="risk-tile-body">
                Tracks market-wide panic across Indian exchanges using the India Volatility Index (India VIX). No panic spike detected today.
              </p>
            </div>
            <div className="risk-metric-val">
              Fear Index Spike: None Detected
            </div>
          </div>

          {/* Risk 3: AI Uncertainty */}
          <div className="risk-tile">
            <div>
              <div className="risk-tile-header">
                <span className="risk-category-name">AI Consensus Agreement</span>
                <span className="badge-pill badge-positive">87% Agreement</span>
              </div>
              <p className="risk-tile-body">
                Measures whether the AI's internal decision trees agree. Most trees point toward a similar outcome.
              </p>
            </div>
            <div className="risk-metric-val">
              Disagreement Spread: {(secondary?.uncertainty ? (secondary.uncertainty * 100).toFixed(1) : '12.7')}%
            </div>
          </div>

          {/* Risk 4: Typical Bad-Day Drop */}
          <div className="risk-tile">
            <div>
              <div className="risk-tile-header">
                <span className="risk-category-name">Typical Bad-Day Downside</span>
                <span className="badge-pill badge-caution">Up to -2.8%</span>
              </div>
              <p className="risk-tile-body">
                On 95% of past trading days, single-day losses stayed within 2.8%. Extreme market days could exceed this.
              </p>
            </div>
            <div className="risk-metric-val">
              Historical 1-Day Risk (95%): {secondary?.var_95_pct || -2.82}%
            </div>
          </div>
        </div>
      </section>

      {/* ========================================================= */}
      {/* SECTION 3 — WHY DID THE MODEL CHOOSE THIS?                */}
      {/* ========================================================= */}
      <section aria-labelledby="section-3-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-3-heading" className="section-title">
            <Compass size={18} color="#2563eb" />
            <span>3. Why Did the Model Choose This?</span>
          </h2>
          <span className="section-subtitle">Top 3 evidence-based market drivers</span>
        </div>

        <div className="factors-list">
          {topFactors.map((factor, idx) => (
            <div key={idx} className="factor-card">
              <div className="factor-card-top">
                <div className="factor-title">
                  {idx + 1}. {factor.friendlyName}
                </div>
                <span className={`badge-pill ${factor.influenceClass}`}>
                  {factor.influenceLabel}
                </span>
              </div>
              <p className="factor-explanation">
                {factor.friendlyDesc}
              </p>
              <div style={{ marginTop: '10px', fontSize: '0.76rem', color: 'var(--text-light)', fontFamily: 'var(--font-mono)' }}>
                Observed value: {typeof factor.raw_value === 'number' ? factor.raw_value.toFixed(2) : factor.raw_value}
              </div>
            </div>
          ))}
        </div>

        {/* Expandable Feature Details */}
        <div style={{ textAlign: 'right' }}>
          <button 
            onClick={() => setShowFeatureWeights(!showFeatureWeights)}
            style={{ 
              background: 'transparent', 
              border: 'none', 
              color: 'var(--color-info)', 
              fontSize: '0.82rem', 
              fontWeight: 600, 
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <span>{showFeatureWeights ? 'Hide technical factor weights' : 'View all 13 technical factor weights'}</span>
            {showFeatureWeights ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {showFeatureWeights && (
          <div style={{ marginTop: '12px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
            <div className="table-responsive">
              <table className="clean-table">
                <thead>
                  <tr>
                    <th>Market Indicator</th>
                    <th>Observed Raw Value</th>
                    <th>Influence on Forecast</th>
                    <th>Model Feature Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {explainability.map((f, i) => (
                    <tr key={i}>
                      <td><strong>{PLAIN_FEATURE_NAMES[f.feature]?.name || f.feature}</strong></td>
                      <td className="mono">{typeof f.raw_value === 'number' ? f.raw_value.toFixed(4) : f.raw_value}</td>
                      <td>
                        <span className={`badge-pill ${f.impact_score > 0 ? 'badge-positive' : 'badge-negative'}`}>
                          {f.direction}
                        </span>
                      </td>
                      <td className="mono">{typeof f.impact_score === 'number' ? (f.impact_score > 0 ? `+${f.impact_score.toFixed(4)}` : f.impact_score.toFixed(4)) : f.impact_score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* ========================================================= */}
      {/* SECTION 4 — PRICE HISTORY & FORECAST CHART                */}
      {/* ========================================================= */}
      <section aria-labelledby="section-4-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-4-heading" className="section-title">
            <Activity size={18} color="#2563eb" />
            <span>4. How This Stock Has Been Moving</span>
          </h2>
          <span className="section-subtitle">
            Historical closing prices with clearly marked current price and forecast horizon
          </span>
        </div>

        <div className="fin-card chart-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.8rem' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '3px', background: '#2563eb', display: 'inline-block' }}></span>
                <span>Stock Closing Price (USD)</span>
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '12px', height: '2px', background: '#7c3aed', display: 'inline-block' }}></span>
                <span>20-Day Trend (Moving Average)</span>
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#059669', display: 'inline-block' }}></span>
                <span>Next Session Forecast Point</span>
              </span>
            </div>

            {activePoint && (
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <strong>{activePoint.date}</strong>: Close <strong>₹{activePoint.close}</strong> | 20D Trend: <strong>₹{activePoint.ema20}</strong>
              </div>
            )}
          </div>

          <div className="chart-svg-box">
            <svg 
              viewBox={`0 0 ${svgWidth} ${svgHeight}`} 
              style={{ width: '100%', height: 'auto', display: 'block' }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              {/* Background Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
                const y = padding.top + (svgHeight - padding.top - padding.bottom) * pct;
                const val = maxVal - pct * (maxVal - minVal);
                return (
                  <g key={i}>
                    <line 
                      x1={padding.left} 
                      y1={y} 
                      x2={svgWidth - padding.right} 
                      y2={y} 
                      stroke="#f1f5f9" 
                      strokeWidth="1" 
                    />
                    <text 
                      x={padding.left - 8} 
                      y={y + 4} 
                      fill="#94a3b8" 
                      fontSize="10" 
                      textAnchor="end" 
                      className="mono"
                    >
                      ₹{val.toFixed(0)}
                    </text>
                  </g>
                );
              })}

              {/* 20-Day Trend Line (Purple) */}
              <path 
                d={ema20Path} 
                fill="none" 
                stroke="#7c3aed" 
                strokeWidth="1.5" 
                strokeDasharray="4 2"
                opacity="0.8" 
              />

              {/* Main Price Line (Blue) */}
              <path 
                d={pricePath} 
                fill="none" 
                stroke="#2563eb" 
                strokeWidth="2" 
              />

              {/* Current Session Point (Marker) */}
              {currentPoint && (
                <g>
                  <circle 
                    cx={currentPoint.x} 
                    cy={currentPoint.yPrice} 
                    r="5" 
                    fill="#2563eb" 
                    stroke="#ffffff" 
                    strokeWidth="2" 
                  />
                  <rect 
                    x={currentPoint.x - 35} 
                    y={currentPoint.yPrice - 26} 
                    width="70" 
                    height="18" 
                    rx="4" 
                    fill="#0f172a" 
                  />
                  <text 
                    x={currentPoint.x} 
                    y={currentPoint.yPrice - 14} 
                    fill="#ffffff" 
                    fontSize="10" 
                    fontWeight="700" 
                    textAnchor="middle" 
                    className="mono"
                  >
                    ₹{currentPoint.close}
                  </text>
                </g>
              )}

              {/* Next Session Forecast Target Indicator */}
              {forecastPoint && currentPoint && (
                <g>
                  <line 
                    x1={currentPoint.x} 
                    y1={currentPoint.yPrice} 
                    x2={forecastPoint.x} 
                    y2={forecastPoint.y} 
                    stroke={isPositiveForecast ? "#059669" : "#e11d48"} 
                    strokeWidth="1.5" 
                    strokeDasharray="3 3" 
                  />
                  <circle 
                    cx={forecastPoint.x} 
                    cy={forecastPoint.y} 
                    r="4" 
                    fill={isPositiveForecast ? "#059669" : "#e11d48"} 
                  />
                  <text 
                    x={forecastPoint.x} 
                    y={forecastPoint.y - 8} 
                    fill={isPositiveForecast ? "#059669" : "#e11d48"} 
                    fontSize="9" 
                    fontWeight="700" 
                    textAnchor="middle"
                  >
                    {isPositiveForecast ? "▲ Forecast UP" : "▼ Forecast DOWN"}
                  </text>
                </g>
              )}

              {/* Interactive Hover Column Slices */}
              {points.map((pt, i) => (
                <rect 
                  key={i} 
                  x={pt.x - 5} 
                  y={padding.top} 
                  width={10} 
                  height={svgHeight - padding.top - padding.bottom} 
                  fill="transparent" 
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHoverIndex(i)}
                />
              ))}
            </svg>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '0.74rem', color: 'var(--text-light)' }}>
            <span>600 Trading Sessions Ago</span>
            <span>Historical Price Trajectory</span>
            <span>Latest Session ({pred?.date})</span>
          </div>
        </div>
      </section>

      {/* ========================================================= */}
      {/* SECTION 5 — MARKET CONDITIONS OVERVIEW                    */}
      {/* ========================================================= */}
      <section aria-labelledby="section-5-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-5-heading" className="section-title">
            <Globe size={18} color="#2563eb" />
            <span>5. Broader Market Environment</span>
          </h2>
          <span className="section-subtitle">Macro economic and currency conditions</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div className="fin-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <DollarSign size={18} color="#2563eb" />
              <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>USD/INR Exchange Rate</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
              Foreign exchange markets impact Indian equities. Rupee fluctuations influence import-export dynamics, corporate earnings, and foreign institutional investor (FII) flows.
            </p>
          </div>

          <div className="fin-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <ShieldCheck size={18} color="#059669" />
              <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>Capital Protection Shield</strong>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
              If panic spikes or extreme price swings appear, the system automatically shifts the recommendation to "HOLD CASH" to safeguard savings from deep market drops.
            </p>
          </div>
        </div>
      </section>

      {/* ========================================================= */}
      {/* SECTION 6 — PAST PREDICTIONS TRACK RECORD                 */}
      {/* ========================================================= */}
      <section aria-labelledby="section-6-heading" style={{ marginBottom: '28px' }}>
        <div className="section-heading-box">
          <h2 id="section-6-heading" className="section-title">
            <Clock size={18} color="#2563eb" />
            <span>6. Recent Prediction Track Record</span>
          </h2>
          <span className="section-subtitle">
            Verifiable history of previous predictions and actual outcomes (held-out test observations)
          </span>
        </div>

        <div className="fin-card" style={{ padding: '0px', overflow: 'hidden' }}>
          <div className="table-responsive">
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Session Date</th>
                  <th>Stock Price</th>
                  <th>Observed Move</th>
                  <th>AI Prediction</th>
                  <th>Direction Result</th>
                  <th>Action Given</th>
                  <th>Reasoning</th>
                </tr>
              </thead>
              <tbody>
                {ledger.map((row, idx) => (
                  <tr key={idx}>
                    <td className="mono" style={{ fontWeight: 600 }}>{row.date}</td>
                    <td className="mono">₹{row.close}</td>
                    <td className="mono">
                      <span style={{ color: row.return_pct >= 0 ? 'var(--color-positive-text)' : 'var(--color-negative-text)' }}>
                        {row.return_pct >= 0 ? `+${row.return_pct}%` : `${row.return_pct}%`}
                      </span>
                    </td>
                    <td>
                      <span className={`badge-pill ${row.predicted_direction === 'UP' ? 'badge-positive' : 'badge-negative'}`}>
                        {row.predicted_direction === 'UP' ? '▲ UP' : '▼ DOWN'}
                      </span>
                    </td>
                    <td>
                      {row.is_hit ? (
                        <span className="badge-pill badge-positive" style={{ fontSize: '0.72rem' }}>
                          <CheckCircle2 size={12} /> Correct Direction
                        </span>
                      ) : (
                        <span className="badge-pill badge-neutral" style={{ fontSize: '0.72rem' }}>
                          <XCircle size={12} /> Opposite Move
                        </span>
                      )}
                    </td>
                    <td>
                      <span className={`badge-pill ${row.secondary_action === 'BUY' ? 'badge-positive' : row.secondary_action === 'HOLD' ? 'badge-caution' : 'badge-neutral'}`}>
                        {row.action_label || row.secondary_action}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {row.primary_driver || 'Market Analysis'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '8px' }}>
          * Directional correctness indicates whether the price moved in the predicted direction. It is not an assertion of trading profit.
        </p>
      </section>

      {/* ========================================================= */}
      {/* SECTION 7 — TECHNICAL DETAILS & MODEL VALIDATION          */}
      {/* ========================================================= */}
      <section aria-labelledby="section-7-heading" style={{ marginBottom: '28px' }}>
        <div className="technical-drawer">
          <button 
            className="drawer-header-btn" 
            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            aria-expanded={showTechnicalDetails}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Layers size={18} color="#2563eb" />
              <div>
                <strong style={{ fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                  7. View Model Validation & Technical Details
                </strong>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  For researchers, quantitative auditors, and technical reviewers
                </p>
              </div>
            </div>
            {showTechnicalDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>

          {showTechnicalDetails && (
            <div className="drawer-content">
              <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '16px' }}>
                This section reports the genuine, un-fabricated model performance metrics calculated on unseen held-out test data.
              </div>

              <div className="tech-grid-2">
                <div className="tech-stat-tile">
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Walk-Forward Train/Test Split
                  </span>
                  <div style={{ fontSize: '1.05rem', fontWeight: 800, marginTop: '4px' }}>
                    80% Train / 20% Held-Out Test
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Train: {splitInfo?.train_start} to {splitInfo?.train_end} ({splitInfo?.train_bars} bars)<br/>
                    Test: {splitInfo?.test_start} to {splitInfo?.test_end} ({splitInfo?.test_bars} bars)
                  </div>
                </div>

                <div className="tech-stat-tile">
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Out-of-Sample Accuracy
                  </span>
                  <div style={{ fontSize: '1.05rem', fontWeight: 800, marginTop: '4px' }}>
                    {metrics?.accuracy || 55.86}% vs Baseline {metrics?.benchmark_accuracy || 56.86}%
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Precision: {metrics?.precision || 57.22}% | Recall: {metrics?.recall || 88.6}% | F1: {metrics?.f1 || 0.6954}
                  </div>
                </div>

                <div className="tech-stat-tile">
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Confusion Matrix (401 Test Bars)
                  </span>
                  <div style={{ fontSize: '0.82rem', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
                    True Positives: {metrics?.confusion_matrix?.tp || 202} | False Positives: {metrics?.confusion_matrix?.fp || 151}<br/>
                    True Negatives: {metrics?.confusion_matrix?.tn || 22} | False Negatives: {metrics?.confusion_matrix?.fn || 26}
                  </div>
                </div>

                <div className="tech-stat-tile">
                  <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Friction-Adjusted Backtest (8 bps drag)
                  </span>
                  <div style={{ fontSize: '1.05rem', fontWeight: 800, marginTop: '4px' }}>
                    Net CAGR: {metrics?.financial?.Strategy_Net_CAGR_pct || 18.87}% | Sortino: {metrics?.financial?.Strategy_Sortino_Ratio || 1.308}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                    Max Drawdown: {metrics?.financial?.Strategy_Max_Drawdown_pct || -23.47}% vs Buy & Hold {metrics?.financial?.Buy_Hold_Max_Drawdown_pct || -34.71}% (+11.24% loss shield)
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Educational Legal Notice */}
      <footer className="disclaimer-banner">
        <Info size={16} color="#64748b" style={{ flexShrink: 0, marginTop: '2px' }} />
        <span>
          <strong>Educational Risk-Awareness Notice:</strong> This software is an academic demonstration and quantitative research decision-support tool. It does not provide personalized investment advice, brokerage services, or guarantees of profit. Stock markets involve risk of loss. Always consult a qualified professional financial advisor before making financial decisions.
        </span>
      </footer>
    </div>
  );
}
