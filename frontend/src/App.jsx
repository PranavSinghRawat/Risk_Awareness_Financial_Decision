import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  BarChart3, 
  ShieldCheck, 
  Database, 
  Cpu, 
  RefreshCw, 
  Layers, 
  AlertCircle,
  Clock,
  Compass,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';

const API_BASE = '/api';

export default function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [featuredTickers, setFeaturedTickers] = useState(['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']);
  const [allTickers, setAllTickers] = useState([]);
  const [splitRatio, setSplitRatio] = useState(0.80);
  const [maxDepth, setMaxDepth] = useState(5);
  const [pipelineData, setPipelineData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [activeTab, setActiveTab] = useState('chart');

  // Load available tickers on mount
  useEffect(() => {
    fetch(`${API_BASE}/tickers`)
      .then(res => res.json())
      .then(data => {
        if (data.featured && data.featured.length > 0) setFeaturedTickers(data.featured);
        if (data.all_tickers) setAllTickers(data.all_tickers);
      })
      .catch(err => console.error("Error fetching tickers:", err));
  }, []);

  // Run pipeline when ticker, splitRatio, or maxDepth change
  const fetchPipeline = () => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/pipeline?ticker=${ticker}&split_ratio=${splitRatio}&max_depth=${maxDepth}`)
      .then(res => {
        if (!res.ok) throw new Error(`Server returned error ${res.status}`);
        return res.json();
      })
      .then(data => {
        setPipelineData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Pipeline run error:", err);
        setError("Failed to execute quantitative model pipeline. Please check backend connection.");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchPipeline();
  }, [ticker, splitRatio, maxDepth]);

  // Chart data calculations
  const chartSeries = pipelineData?.chart_series || [];
  const svgWidth = 840;
  const svgHeight = 360;
  const padding = { top: 20, right: 30, bottom: 40, left: 60 };

  const { minVal, maxVal, splitX, pricePath, ema20Path, ema50Path, points } = useMemo(() => {
    if (!chartSeries || chartSeries.length === 0) {
      return { minVal: 0, maxVal: 100, splitX: 0, pricePath: '', ema20Path: '', ema50Path: '', points: [] };
    }

    const prices = chartSeries.map(d => d.close);
    const ema20s = chartSeries.map(d => d.ema20);
    const ema50s = chartSeries.map(d => d.ema50);
    const allVals = [...prices, ...ema20s, ...ema50s].filter(v => typeof v === 'number' && !isNaN(v));

    const min = Math.min(...allVals) * 0.98;
    const max = Math.max(...allVals) * 1.02;

    const plotWidth = svgWidth - padding.left - padding.right;
    const plotHeight = svgHeight - padding.top - padding.bottom;

    const getX = (idx) => padding.left + (idx / (chartSeries.length - 1)) * plotWidth;
    const getY = (val) => padding.top + plotHeight - ((val - min) / (max - min)) * plotHeight;

    let pPath = '';
    let e20Path = '';
    let e50Path = '';
    let firstTestIdx = chartSeries.findIndex(d => d.is_test);

    const calculatedPoints = chartSeries.map((d, idx) => {
      const x = getX(idx);
      const yPrice = getY(d.close);
      const yE20 = getY(d.ema20);
      const yE50 = getY(d.ema50);

      const cmd = idx === 0 ? 'M' : 'L';
      pPath += `${cmd} ${x.toFixed(1)} ${yPrice.toFixed(1)} `;
      e20Path += `${cmd} ${x.toFixed(1)} ${yE20.toFixed(1)} `;
      e50Path += `${cmd} ${x.toFixed(1)} ${yE50.toFixed(1)} `;

      return { ...d, x, yPrice, yE20, yE50 };
    });

    const splitPos = firstTestIdx >= 0 ? getX(firstTestIdx) : plotWidth * 0.8 + padding.left;

    return {
      minVal: min,
      maxVal: max,
      splitX: splitPos,
      pricePath: pPath,
      ema20Path: e20Path,
      ema50Path: e50Path,
      points: calculatedPoints
    };
  }, [chartSeries]);

  const activePoint = hoverIndex !== null && points[hoverIndex] ? points[hoverIndex] : (points.length > 0 ? points[points.length - 1] : null);

  const forecast = pipelineData?.latest_forecast;
  const metrics = pipelineData?.metrics;
  const cm = metrics?.confusion_matrix;
  const metadata = pipelineData?.metadata;

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="top-nav">
        <div className="nav-brand">
          <div className="brand-icon-box">
            <Compass className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="brand-title">QuantML // AlphaEngine</h1>
            <div className="brand-tagline">S&P 500 Directional Prediction & Out-of-Sample Validation</div>
          </div>
        </div>

        <div className="nav-status">
          <div className="status-pill">
            <span className="status-dot"></span>
            <span>API Connected (Port 8000)</span>
          </div>
          <div className="status-pill" style={{ background: 'rgba(99, 102, 241, 0.1)', borderColor: 'rgba(99, 102, 241, 0.3)', color: '#818cf8' }}>
            <Cpu size={14} />
            <span>Random Forest Baseline</span>
          </div>
        </div>
      </header>

      {/* Control Bar: Tickers & Hyperparameters */}
      <section className="control-bar">
        <div className="ticker-quick-list">
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, marginRight: '4px' }}>
            TICKER:
          </span>
          {featuredTickers.map(sym => (
            <button
              key={sym}
              className={`quick-ticker-btn ${ticker === sym ? 'active' : ''}`}
              onClick={() => setTicker(sym)}
            >
              {sym}
            </button>
          ))}
          {allTickers.length > 0 && (
            <select 
              className="custom-select" 
              value={ticker} 
              onChange={(e) => setTicker(e.target.value)}
              title="Select from 500 S&P stocks"
            >
              <option value="">More Tickers ({allTickers.length})...</option>
              {allTickers.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          )}
        </div>

        <div className="controls-right">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>SPLIT:</span>
            <select 
              className="custom-select" 
              value={splitRatio} 
              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
            >
              <option value="0.70">70% / 30%</option>
              <option value="0.80">80% / 20% (Default)</option>
              <option value="0.85">85% / 15%</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>MAX DEPTH:</span>
            <select 
              className="custom-select" 
              value={maxDepth} 
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
            >
              <option value="3">Depth 3 (High Reg)</option>
              <option value="5">Depth 5 (Standard)</option>
              <option value="8">Depth 8 (Low Reg)</option>
            </select>
          </div>

          <button 
            className="run-btn" 
            onClick={fetchPipeline}
            disabled={loading}
          >
            {loading ? <div className="spinner" /> : <RefreshCw size={15} />}
            <span>{loading ? 'Retraining...' : 'Re-Run Model'}</span>
          </button>
        </div>
      </section>

      {error && (
        <div style={{ 
          background: 'rgba(244, 63, 94, 0.15)', 
          border: '1px solid rgba(244, 63, 94, 0.3)', 
          color: '#fb7185', 
          padding: '16px 20px', 
          borderRadius: 'var(--radius-md)', 
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Hero Metrics Row */}
      <section className="hero-grid">
        {/* Next-Day Signal Card */}
        <div className="quant-card forecast-card">
          <div className="forecast-header">
            <div>
              <div className="card-label">NEXT-DAY DIRECTIONAL FORECAST</div>
              <div className="card-subtext">Day Ahead: t+1 Binary Probability</div>
            </div>
            {forecast && (
              <span className={`forecast-badge ${forecast.predicted_direction === 1 ? 'up' : 'down'}`}>
                {forecast.predicted_direction === 1 ? (
                  <>
                    <TrendingUp size={16} />
                    <span>PREDICT UP</span>
                  </>
                ) : (
                  <>
                    <TrendingDown size={16} />
                    <span>PREDICT DOWN</span>
                  </>
                )}
              </span>
            )}
          </div>

          {forecast ? (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginTop: '6px' }}>
                <span className="card-value-large mono">
                  ${forecast.close}
                </span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  As of {forecast.date}
                </span>
              </div>

              <div className="confidence-bar-wrapper">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
                  <span style={{ color: '#34d399', fontWeight: 600 }}>Up Prob: {forecast.confidence_up}%</span>
                  <span style={{ color: '#fb7185', fontWeight: 600 }}>Down Prob: {forecast.confidence_down}%</span>
                </div>
                <div className="confidence-bar-track">
                  <div 
                    className={`confidence-bar-fill ${forecast.predicted_direction === 1 ? 'up' : 'down'}`}
                    style={{ width: `${forecast.predicted_direction === 1 ? forecast.confidence_up : forecast.confidence_down}%` }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', gap: '14px', marginTop: '14px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                <span>14-day RSI: <strong className="mono" style={{ color: forecast.rsi_14 > 70 ? '#f43f5e' : (forecast.rsi_14 < 30 ? '#10b981' : '#f8fafc') }}>{forecast.rsi_14}</strong></span>
                <span>Trend Spread: <strong className="mono" style={{ color: forecast.trend_spread > 0 ? '#10b981' : '#f43f5e' }}>{forecast.trend_spread > 0 ? '+' : ''}{forecast.trend_spread}%</strong></span>
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '16px' }}>Computing forecast...</div>
          )}
        </div>

        {/* Directional Accuracy Card */}
        <div className="quant-card">
          <div className="card-label">OUT-OF-SAMPLE ACCURACY</div>
          <div className="card-value-large mono" style={{ color: '#38bdf8' }}>
            {metrics ? `${metrics.accuracy}%` : '--'}
          </div>
          <div className="card-subtext" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>Benchmark: {metrics?.buy_and_hold_acc}%</span>
            {metrics && (
              <span style={{ 
                color: metrics.alpha_spread >= 0 ? '#10b981' : '#f43f5e', 
                fontWeight: 600,
                fontSize: '0.75rem',
                padding: '1px 6px',
                borderRadius: '4px',
                background: metrics.alpha_spread >= 0 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)'
              }}>
                {metrics.alpha_spread >= 0 ? `+${metrics.alpha_spread}%` : `${metrics.alpha_spread}%`} spread
              </span>
            )}
          </div>
          <div style={{ marginTop: '10px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Naive Buy-and-Hold accuracy equals test Up-day frequency.
          </div>
        </div>

        {/* Precision & Recall (Up) */}
        <div className="quant-card">
          <div className="card-label">PRECISION & RECALL (UP CLASS)</div>
          <div style={{ display: 'flex', gap: '20px', marginTop: '4px' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Precision</div>
              <div className="mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
                {metrics ? `${metrics.precision}%` : '--'}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Recall</div>
              <div className="mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
                {metrics ? `${metrics.recall}%` : '--'}
              </div>
            </div>
          </div>
          <div style={{ marginTop: '10px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            F1-Score: <strong className="mono" style={{ color: '#e2e8f0' }}>{metrics ? `${metrics.f1}%` : '--'}</strong>
          </div>
        </div>

        {/* Dataset & Backtest Scope */}
        <div className="quant-card">
          <div className="card-label">CHRONOLOGICAL SPLIT SCOPE</div>
          <div className="mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc', marginTop: '4px' }}>
            {metadata ? `${metadata.test_samples} Days` : '--'}
          </div>
          <div className="card-subtext" style={{ marginTop: '4px' }}>
            Out-of-sample window:
          </div>
          <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
            {metadata ? `${metadata.test_start} → ${metadata.test_end}` : '--'}
          </div>
          <div style={{ marginTop: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Train: {metadata ? metadata.train_samples : '--'} days (zero look-ahead leakage)
          </div>
        </div>
      </section>

      {/* Main Grid: Chart + Side Panel (Confusion Matrix & Features) */}
      <section className="dashboard-main-grid">
        {/* Interactive Price & EMA Trend Chart */}
        <div className="quant-card chart-container">
          <div className="chart-header">
            <div>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
                {ticker} Price Action & Trend Indicators (20 & 50 EMA)
              </h2>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Historical price path with fast/slow trend filter and out-of-sample chronological test partition
              </div>
            </div>

            <div className="chart-legend">
              <span className="legend-item">
                <span className="legend-line" style={{ background: '#38bdf8' }} />
                <span>Price</span>
              </span>
              <span className="legend-item">
                <span className="legend-line" style={{ background: '#f59e0b' }} />
                <span>EMA 20</span>
              </span>
              <span className="legend-item">
                <span className="legend-line" style={{ background: '#10b981' }} />
                <span>EMA 50</span>
              </span>
              <span className="legend-item">
                <span className="legend-line" style={{ background: '#f43f5e', borderTop: '2px dotted #f43f5e' }} />
                <span>Split Line</span>
              </span>
            </div>
          </div>

          {/* SVG Canvas Chart */}
          <div className="svg-chart-wrapper">
            {activePoint && (
              <div className="chart-tooltip">
                <div style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '2px' }}>{activePoint.date}</div>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <span>Price: <strong className="mono" style={{ color: '#38bdf8' }}>${activePoint.close}</strong></span>
                  <span>EMA20: <strong className="mono" style={{ color: '#f59e0b' }}>${activePoint.ema20}</strong></span>
                  <span>EMA50: <strong className="mono" style={{ color: '#10b981' }}>${activePoint.ema50}</strong></span>
                  <span>RSI: <strong className="mono">{activePoint.rsi}</strong></span>
                </div>
              </div>
            )}

            <svg 
              className="svg-chart" 
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              preserveAspectRatio="none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = ((e.clientX - rect.left) / rect.width) * svgWidth;
                const plotWidth = svgWidth - padding.left - padding.right;
                if (mouseX >= padding.left && mouseX <= svgWidth - padding.right && points.length > 0) {
                  const idx = Math.round(((mouseX - padding.left) / plotWidth) * (points.length - 1));
                  if (idx >= 0 && idx < points.length) setHoverIndex(idx);
                }
              }}
              onMouseLeave={() => setHoverIndex(null)}
            >
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.25" />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.0" />
                </linearGradient>
                <linearGradient id="testAreaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.12" />
                  <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.02" />
                </linearGradient>
              </defs>

              {/* Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
                const y = padding.top + (svgHeight - padding.top - padding.bottom) * pct;
                const priceLevel = (maxVal - pct * (maxVal - minVal)).toFixed(1);
                return (
                  <g key={i}>
                    <line 
                      x1={padding.left} 
                      y1={y} 
                      x2={svgWidth - padding.right} 
                      y2={y} 
                      stroke="rgba(255, 255, 255, 0.05)" 
                      strokeDasharray="4 4" 
                    />
                    <text 
                      x={padding.left - 8} 
                      y={y + 4} 
                      fill="var(--text-muted)" 
                      fontSize="10" 
                      fontFamily="JetBrains Mono" 
                      textAnchor="end"
                    >
                      ${priceLevel}
                    </text>
                  </g>
                );
              })}

              {/* Shaded Out-of-Sample Test Region */}
              {splitX > 0 && (
                <rect 
                  x={splitX} 
                  y={padding.top} 
                  width={svgWidth - padding.right - splitX} 
                  height={svgHeight - padding.top - padding.bottom} 
                  fill="url(#testAreaGradient)" 
                />
              )}

              {/* Chronological Split Line */}
              {splitX > 0 && (
                <g>
                  <line 
                    x1={splitX} 
                    y1={padding.top} 
                    x2={splitX} 
                    y2={svgHeight - padding.bottom} 
                    stroke="#f43f5e" 
                    strokeWidth="1.5" 
                    strokeDasharray="3 3" 
                  />
                  <text 
                    x={splitX + 6} 
                    y={padding.top + 16} 
                    fill="#fb7185" 
                    fontSize="10" 
                    fontWeight="700" 
                    fontFamily="Inter"
                  >
                    80/20 Chronological Split
                  </text>
                </g>
              )}

              {/* Price Line & EMAs */}
              <path d={ema50Path} fill="none" stroke="#10b981" strokeWidth="1.6" strokeDasharray="5 3" />
              <path d={ema20Path} fill="none" stroke="#f59e0b" strokeWidth="1.6" strokeDasharray="4 2" />
              <path d={pricePath} fill="none" stroke="#38bdf8" strokeWidth="2.2" />

              {/* Active Hover Crosshair Line */}
              {activePoint && (
                <line 
                  x1={activePoint.x} 
                  y1={padding.top} 
                  x2={activePoint.x} 
                  y2={svgHeight - padding.bottom} 
                  stroke="rgba(255, 255, 255, 0.4)" 
                  strokeWidth="1" 
                  strokeDasharray="2 2" 
                />
              )}
            </svg>
          </div>
        </div>

        {/* Side Panel: Confusion Matrix & Features */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Confusion Matrix Card */}
          <div className="quant-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Out-of-Sample Confusion Matrix</h3>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Directional Classification on {cm ? cm.total : 0} Unseen Days
                </div>
              </div>
              <ShieldCheck size={18} className="text-cyan-400" />
            </div>

            {cm ? (
              <div className="cm-grid">
                {/* True Negative */}
                <div className="cm-cell hit">
                  <div className="cm-cell-count mono" style={{ color: '#34d399' }}>{cm.tn}</div>
                  <div className="cm-cell-label" style={{ color: '#34d399' }}>True Negative (TN)</div>
                  <div className="cm-cell-sub mono">{cm.tn_pct}% | Correct 'Down'</div>
                </div>

                {/* False Positive */}
                <div className="cm-cell miss">
                  <div className="cm-cell-count mono" style={{ color: '#fb7185' }}>{cm.fp}</div>
                  <div className="cm-cell-label" style={{ color: '#fb7185' }}>False Positive (FP)</div>
                  <div className="cm-cell-sub mono">{cm.fp_pct}% | Type I Error</div>
                </div>

                {/* False Negative */}
                <div className="cm-cell miss">
                  <div className="cm-cell-count mono" style={{ color: '#fb7185' }}>{cm.fn}</div>
                  <div className="cm-cell-label" style={{ color: '#fb7185' }}>False Negative (FN)</div>
                  <div className="cm-cell-sub mono">{cm.fn_pct}% | Type II Error</div>
                </div>

                {/* True Positive */}
                <div className="cm-cell hit">
                  <div className="cm-cell-count mono" style={{ color: '#34d399' }}>{cm.tp}</div>
                  <div className="cm-cell-label" style={{ color: '#34d399' }}>True Positive (TP)</div>
                  <div className="cm-cell-sub mono">{cm.tp_pct}% | Correct 'Up'</div>
                </div>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '16px' }}>Loading matrix...</div>
            )}
          </div>

          {/* Feature Importance Card */}
          <div className="quant-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Quant Feature Importance</h3>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Gini Impurity Reduction in Random Forest
                </div>
              </div>
              <Layers size={18} className="text-indigo-400" />
            </div>

            <div className="feature-list">
              {pipelineData?.feature_importances?.map((feat) => (
                <div key={feat.feature} className="feature-row">
                  <div className="feature-meta">
                    <span className="feature-name">{feat.feature}</span>
                    <span className="mono" style={{ color: 'var(--text-secondary)' }}>{feat.percentage}%</span>
                  </div>
                  <div className="feature-bar-track">
                    <div 
                      className="feature-bar-fill" 
                      style={{ width: `${Math.min(100, feat.percentage * 3.5)}%` }} 
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Recent Out-of-Sample Predictions Ledger */}
      <section className="quant-card table-card">
        <div className="table-header-row">
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Recent Out-of-Sample Predictions Ledger</h3>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Trailing 30 trading days evaluated strictly on out-of-sample data
            </div>
          </div>
          <Clock size={18} className="text-gray-400" />
        </div>

        <div className="quant-table-wrapper">
          <table className="quant-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Close Price</th>
                <th>Daily Return</th>
                <th>Actual Direction</th>
                <th>Predicted Direction</th>
                <th>Confidence</th>
                <th>Validation Status</th>
              </tr>
            </thead>
            <tbody>
              {pipelineData?.recent_predictions?.slice().reverse().map((item, idx) => (
                <tr key={idx}>
                  <td className="mono" style={{ color: '#f8fafc', fontWeight: 500 }}>{item.date}</td>
                  <td className="mono">${item.close}</td>
                  <td className="mono" style={{ color: item.return_pct >= 0 ? '#34d399' : '#fb7185' }}>
                    {item.return_pct >= 0 ? `+${item.return_pct}%` : `${item.return_pct}%`}
                  </td>
                  <td>
                    <span className={`badge ${item.actual_direction === 1 ? 'up' : 'down'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                      {item.actual_direction === 1 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                      {item.actual_direction === 1 ? 'UP' : 'DOWN'}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${item.predicted_direction === 1 ? 'up' : 'down'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}>
                      {item.predicted_direction === 1 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
                      {item.predicted_direction === 1 ? 'UP' : 'DOWN'}
                    </span>
                  </td>
                  <td className="mono">{item.confidence}%</td>
                  <td>
                    <span className={`badge ${item.is_correct ? 'hit' : 'miss'}`}>
                      {item.is_correct ? 'HIT (CORRECT)' : 'MISS (ERROR)'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
