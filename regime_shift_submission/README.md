# Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

This repository is the final submission for the **Summer of Quant Advanced Project**.
It builds a walk-forward tactical asset allocation system that detects market regimes with a Hidden Markov Model and changes portfolio weights across equity, bonds, and gold using convex optimization.

## What this project does

The full pipeline is:

```text
data download → return/features → HMM regime detection → regime-wise CVXPY weights → walk-forward backtest → plots/results
```

The system classifies each day into three regimes:

- **Bull**: lower-volatility / positive-trend state
- **Bear**: weaker-momentum state
- **Crisis**: highest-volatility state

The portfolio then uses a different optimization objective by regime:

| Regime | Optimization logic |
|---|---|
| Bull | Sharpe-oriented mean-variance objective with meaningful equity exposure |
| Bear | More defensive mean-variance objective with equity cap and gold floor |
| Crisis | Minimum-volatility objective with strict equity cap and defensive asset floors |

## Key design decisions

### Why 3 regimes?

The project statement asks for a market detector that distinguishes **Bull / Bear / Crisis** states. Three HMM states are therefore the cleanest modelling choice: enough to separate calm upside, weak/down markets, and crisis volatility without overfitting into many economically meaningless clusters.

### Why these features?

The HMM uses a compact feature set that captures the main dimensions of market state:

- daily equity log return
- 21-day and 63-day equity momentum
- 21-day annualized realized volatility
- VIX / India VIX level
- 63-day equity-bond rolling correlation

The script also saves additional engineered features for inspection, including 5/21/63/126-day momentum and 5/21/63-day volatility. All rolling features use only present and past data.

### How lookahead bias is avoided

The backtest follows strict walk-forward rules:

1. Each fold trains only on past data.
2. Feature scaling is fit only on the fold's training window.
3. The HMM is re-fit separately inside every fold.
4. Out-of-sample regimes are assigned using an online HMM forward filter, not full-test-window Viterbi decoding.
5. Portfolio weights are shifted by one trading day before applying returns, so today's close-based signal is not used to earn today's return.
6. Transaction costs are explicitly charged using daily turnover.

## Repository files

```text
regime_shift_pipeline.py       # Full end-to-end pipeline script
regime_shift_all_in_one.ipynb  # Notebook wrapper that runs the full pipeline and displays results
requirements.txt               # Python dependencies
README.md                      # This file
.gitignore                     # Keeps generated data/outputs out of git by default
```

## How to run

Create a virtual environment, install requirements, then run either the notebook or the script.

```bash
pip install -r requirements.txt
python regime_shift_pipeline.py
```

Or open:

```bash
jupyter notebook regime_shift_all_in_one.ipynb
```

## Data sources

The production path uses `yfinance`.
Ticker candidates are tried in order so the code is robust if one Yahoo symbol is unavailable:

| Series | Candidate tickers |
|---|---|
| Equity | `^NSEI`, `NIFTYBEES.NS`, `^BSESN`, `SPY` |
| Bond | `GILT5YBEES.NS`, `SETF10GILT.NS`, `IN10Y-BD.F`, `IEF`, `TLT` |
| Gold | `GOLDBEES.NS`, `GC=F`, `GLD` |
| Volatility index | `^INDIAVIX`, `^VIX` |

The selected tickers are saved to:

```text
outputs/ticker_selection.csv
```

## Generated outputs

After running, the script creates:

```text
outputs/aligned_prices.csv
outputs/vix_aligned.csv
outputs/features.csv
outputs/asset_returns.csv
outputs/walk_forward_regime_labels.csv
outputs/final_hmm_regime_labels.csv
outputs/final_transition_matrix.csv
outputs/regime_weight_table.csv
outputs/walk_forward_signal_weights.csv
outputs/actual_trade_weights_shifted.csv
outputs/strategy_and_benchmark_returns.csv
outputs/performance_summary.csv
outputs/models/final_hmm_model.joblib
outputs/plots/final_hmm_regime_overlay.png
outputs/plots/walk_forward_regime_overlay.png
outputs/plots/equity_curves.png
outputs/plots/drawdowns.png
outputs/plots/dynamic_weights.png
```

The required performance table is saved as:

```text
outputs/performance_summary.csv
```

It compares:

- Dynamic Strategy Gross
- Dynamic Strategy Net, after transaction costs
- Equal Weight benchmark
- Static 60/40 benchmark

Metrics included:

- CAGR
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio
- turnover

## Transaction costs

Default transaction cost is **7.5 bps per unit of turnover**, which lies inside the requested 5–10 bps range.

You can change it in `Config`:

```python
Config(transaction_cost_bps=10.0)
```

## Offline smoke test

Final reported results should use real `yfinance` data. For debugging only, the script includes a synthetic-data smoke test:

```bash
FORCE_SYNTHETIC_DATA=1 ALLOW_SYNTHETIC_FALLBACK=1 python regime_shift_pipeline.py
```

Do not use synthetic results in the final report; it is only there to verify that the code path runs on a machine without internet.
