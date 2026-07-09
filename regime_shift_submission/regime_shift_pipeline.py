"""
Regime-Shift: Macro-Aware Tactical Asset Allocation Engine

Full pipeline:
    data -> features -> HMM regime detection -> convex optimization -> walk-forward backtest -> results

Run:
    python regime_shift_pipeline.py

Notes:
- The production path uses yfinance data. Keep allow_synthetic_fallback=False for submission results.
- Walk-forward backtest avoids lookahead bias:
    * scalers are fit on train folds only
    * HMMs are fit on train folds only
    * out-of-sample states are assigned by an online forward filter, not full-chunk Viterbi
    * portfolio weights are shifted by one day before applying returns
"""

from __future__ import annotations

import os
# Keep numerical libraries from oversubscribing threads during repeated HMM fits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from hmmlearn import hmm
except Exception as exc:  # pragma: no cover
    raise ImportError("hmmlearn is required. Install with: pip install hmmlearn") from exc

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    raise ImportError("cvxpy is required. Install with: pip install cvxpy") from exc

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

warnings.filterwarnings("ignore", category=FutureWarning)

# Compact HMM feature set: enough to distinguish direction, volatility, fear and stock-bond correlation.
# Extra engineered columns are still saved for inspection, but keeping the HMM compact improves stability.
HMM_FEATURE_COLUMNS = ["eq_log_ret", "mom_21d", "mom_63d", "vol_21d", "vix_level", "eq_bond_corr_63d"]


def select_hmm_features(features: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in HMM_FEATURE_COLUMNS if c in features.columns]
    return features[available].dropna()


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    initial_train_days: int = 756       # ~3 trading years
    test_days: int = 252                # ~1 year; expanding walk-forward without excessive runtime
    n_regimes: int = 3
    random_state: int = 42
    transaction_cost_bps: float = 7.5
    max_weight: float = 0.80
    risk_free_rate_annual: float = 0.00
    output_dir: str = "outputs"
    data_dir: str = "data"
    allow_synthetic_fallback: bool = False

    # Candidate tickers are tried in order. Defaults prioritize Indian proxies.
    ticker_candidates: Optional[Dict[str, List[str]]] = None
    vix_candidates: Optional[List[str]] = None

    def __post_init__(self):
        if self.ticker_candidates is None:
            self.ticker_candidates = {
                "Equity_NSE": ["^NSEI", "NIFTYBEES.NS", "^BSESN", "SPY"],
                "Bond": ["GILT5YBEES.NS", "SETF10GILT.NS", "IN10Y-BD.F", "IEF", "TLT"],
                "Gold": ["GOLDBEES.NS", "GC=F", "GLD"],
            }
        if self.vix_candidates is None:
            self.vix_candidates = ["^INDIAVIX", "^VIX"]


# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs(config: Config) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir, "plots").mkdir(parents=True, exist_ok=True)
    Path(config.output_dir, "models").mkdir(parents=True, exist_ok=True)


def annualization_factor(index: pd.Index) -> int:
    """Use 252 for daily market data. Kept as a function for clarity."""
    return 252


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def performance_stats(returns: pd.Series, turnover: Optional[pd.Series] = None,
                      rf_annual: float = 0.0) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
                "Max Drawdown": np.nan, "Calmar": np.nan, "Turnover": np.nan}

    ann = annualization_factor(returns.index)
    rf_daily = rf_annual / ann
    excess = returns - rf_daily
    equity = (1.0 + returns).cumprod()
    years = max(len(returns) / ann, 1e-12)
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = returns.std(ddof=0) * np.sqrt(ann)
    downside = returns[returns < rf_daily] - rf_daily
    downside_vol = downside.std(ddof=0) * np.sqrt(ann) if len(downside) > 1 else np.nan
    sharpe = (excess.mean() * ann) / vol if vol and vol > 0 else np.nan
    sortino = (excess.mean() * ann) / downside_vol if downside_vol and downside_vol > 0 else np.nan
    mdd = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    avg_turnover = float(turnover.reindex(returns.index).fillna(0).mean()) if turnover is not None else 0.0
    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(mdd),
        "Calmar": float(calmar),
        "Turnover": avg_turnover,
    }


def extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract adjusted close series from a yfinance result, robust to column shape."""
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                if ticker in close.columns:
                    close = close[ticker]
                else:
                    close = close.iloc[:, 0]
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = raw.iloc[:, 0]
    else:
        if "Close" in raw.columns:
            close = raw["Close"]
        elif "Adj Close" in raw.columns:
            close = raw["Adj Close"]
        else:
            close = raw.iloc[:, 0]

    close = pd.Series(close, index=raw.index, name=ticker).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if len(close) < 500:
        raise ValueError(f"Too few observations for {ticker}: {len(close)}")
    return close.astype(float)


def download_first_available(name: str, candidates: Iterable[str], start: str, end: Optional[str]) -> Tuple[str, pd.Series]:
    if yf is None:
        raise ImportError("yfinance is not installed. Install with: pip install yfinance")

    last_error = None
    for ticker in candidates:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False, timeout=10)
            close = extract_close(raw, ticker)
            close.name = name
            print(f"Loaded {name}: {ticker}, {len(close)} rows")
            return ticker, close
        except Exception as exc:
            last_error = exc
            print(f"Skipping {name} candidate {ticker}: {exc}")
    raise RuntimeError(f"Could not load {name}. Tried {list(candidates)}. Last error: {last_error}")


def synthetic_market_data(config: Config) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Offline smoke-test data only. Do not use for final reported project results."""
    rng = np.random.default_rng(config.random_state)
    dates = pd.date_range(config.start_date, config.end_date or "2024-12-31", freq="B")
    n = len(dates)
    regimes = np.zeros(n, dtype=int)
    p = np.array([[0.97, 0.025, 0.005], [0.03, 0.94, 0.03], [0.02, 0.08, 0.90]])
    for i in range(1, n):
        regimes[i] = rng.choice([0, 1, 2], p=p[regimes[i-1]])
    means = np.array([[0.00065, 0.00015, 0.00020], [-0.00025, 0.00030, 0.00025], [-0.00180, 0.00055, 0.00095]])
    vols = np.array([[0.009, 0.004, 0.007], [0.014, 0.005, 0.009], [0.035, 0.008, 0.018]])
    rets = np.vstack([rng.normal(means[s], vols[s]) for s in regimes])
    prices = 100 * np.exp(pd.DataFrame(rets, index=dates, columns=["Equity_NSE", "Bond", "Gold"]).cumsum())
    vix = pd.Series(13 + 8 * regimes + rng.normal(0, 2, n), index=dates, name="VIX").clip(lower=8)
    tickers = {"Equity_NSE": "SYNTH_EQ", "Bond": "SYNTH_BOND", "Gold": "SYNTH_GOLD", "VIX": "SYNTH_VIX"}
    return prices, vix, tickers


def load_market_data(config: Config) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Download and align asset prices and VIX level."""
    if config.allow_synthetic_fallback and os.environ.get("FORCE_SYNTHETIC_DATA", "0") == "1":
        print("FORCE_SYNTHETIC_DATA=1: using synthetic smoke-test data only.")
        return synthetic_market_data(config)
    cache_prices = Path(config.data_dir) / "prices.csv"
    cache_vix = Path(config.data_dir) / "vix.csv"
    cache_meta = Path(config.data_dir) / "ticker_selection.csv"

    # If a user has already run/downloaded once, reuse exact cached data for reproducibility.
    if cache_prices.exists() and cache_vix.exists() and cache_meta.exists():
        prices = pd.read_csv(cache_prices, index_col=0, parse_dates=True)
        vix = pd.read_csv(cache_vix, index_col=0, parse_dates=True).iloc[:, 0]
        tickers = pd.read_csv(cache_meta).set_index("series")["ticker"].to_dict()
        print("Loaded cached data from data/")
        return prices, vix, tickers

    try:
        selected = {}
        series = []
        for name, candidates in config.ticker_candidates.items():
            ticker, close = download_first_available(name, candidates, config.start_date, config.end_date)
            selected[name] = ticker
            series.append(close)
        vix_ticker, vix = download_first_available("VIX", config.vix_candidates, config.start_date, config.end_date)
        selected["VIX"] = vix_ticker

        prices = pd.concat(series, axis=1).dropna(how="any").sort_index()
        vix = vix.reindex(prices.index).ffill().dropna()
        prices = prices.reindex(vix.index).dropna(how="any")
        vix = vix.reindex(prices.index)

        prices.to_csv(cache_prices)
        vix.to_frame("VIX").to_csv(cache_vix)
        pd.Series(selected, name="ticker").rename_axis("series").reset_index().to_csv(cache_meta, index=False)
        return prices, vix, selected
    except Exception as exc:
        if config.allow_synthetic_fallback:
            print("WARNING: yfinance download failed. Using synthetic smoke-test data only.")
            print(f"Download error: {exc}")
            return synthetic_market_data(config)
        raise


# -----------------------------
# Feature Engineering
# -----------------------------

def build_features(prices: pd.DataFrame, vix: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe rolling features and asset log returns."""
    prices = prices.sort_index().dropna()
    asset_returns = np.log(prices).diff().dropna()
    eq = prices.iloc[:, 0]
    eq_ret = asset_returns.iloc[:, 0]

    features = pd.DataFrame(index=prices.index)
    features["eq_log_ret"] = np.log(eq).diff()
    for window in [5, 21, 63, 126]:
        features[f"mom_{window}d"] = eq.pct_change(window)
    for window in [5, 21, 63]:
        features[f"vol_{window}d"] = features["eq_log_ret"].rolling(window).std() * np.sqrt(252)
    features["vix_level"] = vix.reindex(features.index).ffill()
    features["vix_change_5d"] = features["vix_level"].pct_change(5)
    features["eq_bond_corr_63d"] = asset_returns.iloc[:, 0].rolling(63).corr(asset_returns.iloc[:, 1])
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    # Align returns to feature dates.
    asset_returns = asset_returns.reindex(features.index).dropna()
    features = features.reindex(asset_returns.index).dropna()
    return features, asset_returns


# -----------------------------
# HMM / Walk-forward validation
# -----------------------------

def expanding_walk_forward_splits(index: pd.Index, initial_train_days: int, test_days: int):
    n = len(index)
    start_test = initial_train_days
    while start_test < n:
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, min(start_test + test_days, n))
        if len(test_idx) == 0:
            break
        yield train_idx, test_idx
        start_test += test_days


def fit_hmm(X_train: np.ndarray, n_regimes: int, random_state: int) -> hmm.GaussianHMM:
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=30,
        tol=1e-4,
        random_state=random_state,
        min_covar=1e-4,
        verbose=False,
        implementation="scaling",
    )
    model.fit(X_train)
    return model


def logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def online_hmm_filter_states(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Assign states sequentially with the HMM forward filter.

    Unlike model.predict(X) over a whole test chunk, this does not allow later test observations
    to influence earlier labels.
    """
    log_likelihood = model._compute_log_likelihood(X)  # shape: n_obs x n_states
    log_trans = np.log(np.maximum(model.transmat_, 1e-300))
    log_start = np.log(np.maximum(model.startprob_, 1e-300))

    states = []
    log_alpha = None
    for t in range(X.shape[0]):
        if t == 0 or log_alpha is None:
            pred = log_start
        else:
            pred = np.array([logsumexp(log_alpha + log_trans[:, j]) for j in range(model.n_components)])
        log_alpha = pred + log_likelihood[t]
        log_alpha = log_alpha - logsumexp(log_alpha)
        states.append(int(np.argmax(log_alpha)))
    return np.array(states, dtype=int)


def map_states_to_regimes(train_features: pd.DataFrame, train_states: np.ndarray) -> Dict[int, str]:
    """Map arbitrary HMM state numbers to Bull/Bear/Crisis using train-set statistics only."""
    tmp = train_features.copy()
    tmp["state"] = train_states
    stats = tmp.groupby("state").agg(
        vol_21d=("vol_21d", "mean"),
        mom_63d=("mom_63d", "mean"),
        eq_ret=("eq_log_ret", "mean"),
        vix=("vix_level", "mean"),
    )

    available = list(stats.index)
    if len(available) < 3:
        ordered = stats.sort_values(["vol_21d", "mom_63d"], ascending=[True, False]).index.tolist()
        labels = ["Bull", "Bear", "Crisis"]
        return {state: labels[min(i, 2)] for i, state in enumerate(ordered)}

    crisis_state = stats["vol_21d"].idxmax()
    remaining = [s for s in available if s != crisis_state]
    bear_state = stats.loc[remaining, "mom_63d"].idxmin()
    bull_state = [s for s in remaining if s != bear_state][0]
    return {int(bull_state): "Bull", int(bear_state): "Bear", int(crisis_state): "Crisis"}


def scale_train_test(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mu = train.mean()
    sigma = train.std(ddof=0).replace(0, 1.0)
    return (train - mu) / sigma, (test - mu) / sigma, mu, sigma


def run_walk_forward_hmm(features: pd.DataFrame, asset_returns: pd.DataFrame, config: Config) -> Tuple[pd.Series, pd.Series, Dict[int, pd.DataFrame]]:
    """Return out-of-sample regime labels and raw states from walk-forward validation."""
    labels = pd.Series(index=features.index, dtype="object", name="Regime")
    raw_states = pd.Series(index=features.index, dtype="float", name="RawState")
    fold_transition_mats: Dict[int, pd.DataFrame] = {}

    splits = list(expanding_walk_forward_splits(features.index, config.initial_train_days, config.test_days))
    if not splits:
        raise ValueError("Not enough observations for the configured walk-forward split.")

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_features = features.iloc[train_idx]
        test_features = features.iloc[test_idx]
        train_hmm = select_hmm_features(train_features)
        test_hmm = select_hmm_features(test_features)
        # Keep the original train/test rows aligned after feature-column selection.
        train_features = train_features.reindex(train_hmm.index)
        test_features = test_features.reindex(test_hmm.index)
        train_scaled, test_scaled, _, _ = scale_train_test(train_hmm, test_hmm)

        model = fit_hmm(train_scaled.to_numpy(), config.n_regimes, config.random_state + fold)
        train_states = online_hmm_filter_states(model, train_scaled.to_numpy())
        state_map = map_states_to_regimes(train_features, train_states)
        test_states = online_hmm_filter_states(model, test_scaled.to_numpy())
        test_labels = pd.Series(test_states, index=test_features.index).map(state_map).fillna("Bear")

        raw_states.iloc[test_idx] = test_states
        labels.iloc[test_idx] = test_labels.values
        fold_transition_mats[fold] = pd.DataFrame(
            model.transmat_,
            index=[f"State_{i}" for i in range(config.n_regimes)],
            columns=[f"State_{i}" for i in range(config.n_regimes)],
        )
        print(f"Fold {fold:02d}: train {train_features.index[0].date()} -> {train_features.index[-1].date()}, "
              f"test {test_features.index[0].date()} -> {test_features.index[-1].date()}")

    labels = labels.dropna()
    raw_states = raw_states.reindex(labels.index)
    return labels, raw_states, fold_transition_mats


def fit_final_hmm(features: pd.DataFrame, prices: pd.DataFrame, config: Config):
    """Fit a final model on all data for final regime visualization/model export, not for backtest performance."""
    hmm_features = select_hmm_features(features)
    scaled, _, mu, sigma = scale_train_test(hmm_features, hmm_features)
    model = fit_hmm(scaled.to_numpy(), config.n_regimes, config.random_state)
    states = online_hmm_filter_states(model, scaled.to_numpy())
    full_features = features.reindex(hmm_features.index)
    state_map = map_states_to_regimes(full_features, states)
    labels = pd.Series(states, index=hmm_features.index, name="RawState").map(state_map).rename("Regime")
    transition = pd.DataFrame(
        model.transmat_,
        index=[f"State_{i}" for i in range(config.n_regimes)],
        columns=[f"State_{i}" for i in range(config.n_regimes)],
    )
    return model, labels, states, state_map, transition, mu, sigma


# -----------------------------
# Optimization / Backtest
# -----------------------------

def make_psd_cov(returns: pd.DataFrame) -> np.ndarray:
    cov = returns.cov().to_numpy() * 252
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    eig_min = np.linalg.eigvalsh(cov).min()
    if eig_min < 1e-8:
        cov += np.eye(cov.shape[0]) * (1e-8 - eig_min + 1e-6)
    return cov


def regime_constraints(w, regime: str, n_assets: int, config: Config):
    constraints = [cp.sum(w) == 1, w >= 0, w <= config.max_weight]
    # Asset ordering: Equity_NSE, Bond, Gold by construction.
    if n_assets >= 3:
        if regime == "Bull":
            constraints += [w[0] >= 0.35]
        elif regime == "Bear":
            constraints += [w[0] <= 0.55, w[2] >= 0.10]
        elif regime == "Crisis":
            constraints += [w[0] <= 0.25, w[1] >= 0.25, w[2] >= 0.25]
    return constraints


def optimize_regime_weights(train_returns: pd.DataFrame, regime: str, config: Config) -> pd.Series:
    """Use CVXPY to produce long-only regime-specific portfolio weights."""
    n = train_returns.shape[1]
    cols = train_returns.columns
    if len(train_returns) < 60:
        return pd.Series(np.repeat(1 / n, n), index=cols, name=regime)

    mu = train_returns.mean().to_numpy() * 252
    cov = make_psd_cov(train_returns)
    w = cp.Variable(n)
    variance = cp.quad_form(w, cov)
    ret = mu @ w
    constraints = regime_constraints(w, regime, n, config)

    if regime == "Bull":
        # Sharpe-oriented convex proxy: maximize return penalized by variance.
        objective = cp.Maximize(ret - 2.0 * variance)
    elif regime == "Bear":
        # Balance return with stronger risk control.
        objective = cp.Maximize(0.50 * ret - 6.0 * variance)
    else:
        # Crisis: capital preservation / minimum volatility.
        objective = cp.Minimize(variance)

    problem = cp.Problem(objective, constraints)
    solved = False
    for solver in ["CLARABEL", "OSQP", "SCS", "ECOS"]:
        try:
            problem.solve(solver=solver, verbose=False)
            if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
                solved = True
                break
        except Exception:
            continue

    if not solved:
        # Defensive fallback: inverse volatility with regime caps approximated by clipping.
        inv_vol = 1 / train_returns.std(ddof=0).replace(0, np.nan)
        weights = inv_vol / inv_vol.sum()
        weights = weights.fillna(1 / n).to_numpy()
    else:
        weights = np.asarray(w.value).reshape(-1)

    weights = np.clip(weights, 0, config.max_weight)
    weights = weights / weights.sum()
    return pd.Series(weights, index=cols, name=regime)


def compute_regime_weight_table(features: pd.DataFrame, asset_returns: pd.DataFrame,
                                wf_labels: pd.Series, config: Config) -> pd.DataFrame:
    """
    Compute one final weight vector per regime using only the pre-backtest training region.
    This avoids choosing weights with knowledge from the evaluation period.
    """
    first_test_date = wf_labels.index.min()
    train_returns = asset_returns.loc[asset_returns.index < first_test_date]
    train_features = features.loc[features.index < first_test_date]

    # Fit an HMM only on this training segment to identify train regimes for conditional estimates.
    train_hmm = select_hmm_features(train_features)
    train_features = train_features.reindex(train_hmm.index)
    train_scaled, _, _, _ = scale_train_test(train_hmm, train_hmm)
    model = fit_hmm(train_scaled.to_numpy(), config.n_regimes, config.random_state + 999)
    train_states = online_hmm_filter_states(model, train_scaled.to_numpy())
    state_map = map_states_to_regimes(train_features, train_states)
    train_labels = pd.Series(train_states, index=train_features.index).map(state_map)

    weights = []
    for regime in ["Bull", "Bear", "Crisis"]:
        regime_dates = train_labels[train_labels == regime].index
        # If a regime is rare in the first training window, use all training returns rather than future data.
        subset = train_returns.reindex(regime_dates).dropna()
        if len(subset) < 60:
            subset = train_returns.copy()
        weights.append(optimize_regime_weights(subset, regime, config))
    weight_table = pd.DataFrame(weights)
    return weight_table


def build_daily_signal_weights(wf_labels: pd.Series, weight_table: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(index=wf_labels.index, columns=asset_returns.columns, dtype=float)
    for regime, row in weight_table.iterrows():
        weights.loc[wf_labels == regime, :] = row.values
    weights = weights.ffill().dropna()
    return weights


def backtest_strategy(asset_returns: pd.DataFrame, signal_weights: pd.DataFrame, config: Config) -> Dict[str, pd.Series | pd.DataFrame]:
    """Backtest with one-day signal delay and transaction costs."""
    ret = asset_returns.reindex(signal_weights.index).dropna()
    signal_weights = signal_weights.reindex(ret.index).ffill()

    # Trade tomorrow using today's signal. This is the key anti-lookahead step.
    trade_weights = signal_weights.shift(1)
    first_weight = pd.Series(np.repeat(1 / ret.shape[1], ret.shape[1]), index=ret.columns)
    trade_weights.iloc[0] = first_weight
    trade_weights = trade_weights.ffill()

    gross = (trade_weights * ret).sum(axis=1)
    turnover = trade_weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (config.transaction_cost_bps / 10000.0)
    net = gross - cost
    eq_gross = (1 + gross).cumprod().rename("Dynamic Strategy Gross")
    eq_net = (1 + net).cumprod().rename("Dynamic Strategy Net")
    return {
        "gross_returns": gross.rename("Dynamic Strategy Gross"),
        "net_returns": net.rename("Dynamic Strategy Net"),
        "turnover": turnover.rename("Turnover"),
        "trade_weights": trade_weights,
        "equity_gross": eq_gross,
        "equity_net": eq_net,
    }


def benchmark_returns(asset_returns: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    ret = asset_returns.reindex(index).dropna()
    n = ret.shape[1]
    equal_w = np.repeat(1 / n, n)
    sixty_forty = np.zeros(n)
    sixty_forty[0] = 0.60
    if n >= 2:
        sixty_forty[1] = 0.40
    else:
        sixty_forty[0] = 1.0
    bench = pd.DataFrame(index=ret.index)
    bench["Equal Weight"] = ret @ equal_w
    bench["Static 60/40"] = ret @ sixty_forty
    return bench


def make_performance_summary(bt: Dict[str, pd.Series | pd.DataFrame], bench: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    rows.append(pd.Series(performance_stats(bt["gross_returns"], bt["turnover"], config.risk_free_rate_annual), name="Dynamic Strategy Gross"))
    rows.append(pd.Series(performance_stats(bt["net_returns"], bt["turnover"], config.risk_free_rate_annual), name="Dynamic Strategy Net"))
    for col in bench.columns:
        rows.append(pd.Series(performance_stats(bench[col], pd.Series(0.0, index=bench.index), config.risk_free_rate_annual), name=col))
    summary = pd.DataFrame(rows)
    return summary[["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Turnover"]]


# -----------------------------
# Plots / Output
# -----------------------------

def shade_regimes(ax, labels: pd.Series) -> None:
    colors = {"Bull": "#2ca02c", "Bear": "#ff7f0e", "Crisis": "#d62728"}
    labels = labels.dropna()
    if labels.empty:
        return
    start = labels.index[0]
    current = labels.iloc[0]
    prev = labels.index[0]
    for dt, label in labels.iloc[1:].items():
        if label != current:
            ax.axvspan(start, prev, color=colors.get(current, "grey"), alpha=0.18, linewidth=0)
            start = dt
            current = label
        prev = dt
    ax.axvspan(start, prev, color=colors.get(current, "grey"), alpha=0.18, linewidth=0)


def plot_regimes(prices: pd.DataFrame, labels: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    equity_price = prices.iloc[:, 0].reindex(labels.index).dropna()
    ax.plot(equity_price.index, equity_price, linewidth=1.2, color="black", label=prices.columns[0])
    shade_regimes(ax, labels.reindex(equity_price.index))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in ["#2ca02c", "#ff7f0e", "#d62728"]]
    ax.legend(handles + [ax.lines[0]], ["Bull", "Bear", "Crisis", prices.columns[0]], loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("Price")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_equity_curves(bt: Dict[str, pd.Series | pd.DataFrame], bench: pd.DataFrame, path: Path) -> None:
    equity = pd.DataFrame(index=bench.index)
    equity["Dynamic Strategy Net"] = (1 + bt["net_returns"].reindex(bench.index)).cumprod()
    equity["Dynamic Strategy Gross"] = (1 + bt["gross_returns"].reindex(bench.index)).cumprod()
    for col in bench.columns:
        equity[col] = (1 + bench[col]).cumprod()

    fig, ax = plt.subplots(figsize=(13, 6))
    for col in equity.columns:
        ax.plot(equity.index, equity[col], linewidth=1.4, label=col)
    ax.set_title("Equity Curves: Dynamic Strategy vs Static Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_drawdown(bt: Dict[str, pd.Series | pd.DataFrame], bench: pd.DataFrame, path: Path) -> None:
    returns = pd.concat([bt["net_returns"].rename("Dynamic Strategy Net"), bench], axis=1).dropna()
    dd = (1 + returns).cumprod()
    dd = dd / dd.cummax() - 1
    fig, ax = plt.subplots(figsize=(13, 5))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], linewidth=1.2, label=col)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    weights.plot.area(ax=ax, linewidth=0)
    ax.set_title("Walk-forward Dynamic Portfolio Weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_outputs(config: Config, prices: pd.DataFrame, vix: pd.Series, tickers: Dict[str, str],
                 features: pd.DataFrame, asset_returns: pd.DataFrame, wf_labels: pd.Series,
                 final_labels: pd.Series, final_states: np.ndarray, state_map: Dict[int, str],
                 transition: pd.DataFrame, weight_table: pd.DataFrame,
                 signal_weights: pd.DataFrame, bt: Dict[str, pd.Series | pd.DataFrame],
                 bench: pd.DataFrame, summary: pd.DataFrame, model) -> None:
    out = Path(config.output_dir)
    plots = out / "plots"
    prices.to_csv(out / "aligned_prices.csv")
    vix.to_frame("VIX").to_csv(out / "vix_aligned.csv")
    features.to_csv(out / "features.csv")
    asset_returns.to_csv(out / "asset_returns.csv")
    wf_labels.to_frame("WalkForwardRegime").to_csv(out / "walk_forward_regime_labels.csv")
    final_regime = pd.DataFrame({"FinalRawState": final_states, "FinalRegime": final_labels}, index=features.index)
    final_regime.to_csv(out / "final_hmm_regime_labels.csv")
    transition.to_csv(out / "final_transition_matrix.csv")
    weight_table.to_csv(out / "regime_weight_table.csv")
    signal_weights.to_csv(out / "walk_forward_signal_weights.csv")
    bt["trade_weights"].to_csv(out / "actual_trade_weights_shifted.csv")
    pd.concat([bt["gross_returns"], bt["net_returns"], bt["turnover"], bench], axis=1).to_csv(out / "strategy_and_benchmark_returns.csv")
    summary.to_csv(out / "performance_summary.csv")
    pd.Series(tickers, name="ticker").rename_axis("series").to_csv(out / "ticker_selection.csv")
    pd.Series(asdict(config), name="value").to_csv(out / "run_config.csv")

    plot_regimes(prices, final_labels, plots / "final_hmm_regime_overlay.png", "Final HMM Regimes Overlaid on Equity Price")
    plot_regimes(prices, wf_labels, plots / "walk_forward_regime_overlay.png", "Walk-forward Out-of-sample Regimes Overlaid on Equity Price")
    plot_equity_curves(bt, bench, plots / "equity_curves.png")
    plot_drawdown(bt, bench, plots / "drawdowns.png")
    plot_weights(bt["trade_weights"], plots / "dynamic_weights.png")

    if joblib is not None:
        joblib.dump({"model": model, "state_map": state_map}, out / "models" / "final_hmm_model.joblib")


def run_pipeline(config: Optional[Config] = None):
    config = config or Config()
    ensure_dirs(config)
    print("Configuration:")
    print(pd.Series(asdict(config)).to_string())

    prices, vix, tickers = load_market_data(config)
    features, asset_returns = build_features(prices, vix)
    print(f"Feature matrix: {features.shape}; asset returns: {asset_returns.shape}")

    wf_labels, raw_states, fold_transition_mats = run_walk_forward_hmm(features, asset_returns, config)
    model, final_labels, final_states, state_map, transition, mu, sigma = fit_final_hmm(features, prices, config)

    weight_table = compute_regime_weight_table(features, asset_returns, wf_labels, config)
    signal_weights = build_daily_signal_weights(wf_labels, weight_table, asset_returns)
    bt = backtest_strategy(asset_returns, signal_weights, config)
    bench = benchmark_returns(asset_returns, bt["net_returns"].index)
    summary = make_performance_summary(bt, bench, config)

    save_outputs(config, prices, vix, tickers, features, asset_returns, wf_labels,
                 final_labels, final_states, state_map, transition, weight_table,
                 signal_weights, bt, bench, summary, model)

    print("\nSelected tickers:")
    print(pd.Series(tickers).to_string())
    print("\nFinal HMM transition matrix:")
    print(transition.round(3).to_string())
    print("\nRegime weight table:")
    print(weight_table.round(4).to_string())
    print("\nPerformance summary:")
    print(summary.round(4).to_string())
    print(f"\nSaved outputs under: {Path(config.output_dir).resolve()}")

    return {
        "prices": prices,
        "vix": vix,
        "tickers": tickers,
        "features": features,
        "asset_returns": asset_returns,
        "walk_forward_labels": wf_labels,
        "final_labels": final_labels,
        "transition_matrix": transition,
        "weight_table": weight_table,
        "backtest": bt,
        "benchmarks": bench,
        "summary": summary,
    }


if __name__ == "__main__":
    # For final submission results, leave allow_synthetic_fallback=False.
    # Set environment variable ALLOW_SYNTHETIC_FALLBACK=1 only for offline smoke testing.
    cfg = Config(allow_synthetic_fallback=os.environ.get("ALLOW_SYNTHETIC_FALLBACK", "0") == "1")
    run_pipeline(cfg)
