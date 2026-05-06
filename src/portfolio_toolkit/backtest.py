from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Callable

import bt
import pandas as pd
import numpy as np

from .baselines import baseline_weights
from .config import dataset_identifier, resolve_dataset_spec
from .contracts import BacktestResult, DatasetSpec, PortfolioWeights
from .data import load_prices
from .portfolio import (
    weights_from_predictions_rank_long_only,
    weights_from_predictions_risk_adjusted,
    weights_from_predictions_top_k_equal,
)
from .reporting import build_metrics
from .validation import validate_prediction_frame, validate_weights_frame


def _pivot_prices(prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True).tz_localize(None)
    return wide


def _align_weights_to_prices(weights: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    aligned_rows: list[pd.Series] = []
    aligned_index: list[pd.Timestamp] = []
    for date_value, row in weights.sort_index().iterrows():
        requested = pd.Timestamp(date_value)
        position = price_index.searchsorted(requested, side="left")
        if position >= len(price_index):
            continue
        aligned_rows.append(row)
        aligned_index.append(pd.Timestamp(price_index[position]))
    if not aligned_rows:
        raise ValueError("no weight rows align to the available trading calendar")
    aligned = pd.DataFrame(aligned_rows, index=pd.DatetimeIndex(aligned_index))
    aligned.index.name = "date"
    aligned = aligned.groupby(level=0).last()
    row_sums = aligned.sum(axis=1)
    aligned = aligned.div(row_sums.replace(0, np.nan), axis=0)
    return validate_weights_frame(aligned)


def _mask_unavailable_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    adjusted_rows: list[pd.Series] = []
    adjusted_index: list[pd.Timestamp] = []

    for date_value, row in weights.sort_index().iterrows():
        if pd.Timestamp(date_value) not in prices.index:
            continue

        available = prices.loc[pd.Timestamp(date_value) :, row.index].notna().all()
        masked = row.where(available, 0.0).astype(float)
        total = float(masked.sum())

        if total <= 0.0:
            continue

        masked = masked / total
        adjusted_rows.append(masked)
        adjusted_index.append(pd.Timestamp(date_value))

    if not adjusted_rows:
        raise ValueError("no weight rows remain after masking unavailable prices")

    adjusted = pd.DataFrame(adjusted_rows, index=pd.DatetimeIndex(adjusted_index))
    adjusted.index.name = "date"
    return validate_weights_frame(adjusted)


def _make_bt_strategy(name: str, weights: pd.DataFrame) -> bt.Strategy:
    return bt.Strategy(name, [bt.algos.RunDaily(), bt.algos.SelectAll(), bt.algos.WeighTarget(weights), bt.algos.Rebalance()])


def _commission_fn(cost_bps: float) -> Callable[[float, float], float]:
    rate = float(cost_bps) / 10_000.0

    def commission(quantity: float, price: float) -> float:
        return abs(quantity) * price * rate

    return commission


def _compute_turnover(weights: pd.DataFrame) -> pd.Series:
    previous = weights.shift(1).fillna(0.0)
    turnover = (weights.fillna(0.0) - previous).abs().sum(axis=1) / 2.0
    if not turnover.empty:
        turnover.iloc[0] = float(weights.iloc[0].abs().sum())
    turnover.name = "turnover"
    return turnover


def backtest_weights(
    dataset_name: str | DatasetSpec,
    portfolio_weights: PortfolioWeights,
    benchmark: str = "SPY",
    *,
    repo_root: str | Path | None = None,
) -> BacktestResult:
    spec = resolve_dataset_spec(dataset_name, repo_root=repo_root)
    dataset_id = dataset_identifier(spec, repo_root=repo_root)
    prices = load_prices(spec, repo_root=repo_root)
    price_wide = _pivot_prices(prices)
    strategy_tickers = [ticker for ticker in spec.tickers if ticker in portfolio_weights.weights.columns and ticker in price_wide.columns]
    if not strategy_tickers:
        raise ValueError("portfolio weights do not overlap the dataset universe")

    raw_weights = validate_weights_frame(portfolio_weights.weights, dataset_name=spec, repo_root=repo_root)
    aligned_weights = _align_weights_to_prices(raw_weights.loc[:, strategy_tickers], price_wide.index)
    aligned_weights = _mask_unavailable_weights(aligned_weights, price_wide.loc[:, strategy_tickers])

    backtests: list[bt.Backtest] = [
        bt.Backtest(
            _make_bt_strategy(portfolio_weights.strategy_name, aligned_weights),
            price_wide[strategy_tickers],
            commissions=_commission_fn(spec.cost_bps),
            integer_positions=False,
        )
    ]

    benchmark_name = benchmark.upper()
    benchmark_strategy_name = (
        benchmark_name if benchmark_name != portfolio_weights.strategy_name.upper() else f"{benchmark_name}_benchmark"
    )
    if benchmark_name in price_wide.columns:
        benchmark_weights = pd.DataFrame({benchmark_name: [1.0] * len(aligned_weights.index)}, index=aligned_weights.index)
        backtests.append(
            bt.Backtest(
                _make_bt_strategy(benchmark_strategy_name, benchmark_weights),
                price_wide[[benchmark_name]],
                commissions=_commission_fn(spec.cost_bps),
                integer_positions=False,
            )
        )

    equal_benchmark = baseline_weights(spec, "equal_weight", repo_root=repo_root).weights
    equal_benchmark = _align_weights_to_prices(equal_benchmark, price_wide.index)
    equal_benchmark = equal_benchmark.loc[:, strategy_tickers]
    equal_benchmark = _mask_unavailable_weights(equal_benchmark, price_wide.loc[:, strategy_tickers])
    equal_benchmark_name = "equal_weight" if portfolio_weights.strategy_name != "equal_weight" else "equal_weight_benchmark"
    backtests.append(
        bt.Backtest(
            _make_bt_strategy(equal_benchmark_name, equal_benchmark),
            price_wide[strategy_tickers],
            commissions=_commission_fn(spec.cost_bps),
            integer_positions=False,
        )
    )

    result = bt.run(*backtests)
    nav = result.prices[portfolio_weights.strategy_name].rename("nav")
    returns = nav.pct_change().fillna(0.0).rename("returns")
    turnover = _compute_turnover(aligned_weights)
    benchmark_returns = pd.DataFrame(index=nav.index)
    for column in result.prices.columns:
        if column == portfolio_weights.strategy_name:
            continue
        benchmark_returns[column] = result.prices[column].pct_change().fillna(0.0)

    backtest_result = BacktestResult(
        strategy_name=portfolio_weights.strategy_name,
        dataset_name=dataset_id,
        weights=aligned_weights,
        nav=nav,
        returns=returns,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
        metrics={},
    )
    backtest_result.metrics = build_metrics(backtest_result)
    return backtest_result


def backtest_predictions(
    dataset_name: str | DatasetSpec,
    predictions: pd.DataFrame,
    builder: str = "top_k_equal",
    *,
    repo_root: str | Path | None = None,
    **builder_kwargs,
) -> BacktestResult:
    spec = resolve_dataset_spec(dataset_name, repo_root=repo_root)
    validated = validate_prediction_frame(predictions, dataset_name=spec, repo_root=repo_root)
    if builder == "top_k_equal":
        k = int(builder_kwargs.get("k", 5))
        weights = weights_from_predictions_top_k_equal(validated, k=k, dataset_name=spec)
    elif builder == "rank_long_only":
        weights = weights_from_predictions_rank_long_only(validated, dataset_name=spec)
    elif builder == "risk_adjusted":
        weights = weights_from_predictions_risk_adjusted(validated, dataset_name=spec)
    else:
        raise KeyError(f"unknown portfolio builder '{builder}'")
    return backtest_weights(spec, weights, repo_root=repo_root)
