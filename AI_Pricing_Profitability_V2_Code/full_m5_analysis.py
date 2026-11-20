"""Memory-bounded full-series M5 demand benchmark and submission report."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor


SERIES_BATCH_SIZE = 256
MAX_TRAINING_DAYS_PER_SERIES = 256
VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2
class _Metrics:
    def __init__(self):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.actual_sum = 0.0
        self.actual_squared_sum = 0.0
        self.actual_revenue = 0.0
        self.revenue_absolute_error = 0.0
        self.contribution_absolute_error = 0.0

    def add(self, actual, predicted, price):
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.maximum(np.asarray(predicted, dtype=np.float64), 0)
        price = np.asarray(price, dtype=np.float64)
        error = predicted - actual
        self.count += len(actual)
        self.absolute_error += float(np.abs(error).sum())
        self.squared_error += float(np.square(error).sum())
        self.actual_sum += float(actual.sum())
        self.actual_squared_sum += float(np.square(actual).sum())
        self.actual_revenue += float((actual * price).sum())
        self.revenue_absolute_error += float((np.abs(error) * price).sum())
        self.contribution_absolute_error += float((np.abs(error) * price * 0.35).sum())

    def values(self):
        if not self.count:
            return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                    "Revenue_MAE": np.nan, "Contribution_MAE": np.nan}
        total_variance = self.actual_squared_sum - self.actual_sum**2 / self.count
        return {
            "MAE": self.absolute_error / self.count,
            "RMSE": np.sqrt(self.squared_error / self.count),
            "R2": (
                1 - self.squared_error / total_variance
                if total_variance > 0 else np.nan
            ),
            "Revenue_MAE": self.revenue_absolute_error / self.count,
            "Contribution_MAE": self.contribution_absolute_error / self.count,
        }


class _BlendSelection:
    def __init__(self):
        self.count = 0
        self.lag_squared_error = 0.0
        self.cross_error = 0.0
        self.model_lag_squared_error = 0.0

    def add(self, actual, prediction, lag):
        model_error = np.maximum(np.asarray(prediction, dtype=np.float64), 0) - actual
        lag_error = np.maximum(np.asarray(lag, dtype=np.float64), 0) - actual
        difference = model_error - lag_error
        self.count += len(actual)
        self.lag_squared_error += float(np.dot(lag_error, lag_error))
        self.cross_error += float(np.dot(difference, lag_error))
        self.model_lag_squared_error += float(np.dot(difference, difference))

    def best_alpha(self):
        if not self.model_lag_squared_error:
            return 0.0
        return float(np.clip(
            -self.cross_error / self.model_lag_squared_error, 0, 1
        ))

    def rmse(self, alpha):
        squared_error = (
            self.lag_squared_error
            + 2 * alpha * self.cross_error
            + alpha**2 * self.model_lag_squared_error
        )
        return float(np.sqrt(max(squared_error, 0) / self.count))


def _read_m5_inputs(sales_path, prices_path, calendar_path):
    sales_header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [column for column in sales_header if column.startswith("d_")]
    if not day_columns or not set(id_columns).issubset(sales_header):
        raise ValueError("The sales file does not have the expected M5 columns.")

    sales_dtypes = {column: "category" for column in id_columns}
    sales_dtypes.update({column: np.uint16 for column in day_columns})
    sales = pd.read_csv(sales_path, dtype=sales_dtypes, low_memory=False)
    prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": np.int32,
            "sell_price": np.float32,
        },
        low_memory=False,
    )
    calendar = pd.read_csv(calendar_path, usecols=["d", "date", "wm_yr_wk"])

    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("The M5 sell-price file has duplicate store/item/week keys.")
    if not calendar["d"].is_unique:
        raise ValueError("The M5 calendar has duplicate day identifiers.")
    return sales, prices, calendar, day_columns


def _make_features(quantity, price, base_price, day_index, start, end):
    target_price = price[:, start:end]
    lag_one = np.zeros_like(target_price, dtype=np.float32)
    lag_seven = np.zeros_like(target_price, dtype=np.float32)
    if start:
        lag_one[:, 0] = quantity[:, start - 1]
    if end - start > 1:
        lag_one[:, 1:] = quantity[:, start:end - 1]
    first_valid_lag_seven = max(0, 7 - start)
    if first_valid_lag_seven < end - start:
        lag_seven[:, first_valid_lag_seven:] = quantity[
            :, start + first_valid_lag_seven - 7:end - 7
        ]

    safe_base = np.where(np.isfinite(base_price) & (base_price > 0), base_price, 1)
    relative_price = target_price / safe_base[:, None]
    day = np.asarray(day_index[start:end], dtype=np.float32)
    annual_sin = np.broadcast_to(
        np.sin(2 * np.pi * day / 365.25)[None, :], target_price.shape
    )
    annual_cos = np.broadcast_to(
        np.cos(2 * np.pi * day / 365.25)[None, :], target_price.shape
    )
    weekly_sin = np.broadcast_to(
        np.sin(2 * np.pi * day / 7)[None, :], target_price.shape
    )
    weekly_cos = np.broadcast_to(
        np.cos(2 * np.pi * day / 7)[None, :], target_price.shape
    )
    features = np.stack(
        [
            np.log1p(target_price),
            relative_price - 1,
            annual_sin,
            annual_cos,
            weekly_sin,
            weekly_cos,
            np.log1p(lag_one),
            np.log1p(lag_seven),
        ],
        axis=-1,
    )
    return features, lag_one, lag_seven


def _make_features_at(quantity, price, base_price, day_indices, rows, days):
    row_base = np.where(
        np.isfinite(base_price[rows]) & (base_price[rows] > 0),
        base_price[rows],
        1,
    )
    lag_one = np.where(days > 0, quantity[rows, np.maximum(days - 1, 0)], 0)
    lag_seven = np.where(days >= 7, quantity[rows, np.maximum(days - 7, 0)], 0)
    actual_price = price[rows, days]
    calendar_days = np.asarray(day_indices[days], dtype=np.float32)
    return np.column_stack(
        [
            np.log1p(actual_price),
            actual_price / row_base - 1,
            np.sin(2 * np.pi * calendar_days / 365.25),
            np.cos(2 * np.pi * calendar_days / 365.25),
            np.sin(2 * np.pi * calendar_days / 7),
            np.cos(2 * np.pi * calendar_days / 7),
            np.log1p(lag_one),
            np.log1p(lag_seven),
        ]
    ).astype(np.float32, copy=False)


def _iter_split_batches(
    store_blocks, sales_quantities, week_positions, day_indices, split_start, split_end,
    progress_callback=None, progress_start=0.0, progress_end=0.0, phase="",
):
    store_count = max(len(store_blocks), 1)
    for store_number, block in enumerate(store_blocks, start=1):
        row_indexes = block["row_indexes"]
        for offset in range(0, len(row_indexes), SERIES_BATCH_SIZE):
            batch_start = offset
            batch_end = min(offset + SERIES_BATCH_SIZE, len(row_indexes))
            indexes = row_indexes[batch_start:batch_end]
            quantity = sales_quantities[indexes].astype(np.float32)
            daily_price = block["weekly_prices"][batch_start:batch_end][
                :, week_positions
            ]
            training_base = block["training_base"][batch_start:batch_end]
            features, lag_one, lag_seven = _make_features(
                quantity, daily_price, training_base, day_indices,
                split_start, split_end,
            )
            mask = np.isfinite(daily_price[:, split_start:split_end])
            rows, local_days = np.nonzero(mask)
            if not len(rows):
                continue
            if progress_callback:
                store_fraction = (
                    (store_number - 1)
                    + min((offset + SERIES_BATCH_SIZE) / len(row_indexes), 1.0)
                ) / store_count
                progress_callback(
                    progress_start + (progress_end - progress_start) * store_fraction,
                    f"{phase} ({store_number}/{store_count} stores)",
                )
            yield {
                "features": features[rows, local_days],
                "actual": quantity[rows, local_days + split_start],
                "price": daily_price[rows, local_days + split_start],
                "lag_one": lag_one[rows, local_days],
                "lag_seven": lag_seven[rows, local_days],
            }


def evaluate_full_m5(sales_path, prices_path, calendar_path, progress_callback=None):
    """Evaluate all M5 item/store series without creating a full long table."""
    started = perf_counter()
    if progress_callback:
        progress_callback(0.01, "Reading M5 input files")
    sales, prices, calendar, day_columns = _read_m5_inputs(
        sales_path, prices_path, calendar_path
    )
    if progress_callback:
        progress_callback(0.04, "Aligning full-series prices")
    day_map = calendar.set_index("d").reindex(day_columns)
    dates = pd.to_datetime(day_map["date"])
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("M5 dates are missing or not in chronological order.")
    day_count = len(day_columns)
    validation_start = int(np.floor(day_count * (1 - VALIDATION_FRACTION - TEST_FRACTION)))
    test_start = int(np.floor(day_count * (1 - TEST_FRACTION)))
    if validation_start < 1 or test_start <= validation_start or test_start >= day_count:
        raise ValueError("M5 does not contain enough dates for temporal evaluation.")

    store_groups = {
        str(store): row_indexes
        for store, row_indexes in sales.groupby(
            "store_id", observed=True, sort=False
        ).indices.items()
    }
    series_count = len(sales)
    price_groups = {
        str(store): row_indexes
        for store, row_indexes in prices.groupby(
            "store_id", observed=True, sort=False
        ).indices.items()
    }
    weeks = day_map["wm_yr_wk"].to_numpy(dtype=np.int32)
    unique_weeks = np.unique(weeks)
    week_positions = np.searchsorted(unique_weeks, weeks)
    training_week_positions = np.searchsorted(
        unique_weeks, np.unique(weeks[:validation_start])
    )
    store_blocks = []
    total_store_groups = max(len(store_groups), 1)
    for store_number, (store, row_indexes) in enumerate(store_groups.items(), start=1):
        price_indexes = price_groups.get(store)
        if price_indexes is None:
            if progress_callback:
                progress_callback(
                    0.04 + 0.06 * store_number / total_store_groups,
                    f"Aligning prices ({store_number}/{total_store_groups} stores)",
                )
            continue
        store_sales = sales.iloc[row_indexes]
        item_ids = store_sales["item_id"].astype(str).to_numpy()
        if len(np.unique(item_ids)) != len(item_ids):
            raise ValueError(f"M5 sales has duplicate item/store series for {store}.")
        price_matrix = prices.iloc[price_indexes].pivot(
            index="item_id", columns="wm_yr_wk", values="sell_price"
        ).reindex(index=item_ids, columns=unique_weeks)
        weekly_prices = price_matrix.to_numpy(dtype=np.float32)
        training_prices = weekly_prices[:, training_week_positions]
        training_base = np.nanmedian(training_prices, axis=1).astype(np.float32)
        store_blocks.append({
            "store": store,
            "row_indexes": row_indexes,
            "weekly_prices": weekly_prices,
            "training_base": training_base,
        })
        if progress_callback:
            progress_callback(
                0.04 + 0.06 * store_number / total_store_groups,
                f"Aligning prices ({store_number}/{total_store_groups} stores)",
            )
    del prices, price_groups
    sales_quantities = sales[day_columns].to_numpy(dtype=np.uint16, copy=False)
    del sales
    day_indices = np.arange(day_count, dtype=np.int32)

    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        average=True,
        random_state=42,
    )
    if progress_callback:
        progress_callback(0.10, "Training bounded incremental model")
    train_rows = 0
    model_fit_rows = 0
    train_units = 0.0
    train_revenue = 0.0
    effective_series = 0
    training_series = 0
    rng = np.random.default_rng(42)

    total_stores = max(len(store_blocks), 1)
    training_order = rng.permutation(len(store_blocks))
    for store_number, block_index in enumerate(training_order, start=1):
        block = store_blocks[block_index]
        row_indexes = block["row_indexes"]
        for offset in range(0, len(row_indexes), SERIES_BATCH_SIZE):
            batch_end = min(offset + SERIES_BATCH_SIZE, len(row_indexes))
            indexes = row_indexes[offset:batch_end]
            quantity = sales_quantities[indexes].astype(np.float32)
            daily_price = block["weekly_prices"][offset:batch_end][:, week_positions]
            training_base = block["training_base"][offset:batch_end]
            active = np.isfinite(daily_price).any(axis=1)
            effective_series += int(active.sum())

            mask = np.isfinite(daily_price[:, :validation_start])
            rows, cols = np.nonzero(mask)
            if len(rows):
                train_rows += len(rows)
                train_units += float(quantity[rows, cols].sum())
                train_revenue += float(
                    (quantity[rows, cols] * daily_price[rows, cols]).sum()
                )
                selected_rows = []
                selected_cols = []
                for series_row in range(len(quantity)):
                    observed_days = np.flatnonzero(mask[series_row])
                    if not len(observed_days):
                        continue
                    training_series += 1
                    if len(observed_days) > MAX_TRAINING_DAYS_PER_SERIES:
                        sample_positions = np.linspace(
                            0,
                            len(observed_days) - 1,
                            MAX_TRAINING_DAYS_PER_SERIES,
                            dtype=np.int32,
                        )
                        observed_days = observed_days[sample_positions]
                    selected_rows.append(
                        np.full(len(observed_days), series_row, dtype=np.int32)
                    )
                    selected_cols.append(observed_days)
                fit_rows = np.concatenate(selected_rows)
                fit_cols = np.concatenate(selected_cols)
                shuffled = rng.permutation(len(fit_rows))
                fit_features = _make_features_at(
                    quantity,
                    daily_price,
                    training_base,
                    day_indices,
                    fit_rows,
                    fit_cols,
                )
                model.partial_fit(
                    fit_features[shuffled],
                    quantity[fit_rows[shuffled], fit_cols[shuffled]],
                )
                model_fit_rows += len(fit_rows)

        if progress_callback:
            progress_callback(
                0.10 + 0.45 * store_number / total_stores,
                f"Training full-M5 incremental model ({store_number}/{total_stores} stores)",
            )

    if not train_rows:
        raise ValueError("No priced M5 observations were available for model training.")
    training_mean = train_units / train_rows

    blend_selection = _BlendSelection()
    model_validation = _Metrics()
    validation_mean = _Metrics()
    validation_lag = _Metrics()
    validation_seasonal = _Metrics()
    validation_count = 0
    if progress_callback:
        progress_callback(0.55, "Scoring validation dates")
    for batch in _iter_split_batches(
        store_blocks, sales_quantities, week_positions, day_indices,
        validation_start, test_start, progress_callback, 0.55, 0.75, "Validating",
    ):
        prediction = np.maximum(model.predict(batch["features"]), 0)
        blend_selection.add(batch["actual"], prediction, batch["lag_one"])
        model_validation.add(batch["actual"], prediction, batch["price"])
        validation_mean.add(
            batch["actual"], np.full(len(batch["actual"]), training_mean), batch["price"]
        )
        validation_lag.add(
            batch["actual"], batch["lag_one"], batch["price"]
        )
        validation_seasonal.add(
            batch["actual"], batch["lag_seven"], batch["price"]
        )
        validation_count += len(batch["actual"])

    validation_results = {
        "Training Mean Baseline": validation_mean.values(),
        "Previous-Day Naive": validation_lag.values(),
        "Seasonal Naive (7-day)": validation_seasonal.values(),
    }
    model_validation = model_validation.values()
    validation_results["Incremental SGDRegressor"] = model_validation
    best_alpha = blend_selection.best_alpha()
    hybrid_name = f"Tuned Hybrid (Incremental SGDRegressor + previous-day, alpha={best_alpha:.2f})"
    validation_results[hybrid_name] = {"RMSE": blend_selection.rmse(best_alpha)}

    candidates = {
        name: metrics["RMSE"] for name, metrics in validation_results.items()
    }
    selected_name = min(candidates, key=candidates.get)

    test_metrics = {name: _Metrics() for name in validation_results}
    test_count = 0
    if progress_callback:
        progress_callback(0.75, "Scoring untouched test dates")
    for batch in _iter_split_batches(
        store_blocks, sales_quantities, week_positions, day_indices,
        test_start, day_count, progress_callback, 0.75, 1.0, "Testing",
    ):
        actual = batch["actual"]
        price = batch["price"]
        lag_one = batch["lag_one"]
        lag_seven = batch["lag_seven"]
        raw_prediction = np.maximum(model.predict(batch["features"]), 0)
        predictions = {
            "Training Mean Baseline": np.full(len(actual), training_mean),
            "Previous-Day Naive": lag_one,
            "Seasonal Naive (7-day)": lag_seven,
            "Incremental SGDRegressor": raw_prediction,
            hybrid_name: best_alpha * raw_prediction + (1 - best_alpha) * lag_one,
        }
        for name, prediction in predictions.items():
            test_metrics[name].add(actual, prediction, price)
        test_count += len(actual)

    if progress_callback:
        progress_callback(1.0, "Full-M5 evaluation complete")

    return {
        "series_count": series_count,
        "effective_series_count": effective_series,
        "eligible_rows": train_rows + validation_count + test_count,
        "train_rows": train_rows,
        "model_fit_rows": model_fit_rows,
        "training_series_count": training_series,
        "validation_rows": validation_count,
        "test_rows": test_count,
        "total_units_sold": (
            train_units + validation_mean.actual_sum
            + test_metrics[selected_name].actual_sum
        ),
        "total_revenue_usd": (
            train_revenue + validation_mean.actual_revenue
            + test_metrics[selected_name].actual_revenue
        ),
        "elapsed_seconds": perf_counter() - started,
        "date_start": dates.iloc[0].date().isoformat(),
        "date_train_end": dates.iloc[validation_start - 1].date().isoformat(),
        "date_validation_start": dates.iloc[validation_start].date().isoformat(),
        "date_test_start": dates.iloc[test_start].date().isoformat(),
        "date_end": dates.iloc[-1].date().isoformat(),
        "training_mean": training_mean,
        "validation_results": validation_results,
        "test_results": {name: metrics.values() for name, metrics in test_metrics.items()},
        "selected_model": selected_name,
        "selected_alpha": best_alpha,
        "selected_test": test_metrics[selected_name].values(),
        "cost_assumption": 0.65,
        "price_currency": "USD",
    }


def render_full_m5_report(results):
    """Create reproducible findings and interpretation for the final review."""
    selected = results["selected_model"]
    selected_validation = results["validation_results"][selected]
    selected_test = results["selected_test"]
    previous_day_rmse = results["test_results"]["Previous-Day Naive"]["RMSE"]
    rmse_improvement = (
        100 * (previous_day_rmse - selected_test["RMSE"]) / previous_day_rmse
        if previous_day_rmse else 0.0
    )
    comparison_rows = [
        "| Model | Validation RMSE (units) | Test MAE (units) | Test RMSE (units) | Test R² | Revenue MAE (USD/record) | Estimated contribution MAE (USD/record) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, validation in results["validation_results"].items():
        test = results["test_results"][name]
        comparison_rows.append(
            f"| {name} | {validation['RMSE']:.4f} | {test['MAE']:.4f} | "
            f"{test['RMSE']:.4f} | {test['R2']:.4f} | "
            f"{test['Revenue_MAE']:.4f} | {test['Contribution_MAE']:.4f} |"
        )
    return "\n".join(
        [
            "# Full M5 Walmart Dataset: Findings, Analysis and Interpretation",
            "",
            "## Scope and method",
            "",
            f"This run includes all {results['series_count']:,} item/store series in the supplied M5 sales file. "
            f"{results['effective_series_count']:,} series have at least one observed day with a matched selling price. "
            f"The analysis covers {results['date_start']} through {results['date_end']}. "
            "Daily observations without a matched M5 selling price are excluded; zero-sales days with a matched price are retained.",
            "",
            f"Priced observations used: {results['eligible_rows']:,} "
            f"(training {results['train_rows']:,}; validation {results['validation_rows']:,}; "
            f"untouched test {results['test_rows']:,}). Chronological date boundaries: "
            f"training ends {results['date_train_end']}; validation runs from "
            f"{results['date_validation_start']} to the day before test; test starts "
            f"{results['date_test_start']}. Validation is used for model selection; test dates are held out.",
            "",
            f"Across the matched-price observations, total units sold were "
            f"{results['total_units_sold']:,.0f}; source-price revenue was "
            f"${results['total_revenue_usd']:,.2f}. These are descriptive sums over the included records.",
            "",
            "To process all product/store series on a memory-constrained local machine, the evaluation reads the sales matrix in bounded batches rather than creating a single 58-million-row long table. "
            "The M5-specific predictive model is an incremental global SGD regressor, trained batch by batch. It is compared with training-mean, previous-day and 7-day seasonal-naive benchmarks. "
            "This model set differs from the interactive sampled benchmark, so its metrics should not be compared as if they came from the same model experiment.",
            "",
            "The global model uses log selling price, price relative to each series' median selling price during the training period, annual and weekly calendar cycles, and previous-day/previous-week demand lags. It does not fit separate item/store-specific model parameters.",
            "",
            "The previous-day and 7-day baselines use actual observed quantity from the corresponding prior calendar day. This is a rolling one-day-ahead evaluation, not a fixed-origin multi-month forecast.",
            "",
            "The incremental model is trained on the training date window only. The validation window selects the benchmark/blend but is not added to model fitting before the held-out test scoring.",
            "",
            f"To bound CPU and memory, the SGD model was fitted on {results['model_fit_rows']:,} "
            f"deterministically sampled training records ({results['model_fit_rows'] / results['train_rows']:.1%} "
            f"of priced training observations), using up to {MAX_TRAINING_DAYS_PER_SERIES} evenly spaced observations "
            f"for each of the {results['training_series_count']:,} series with priced training history. "
            "Validation and test metrics use every eligible priced observation in their respective date windows. "
            "The training-mean baseline uses all priced training observations.",
            "",
            "## Full-dataset forecast results",
            "",
            *comparison_rows,
            "",
            f"Selected by validation RMSE: **{selected}** "
            f"(validation RMSE {selected_validation['RMSE']:.4f} units). "
            f"On untouched test dates, its MAE was {selected_test['MAE']:.4f} units, "
            f"RMSE {selected_test['RMSE']:.4f} units, and R² {selected_test['R2']:.4f}. "
            "These values are the measured output of this full-M5 run, not estimates inferred from the sampled app screenshot.",
            "",
            (
                f"The selected hybrid assigns {results['selected_alpha']:.0%} weight to "
                "the incremental model and the remainder to previous-day demand."
                if selected.startswith("Tuned Hybrid")
                else "The validation-selected method is used without a hybrid blend."
            ),
            "",
                f"Compared with the previous-day naive test baseline (RMSE {previous_day_rmse:.4f}), "
                f"the selected model's test RMSE was {rmse_improvement:.1f}% lower. This is a forecast-error "
                "comparison only, not a financial gain or a causal pricing effect.",
                "",
                f"Full-M5 benchmark elapsed time: {results['elapsed_seconds'] / 60:.1f} minutes.",
                "",
            "Revenue MAE and estimated contribution MAE are conditional forecast-error measures calculated from observed prices. They do not measure revenue or profit caused by a price change.",
            "",
            "## Management interpretation",
            "",
            "This is a management case study using secondary, observed retail data and a decision-support proof of concept. Pricing decisions require balancing demand, revenue and margin rather than maximizing sales volume alone. The benchmark illustrates how forecasts can support comparison of demand scenarios and inform pricing analysis; it is not an automated pricing recommendation or evidence that a tested price will achieve a particular financial outcome.",
            "",
            "## Research justification and no primary survey",
            "",
            "> I chose the M5 Walmart dataset because my research question is about using observed sales and price history to support pricing decisions. Customer surveys can provide useful information about preferences and willingness to buy, but stated intentions may differ from actual purchases. The M5 data provides a longitudinal retail case with product, store, price, demand, and calendar information, allowing me to compare forecasting approaches and demonstrate how their outputs can be used in pricing scenarios.",
            "",
            "No polls, surveys or interviews were conducted. A survey would be appropriate if the main research question concerned customer attitudes or stated willingness to pay. This research instead examines observed sales and pricing patterns, so secondary historical retail data is more directly aligned. The chat was not used to collect user or participant data: no respondents were recruited and no survey questions were administered. A survey could complement this analysis in future research.",
            "",
            "## Limitations",
            "",
            "- The M5 records are observational. Forecast accuracy does not establish causal price effects or prove that a price change caused demand to change.",
            "- M5 does not provide actual product-level cost for this prototype. Unit cost is assumed to be 65% of selling price; estimated contribution is therefore not Walmart's actual accounting profit.",
            "- M5 does not provide a complete verified promotion label or list price. Promotion and discount interpretations based on reference prices are proxies, not verified Walmart promotions.",
            "- The incremental SGD model is a computationally scalable global benchmark and is methodologically different from the interactive sampled XGBoost/ensemble comparison.",
            "- To reduce local runtime and peak memory, model fitting uses a deterministic sample of priced training rows across all product/store batches; validation and test scores are computed over all eligible rows. The model-fit sample fraction is reported above.",
            "- Results apply to the supplied M5 files and their available observed dates. Missing-price observations are excluded; zero-sales observations with prices are included.",
            "",
            "## Final-review conclusion",
            "",
            "The full-series evaluation demonstrates a reproducible, chronological forecasting workflow over the supplied M5 item/store data. Its management contribution is to show how observed demand forecasts and simple benchmarks can inform structured pricing discussion. Conclusions should remain decision-support oriented: the analysis is not causal, and estimated contribution is not actual profit.",
            "",
        ]
    )


def write_full_m5_report(results, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = render_full_m5_report(results)
    path.write_text(report, encoding="utf-8")
    return report
