"""Demand model construction, selection, and temporal evaluation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from model_contracts import DemandEstimatorProvider
from pricing_data import demand_features, prepare_demand_data, split_for_backtest

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


class DefaultDemandEstimatorProvider:
    """Supplies the project's default estimator comparison set."""

    def build_estimators(self):
        estimators = {
            "Training Mean Baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=5, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=160,
                max_depth=16,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=160, learning_rate=0.05, max_depth=3, random_state=42
            ),
        }
        if XGB_AVAILABLE:
            estimators["XGBoost"] = XGBRegressor(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=4,
            )
        return estimators


class BlendedDemandModel:
    def __init__(self, model, alpha, fallback_demand, lag_column="Lag_1"):
        self.model = model
        self.alpha = float(alpha)
        self.fallback_demand = float(fallback_demand)
        self.lag_column = lag_column

    def predict(self, features):
        lag = pd.to_numeric(
            features.get(
                self.lag_column,
                pd.Series(self.fallback_demand, index=features.index),
            ),
            errors="coerce",
        ).fillna(self.fallback_demand).to_numpy(dtype=float)
        if self.model is None:
            return np.maximum(lag, 0)
        predicted = np.maximum(np.asarray(self.model.predict(features), dtype=float), 0)
        return np.maximum(self.alpha * predicted + (1 - self.alpha) * lag, 0)


class ConstantDemandModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def _preprocessor(df, features):
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def _forecast_metrics(actual, frame, predicted):
    demand = np.asarray(predicted, dtype=float)
    observed = np.asarray(actual, dtype=float)
    nonnegative = np.maximum(demand, 0)
    prices = frame["Selling_Price"].to_numpy(dtype=float)
    unit_margin = prices - frame["Unit_Cost"].to_numpy(dtype=float)
    return {
        "MAE": mean_absolute_error(observed, demand),
        "RMSE": mean_squared_error(observed, demand) ** 0.5,
        "R2": r2_score(observed, demand),
        "Revenue_MAE": mean_absolute_error(frame["Revenue"], nonnegative * prices),
        "Contribution_MAE": mean_absolute_error(
            frame["Estimated_Contribution"], nonnegative * unit_margin
        ),
    }


def _lag_predictions(frame, history, lag, fallback):
    prior = history.groupby(
        ["Date", "Product_ID", "Store_ID"], as_index=False
    )["Quantity_Sold"].sum()
    prior["Date"] = prior["Date"] + pd.Timedelta(days=lag)
    prior = prior.rename(columns={"Quantity_Sold": "_lag_demand"})
    joined = frame[["Date", "Product_ID", "Store_ID"]].merge(
        prior,
        on=["Date", "Product_ID", "Store_ID"],
        how="left",
        validate="many_to_one",
    )
    return joined["_lag_demand"].fillna(fallback).to_numpy(dtype=float)


def model_demand(df, estimator_provider: DemandEstimatorProvider | None = None):
    provider = (
        estimator_provider
        if estimator_provider is not None
        else DefaultDemandEstimatorProvider()
    )
    estimators = provider.build_estimators()
    if not estimators or set(estimators) == {"Training Mean Baseline"}:
        raise ValueError("The estimator provider must supply at least one demand model.")
    data = prepare_demand_data(df).sort_values("Date").reset_index(drop=True)
    features = demand_features(data)
    train, validation, train_validation, test = split_for_backtest(data)
    train_mean = float(train["Quantity_Sold"].mean())
    final_mean = float(train_validation["Quantity_Sold"].mean())
    validation_lag1 = validation["Lag_1"].fillna(train_mean).to_numpy(dtype=float)
    test_lag1 = test["Lag_1"].fillna(final_mean).to_numpy(dtype=float)
    validation_lag7 = _lag_predictions(validation, data, 7, train_mean)
    test_lag7 = _lag_predictions(test, data, 7, final_mean)
    alpha_values = np.arange(0, 1.001, 0.05)
    candidates = {}
    rows = []
    trained = {}

    baseline = np.full(len(test), final_mean)
    rows.append(
        {
            "Model": "Training Mean Baseline",
            "Validation_RMSE": mean_squared_error(
                validation["Quantity_Sold"], np.full(len(validation), train_mean)
            ) ** 0.5,
            **_forecast_metrics(test["Quantity_Sold"], test, baseline),
        }
    )
    rows.append(
        {
            "Model": "Previous-Day Naive",
            "Validation_RMSE": mean_squared_error(
                validation["Quantity_Sold"], validation_lag1
            ) ** 0.5,
            **_forecast_metrics(test["Quantity_Sold"], test, test_lag1),
        }
    )
    rows.append(
        {
            "Model": "Seasonal Naive (7-day)",
            "Validation_RMSE": mean_squared_error(
                validation["Quantity_Sold"], validation_lag7
            ) ** 0.5,
            **_forecast_metrics(test["Quantity_Sold"], test, test_lag7),
        }
    )
    trained["Training Mean Baseline"] = ConstantDemandModel(final_mean)
    trained["Previous-Day Naive"] = BlendedDemandModel(None, 0, final_mean)
    trained["Seasonal Naive (7-day)"] = BlendedDemandModel(
        None, 0, final_mean, "Lag_7"
    )

    for name, estimator in estimators.items():
        if name == "Training Mean Baseline":
            continue
        pipeline = Pipeline(
            [("preprocessor", _preprocessor(train, features)), ("model", estimator)]
        )
        pipeline.fit(train[features], train["Quantity_Sold"])
        validation_prediction = np.maximum(pipeline.predict(validation[features]), 0)
        alpha = min(
            alpha_values,
            key=lambda value: mean_squared_error(
                validation["Quantity_Sold"],
                value * validation_prediction + (1 - value) * validation_lag1,
            ),
        )
        validation_blend = np.maximum(
            alpha * validation_prediction + (1 - alpha) * validation_lag1, 0
        )
        validation_metrics = _forecast_metrics(
            validation["Quantity_Sold"], validation, validation_blend
        )
        raw_validation_metrics = _forecast_metrics(
            validation["Quantity_Sold"], validation, validation_prediction
        )
        candidates[name] = {
            "estimator": estimator,
            "alpha": alpha,
            "validation_rmse": validation_metrics["RMSE"],
            "validation_mae": validation_metrics["MAE"],
        }
        rows.append(
            {
                "Model": name,
                "Validation_RMSE": raw_validation_metrics["RMSE"],
                "Validation_MAE": raw_validation_metrics["MAE"],
                **{
                    key: np.nan
                    for key in ["MAE", "RMSE", "R2", "Revenue_MAE", "Contribution_MAE"]
                },
            }
        )

    selected_name = min(
        candidates, key=lambda name: candidates[name]["validation_rmse"]
    )
    selected = candidates[selected_name]
    hybrid_name = f"Tuned Hybrid ({selected_name} + previous-day)"
    rows.append(
        {
            "Model": hybrid_name,
            "Validation_RMSE": selected["validation_rmse"],
            "Validation_MAE": selected["validation_mae"],
            **{
                key: np.nan
                for key in ["MAE", "RMSE", "R2", "Revenue_MAE", "Contribution_MAE"]
            },
        }
    )

    for name, candidate in candidates.items():
        pipeline = Pipeline(
            [
                ("preprocessor", _preprocessor(train_validation, features)),
                ("model", candidate["estimator"]),
            ]
        )
        pipeline.fit(train_validation[features], train_validation["Quantity_Sold"])
        trained[name] = pipeline
        prediction = np.maximum(pipeline.predict(test[features]), 0)
        row = next(row for row in rows if row["Model"] == name)
        row.update(_forecast_metrics(test["Quantity_Sold"], test, prediction))

    hybrid = BlendedDemandModel(trained[selected_name], selected["alpha"], final_mean)
    trained[hybrid_name] = hybrid
    hybrid_prediction = hybrid.predict(test)
    hybrid_row = next(row for row in rows if row["Model"] == hybrid_name)
    hybrid_row.update(_forecast_metrics(test["Quantity_Sold"], test, hybrid_prediction))
    hybrid_row["Selected"] = True

    for row in rows:
        row.setdefault("Validation_MAE", np.nan)
        row.setdefault("Selected", False)
    results = pd.DataFrame(rows).sort_values(
        ["Selected", "Validation_RMSE"], ascending=[False, True]
    ).reset_index(drop=True)
    return results, trained