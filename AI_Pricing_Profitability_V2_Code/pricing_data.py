"""Data validation, feature preparation, and temporal splitting."""

import numpy as np
import pandas as pd


def prepare_pricing_data(df):
    data = df.copy()
    required = ["Date", "Product_ID", "Store_ID", "Category", "Selling_Price", "Quantity_Sold"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ["Selling_Price", "Quantity_Sold"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "Base_Price" not in data:
        data["Base_Price"] = data["Selling_Price"]
    if "Unit_Cost" not in data:
        data["Unit_Cost"] = data["Selling_Price"] * 0.65
    if "Promotion" not in data:
        data["Promotion"] = 0
    if "Discount_Pct" not in data:
        data["Discount_Pct"] = (
            1 - data["Selling_Price"] / data["Base_Price"].replace(0, np.nan)
        ).clip(lower=0) * 100
        data["Discount_Pct"] = data["Discount_Pct"].fillna(0)
    if "Revenue" not in data:
        data["Revenue"] = data["Selling_Price"] * data["Quantity_Sold"]
    if "Estimated_Cost" not in data:
        data["Estimated_Cost"] = data["Unit_Cost"] * data["Quantity_Sold"]
    if "Estimated_Contribution" not in data:
        data["Estimated_Contribution"] = data["Revenue"] - data["Estimated_Cost"]
    if "Margin_Pct" not in data:
        data["Margin_Pct"] = np.where(
            data["Revenue"] != 0,
            100 * data["Estimated_Contribution"] / data["Revenue"],
            0,
        )
    if "Month" not in data:
        data["Month"] = data.Date.dt.month
    if "Day_of_Week" not in data:
        data["Day_of_Week"] = data.Date.dt.dayofweek
    if "Weekend" not in data:
        data["Weekend"] = (data.Day_of_Week >= 5).astype(int)
    if "Holiday_Event" not in data:
        data["Holiday_Event"] = 0
    for column in ["Product_ID", "Store_ID", "Category"]:
        data[column] = data[column].astype(str)
    data = data.drop(columns=["Lag_1", "Lag_7"], errors="ignore")
    return data.dropna(subset=["Date", "Selling_Price", "Quantity_Sold"])


def prepare_demand_data(df):
    data = prepare_pricing_data(df)
    keys = ["Date", "Product_ID", "Store_ID"]
    daily = data.groupby(keys, as_index=False, sort=True).agg(
        Category=("Category", "first"),
        Month=("Month", "first"),
        Day_of_Week=("Day_of_Week", "first"),
        Weekend=("Weekend", "first"),
        Holiday_Event=("Holiday_Event", "max"),
        Base_Price=("Base_Price", "mean"),
        Discount_Pct=("Discount_Pct", "mean"),
        Promotion=("Promotion", "mean"),
        Quantity_Sold=("Quantity_Sold", "sum"),
        Revenue=("Revenue", "sum"),
        Estimated_Cost=("Estimated_Cost", "sum"),
        Average_Selling_Price=("Selling_Price", "mean"),
    )
    daily["Selling_Price"] = np.where(
        daily["Quantity_Sold"] > 0,
        daily["Revenue"] / daily["Quantity_Sold"],
        daily["Average_Selling_Price"],
    )
    daily["Unit_Cost"] = np.where(
        daily["Quantity_Sold"] > 0,
        daily["Estimated_Cost"] / daily["Quantity_Sold"],
        daily["Selling_Price"] * 0.65,
    )
    daily["Estimated_Contribution"] = daily["Revenue"] - daily["Estimated_Cost"]
    daily["Margin_Pct"] = np.where(
        daily["Revenue"] != 0,
        100 * daily["Estimated_Contribution"] / daily["Revenue"],
        0,
    )
    history = daily[keys + ["Quantity_Sold"]]
    for lag in (1, 7):
        prior = history.copy()
        prior["Date"] = prior["Date"] + pd.Timedelta(days=lag)
        daily = daily.merge(
            prior.rename(columns={"Quantity_Sold": f"Lag_{lag}"}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return daily


def demand_features(df):
    return [
        column
        for column in [
            "Selling_Price", "Base_Price", "Discount_Pct", "Promotion", "Month",
            "Day_of_Week", "Weekend", "Holiday_Event", "Product_ID", "Store_ID",
            "Category", "Lag_1", "Lag_7",
        ]
        if column in df.columns
    ]


def split_by_date(df, test_fraction=0.2):
    dates = np.sort(df["Date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required for a temporal holdout.")
    split_index = min(max(int(np.floor(len(dates) * (1 - test_fraction))), 1), len(dates) - 1)
    first_test_date = pd.Timestamp(dates[split_index])
    train = df[df["Date"] < first_test_date].copy()
    test = df[df["Date"] >= first_test_date].copy()
    return train, test


def split_for_backtest(df, validation_fraction=0.2, test_fraction=0.2):
    dates = np.sort(df["Date"].dropna().unique())
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for train/validation/test backtesting.")
    validation_index = min(
        max(int(np.floor(len(dates) * (1 - validation_fraction - test_fraction))), 1),
        len(dates) - 2,
    )
    test_index = min(
        max(int(np.floor(len(dates) * (1 - test_fraction))), validation_index + 1),
        len(dates) - 1,
    )
    validation_start = pd.Timestamp(dates[validation_index])
    test_start = pd.Timestamp(dates[test_index])
    train = df[df["Date"] < validation_start].copy()
    validation = df[(df["Date"] >= validation_start) & (df["Date"] < test_start)].copy()
    train_validation = df[df["Date"] < test_start].copy()
    test = df[df["Date"] >= test_start].copy()
    return train, validation, train_validation, test