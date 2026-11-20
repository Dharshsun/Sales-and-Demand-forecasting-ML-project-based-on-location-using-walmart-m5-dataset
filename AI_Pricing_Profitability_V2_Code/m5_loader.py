import pandas as pd


def load_m5_for_pricing(sales_path, prices_path, calendar_path, max_series=5000):
    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(prices_path)
    calendar = pd.read_csv(calendar_path)
    identifiers = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    if max_series and len(sales) > max_series:
        sales = sales.sample(max_series, random_state=42)

    long = sales.melt(
        id_vars=identifiers,
        value_vars=day_columns,
        var_name="d",
        value_name="Quantity_Sold",
    )
    calendar_columns = [
        column
        for column in [
            "d", "date", "wm_yr_wk", "event_name_1", "event_type_1",
            "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
        ]
        if column in calendar
    ]
    long = long.merge(calendar[calendar_columns], on="d", how="left")
    long = long.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long["Date"] = pd.to_datetime(long.date)
    long["Product_ID"] = long.item_id.astype(str)
    long["Store_ID"] = long.store_id.astype(str)
    long["Category"] = long.cat_id.astype(str)
    long["Selling_Price"] = pd.to_numeric(long.sell_price, errors="coerce")
    long = long.dropna(subset=["Selling_Price", "Quantity_Sold"])

    reference_price = long.groupby(["Product_ID", "Store_ID"])[
        "Selling_Price"
    ].transform("median")
    long["Base_Price"] = reference_price
    long["Discount_Pct"] = (
        100 * (1 - long["Selling_Price"] / reference_price)
    ).clip(lower=0)
    long["Promotion"] = (long["Selling_Price"] < reference_price).astype(int)
    long["Holiday_Event"] = long.get(
        "event_name_1", pd.Series(index=long.index)
    ).notna().astype(int)
    long["Month"] = long.Date.dt.month
    long["Day_of_Week"] = long.Date.dt.dayofweek
    long["Weekend"] = (long.Day_of_Week >= 5).astype(int)
    long["Revenue"] = long.Selling_Price * long.Quantity_Sold

    return long[
        [
            "Date", "Product_ID", "Category", "Store_ID", "Month", "Day_of_Week",
            "Weekend", "Promotion", "Holiday_Event", "Base_Price", "Discount_Pct",
            "Selling_Price", "Quantity_Sold", "Revenue",
        ]
    ]
