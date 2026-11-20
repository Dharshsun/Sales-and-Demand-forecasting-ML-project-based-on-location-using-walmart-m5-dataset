"""Load and normalize the Hugging Face Online Retail transactions."""

from pathlib import Path

import pandas as pd


def load_online_retail(path):
    """Read the UCI Online Retail workbook and return daily product observations."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, parse_dates=["Date"])
        required = [
            "Date",
            "Product_ID",
            "Category",
            "Store_ID",
            "Selling_Price",
            "Quantity_Sold",
            "Revenue",
        ]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Normalized Online Retail CSV is missing columns: {missing}")
        return data

    source = pd.read_excel(path)
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Online Retail workbook is missing columns: {missing}")

    return normalize_online_retail(source)


def normalize_online_retail(source):
    """Normalize Online Retail rows to the pricing app's expected data schema."""
    data = source.copy()
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Online Retail data is missing columns: {missing}")

    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["Date"] = pd.to_datetime(data["InvoiceDate"], errors="coerce").dt.normalize()
    data = data[
        ~data["InvoiceNo"].str.startswith("C", na=False)
        & (data["Quantity"] > 0)
        & (data["UnitPrice"] > 0)
    ].dropna(subset=["Date", "StockCode", "Quantity", "UnitPrice"])

    if data.empty:
        raise ValueError("Online Retail data contains no valid positive-quantity sales.")

    data["Product_ID"] = data["StockCode"].astype(str)
    if "Country" in data:
        data["Store_ID"] = data["Country"].fillna("Unknown").astype(str)
    else:
        data["Store_ID"] = "Online"
    data["Category"] = "Online Retail"
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    reference_prices = data.groupby("Product_ID")["UnitPrice"].transform("median")
    data["Base_Price"] = reference_prices
    data["Selling_Price"] = data["UnitPrice"]
    data["Discount_Pct"] = (
        100 * (1 - data["Selling_Price"] / data["Base_Price"])
    ).clip(lower=0)
    data["Promotion"] = (data["Discount_Pct"] > 0).astype(int)
    data["Quantity_Sold"] = data["Quantity"]
    data["Month"] = data["Date"].dt.month
    data["Day_of_Week"] = data["Date"].dt.dayofweek
    data["Weekend"] = (data["Day_of_Week"] >= 5).astype(int)
    data["Holiday_Event"] = 0

    daily = data.groupby(
        ["Date", "Product_ID", "Store_ID", "Category"],
        as_index=False,
        sort=True,
    ).agg(
        Base_Price=("Base_Price", "first"),
        Revenue=("Revenue", "sum"),
        Quantity_Sold=("Quantity_Sold", "sum"),
        Promotion=("Promotion", "max"),
        Discount_Pct=("Discount_Pct", "mean"),
    )
    daily["Selling_Price"] = daily["Revenue"] / daily["Quantity_Sold"]
    daily["Month"] = daily["Date"].dt.month
    daily["Day_of_Week"] = daily["Date"].dt.dayofweek
    daily["Weekend"] = (daily["Day_of_Week"] >= 5).astype(int)
    daily["Holiday_Event"] = 0
    return daily[
        [
            "Date",
            "Product_ID",
            "Category",
            "Store_ID",
            "Month",
            "Day_of_Week",
            "Weekend",
            "Promotion",
            "Holiday_Event",
            "Base_Price",
            "Discount_Pct",
            "Selling_Price",
            "Quantity_Sold",
            "Revenue",
        ]
    ]
