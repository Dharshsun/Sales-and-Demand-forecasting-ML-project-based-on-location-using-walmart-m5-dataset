import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# Configuration
# -----------------------------
n = 60000

dates = pd.date_range(
    start="2023-01-01",
    end="2025-12-31",
    freq="D"
)

cats = [
    "Electronics",
    "Grocery",
    "Home",
    "Beauty",
    "Sports"
]

products = [f"P{i:03d}" for i in range(1, 51)]
stores = [f"S{i:02d}" for i in range(1, 16)]

# Category demand effects
effects = {
    "Electronics": 1.15,
    "Grocery": 1.35,
    "Home": 0.95,
    "Beauty": 1.05,
    "Sports": 0.90
}

rows = []

# -----------------------------
# Generate dataset
# -----------------------------
for _ in range(n):

    # Convert numpy datetime64 -> pandas Timestamp
    date = pd.Timestamp(np.random.choice(dates))

    cat = np.random.choice(cats)
    product = np.random.choice(products)
    store = np.random.choice(stores)

    # -------------------------
    # Time features
    # -------------------------
    month = date.month
    day_of_week = date.weekday()

    weekend = int(day_of_week >= 5)

    # -------------------------
    # Pricing
    # -------------------------
    base = np.random.uniform(10, 200)

    discount = np.random.choice(
        [0, 0, 0, 5, 10, 15, 20, 25],
        p=[0.30, 0.20, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05]
    )

    promo = int(
        discount >= 10 or np.random.rand() < 0.08
    )

    holiday = int(
        np.random.rand() < 0.08
    )

    selling_price = base * (1 - discount / 100)

    # -------------------------
    # Competitor price
    # -------------------------
    competitor_price = base * np.random.uniform(
        0.90,
        1.10
    )

    competitor_price_index = competitor_price / base

    # -------------------------
    # Store demand effect
    # -------------------------
    store_demand_index = np.random.uniform(
        0.80,
        1.25
    )

    # -------------------------
    # Demand generation
    # Price elasticity = -0.65
    # -------------------------
    demand = (
        35
        * effects[cat]
        * store_demand_index
        * (selling_price / 80) ** (-0.65)
        * (1 + 0.15 * promo)
        * (1 + 0.08 * holiday)
        * (1 + 0.06 * weekend)
        * np.random.lognormal(0, 0.18)
    )

    qty = max(
        1,
        int(round(demand))
    )

    # -------------------------
    # Unit cost
    # -------------------------
    unit_cost = (
        base
        * np.random.uniform(0.48, 0.72)
    )

    # -------------------------
    # Revenue and contribution
    # -------------------------
    revenue = selling_price * qty

    estimated_cost = unit_cost * qty

    estimated_contribution = (
        revenue - estimated_cost
    )

    margin_pct = (
        100 * estimated_contribution / revenue
        if revenue > 0
        else 0
    )

    # -------------------------
    # Inventory
    # -------------------------
    inventory_available = int(
        qty * np.random.uniform(1.0, 1.5)
    )

    stockout_flag = int(
        inventory_available < qty
    )

    estimated_lost_sales = max(
        0,
        qty - inventory_available
    )

    # -------------------------
    # Append
    # -------------------------
    rows.append([
        date,
        product,
        cat,
        store,
        month,
        day_of_week,
        weekend,
        promo,
        holiday,
        base,
        discount,
        selling_price,
        unit_cost,
        qty,
        revenue,
        estimated_cost,
        estimated_contribution,
        margin_pct,
        competitor_price,
        competitor_price_index,
        store_demand_index,
        inventory_available,
        stockout_flag,
        estimated_lost_sales
    ])


# -----------------------------
# Create DataFrame
# -----------------------------
columns = [
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
    "Unit_Cost",
    "Quantity_Sold",
    "Revenue",
    "Estimated_Cost",
    "Estimated_Contribution",
    "Margin_Pct",
    "Competitor_Price",
    "Competitor_Price_Index",
    "Store_Demand_Index",
    "Inventory_Available",
    "Stockout_Flag",
    "Estimated_Lost_Sales"
]

df = pd.DataFrame(
    rows,
    columns=columns
)

# -----------------------------
# Sort by date
# -----------------------------
df = df.sort_values(
    "Date"
).reset_index(drop=True)

# -----------------------------
# Save
# -----------------------------
output_file = "retail_pricing_profitability_research_data.csv"

df.to_csv(
    output_file,
    index=False
)

print("=" * 60)
print("Synthetic Pricing & Profitability Dataset Generated")
print("=" * 60)

print(f"Records : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Output  : {output_file}")

print("\nCategories:")
print(df["Category"].value_counts())

print("\nDate Range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nPreview:")
print(df.head())

print("\nDataset shape:")
print(df.shape)