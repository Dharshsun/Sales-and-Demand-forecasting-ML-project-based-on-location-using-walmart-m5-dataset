"""Pricing metrics, scenario analysis, and business interpretation."""

import numpy as np
import pandas as pd

from model_contracts import DemandPredictor
from pricing_data import prepare_demand_data, prepare_pricing_data


def calculate_price_elasticity(df, group_col=None, return_table=False):
    data = prepare_pricing_data(df)

    def calculate(group):
        group = group.sort_values("Date")
        prices = group.Selling_Price.replace(0, np.nan)
        quantity = group.Quantity_Sold.replace(0, np.nan)
        valid = prices.notna() & quantity.notna()
        prices = prices[valid]
        quantity = quantity[valid]
        if len(prices) < 8 or prices.nunique() < 2 or quantity.nunique() < 2:
            return np.nan
        return np.polyfit(np.log(prices), np.log(quantity), 1)[0]

    if group_col is None:
        return float(calculate(data))
    result = (
        data.groupby(group_col)
        .apply(calculate, include_groups=False)
        .reset_index(name="Elasticity")
    )
    return result.sort_values("Elasticity") if return_table else result.Elasticity.mean()


def calculate_promotion_lift(df):
    data = prepare_pricing_data(df)
    promotion = data[data.Promotion == 1]
    non_promotion = data[data.Promotion == 0]

    def lift(promoted, baseline):
        if baseline == 0:
            return np.nan
        return 100 * (promoted - baseline) / baseline

    comparison = pd.DataFrame(
        {
            "Metric": ["Units", "Revenue", "Contribution", "Avg Selling Price"],
            "Promotion": [
                promotion.Quantity_Sold.mean(),
                promotion.Revenue.mean(),
                promotion.Estimated_Contribution.mean(),
                promotion.Selling_Price.mean(),
            ],
            "Non-Promotion": [
                non_promotion.Quantity_Sold.mean(),
                non_promotion.Revenue.mean(),
                non_promotion.Estimated_Contribution.mean(),
                non_promotion.Selling_Price.mean(),
            ],
        }
    )
    return {
        "comparison": comparison,
        "demand_lift_pct": lift(
            promotion.Quantity_Sold.mean(), non_promotion.Quantity_Sold.mean()
        ),
        "revenue_lift_pct": lift(
            promotion.Revenue.mean(), non_promotion.Revenue.mean()
        ),
        "contribution_lift_pct": lift(
            promotion.Estimated_Contribution.mean(),
            non_promotion.Estimated_Contribution.mean(),
        ),
    }


def assess_pricing_constraints(
    df, min_margin=15, max_discount=30, max_price_change=15
):
    """Summarize historical rows meeting all selected pricing constraints."""
    data = prepare_pricing_data(df)
    reference_price = data["Base_Price"].replace(0, np.nan)
    data["Reference_Price_Change_Pct"] = (
        100 * (data["Selling_Price"] / reference_price - 1)
    ).abs()
    eligible_mask = (
        (data["Margin_Pct"] >= min_margin)
        & (data["Discount_Pct"] <= max_discount)
        & (data["Reference_Price_Change_Pct"] <= max_price_change)
    )
    eligible = data.loc[eligible_mask]
    categorized = data.assign(
        Eligible_Records=eligible_mask.astype(int),
        Excluded_Records=(~eligible_mask).astype(int),
        Eligible_Contribution=data["Estimated_Contribution"].where(eligible_mask, 0),
    )
    category_summary = (
        categorized.groupby("Category", as_index=False)
        .agg(
            Eligible_Records=("Eligible_Records", "sum"),
            Excluded_Records=("Excluded_Records", "sum"),
            Eligible_Contribution=("Eligible_Contribution", "sum"),
        )
        .sort_values("Eligible_Records", ascending=False)
    )
    contribution = float(eligible["Estimated_Contribution"].sum())
    if eligible.empty:
        status = "Inefficient: no observed rows meet all selected constraints"
    elif contribution > 0:
        status = "Efficient under selected constraints: positive estimated contribution"
    else:
        status = "Inefficient: feasible rows have zero or negative estimated contribution"
    return {
        "status": status,
        "eligible_rows": len(eligible),
        "total_rows": len(data),
        "eligible_share_pct": 100 * len(eligible) / len(data) if len(data) else 0,
        "estimated_contribution": contribution,
        "category_summary": category_summary,
        "average_contribution": (
            float(eligible["Estimated_Contribution"].mean())
            if not eligible.empty
            else 0.0
        ),
    }


def _row(df, product, store):
    selected = df[
        (df.Product_ID == str(product)) & (df.Store_ID == str(store))
    ]
    if selected.empty:
        selected = df[df.Product_ID == str(product)]
    if selected.empty:
        selected = df
    return selected.sort_values("Date").iloc[-1].copy()


def build_price_response_curve(
    df: pd.DataFrame,
    model: DemandPredictor,
    product_id,
    store_id,
    max_price_change=15,
    points=31,
):
    data = prepare_demand_data(df)
    row = _row(data, product_id, store_id)
    scenarios = []
    for change in np.linspace(-max_price_change, max_price_change, points):
        price = float(row.Selling_Price) * (1 + change / 100)
        scenario = row.copy()
        scenario.Selling_Price = price
        scenario.Discount_Pct = max(
            0, (1 - price / float(row.Base_Price)) * 100
        )
        demand = max(0, float(model.predict(pd.DataFrame([scenario]))[0]))
        revenue = price * demand
        cost = float(row.Unit_Cost) * demand
        contribution = revenue - cost
        scenarios.append(
            {
                "Price_Change_Pct": change,
                "Price": price,
                "Predicted_Demand": demand,
                "Revenue": revenue,
                "Estimated_Cost": cost,
                "Contribution": contribution,
                "Margin_Pct": 100 * contribution / revenue if revenue else 0,
                "Discount_Pct": scenario.Discount_Pct,
            }
        )
    return pd.DataFrame(scenarios)


def run_sensitivity_analysis(
    df,
    model: DemandPredictor,
    product_id,
    store_id,
    min_margin=15,
    max_discount=30,
    max_price_change=20,
    points=21,
):
    curve = build_price_response_curve(
        df, model, product_id, store_id, max_price_change, points
    )
    rows = []
    demand_cases = [
        (0.8, "Conservative: demand 20% below forecast"),
        (1.0, "Base: model forecast"),
        (1.2, "Optimistic: demand 20% above forecast"),
    ]
    for factor, label in demand_cases:
        scenario = curve.copy()
        scenario["Adjusted_Demand"] = scenario.Predicted_Demand * factor
        scenario["Adjusted_Revenue"] = scenario.Price * scenario.Adjusted_Demand
        unit_cost = np.where(
            scenario.Predicted_Demand > 0,
            scenario.Estimated_Cost / scenario.Predicted_Demand,
            0,
        )
        scenario["Adjusted_Contribution"] = (
            scenario.Adjusted_Revenue - unit_cost * scenario.Adjusted_Demand
        )
        feasible = scenario[
            (scenario.Margin_Pct >= min_margin)
            & (scenario.Discount_Pct <= max_discount)
        ]
        if not feasible.empty:
            best = feasible.loc[feasible.Adjusted_Contribution.idxmax()]
            rows.append(
                {
                    "Scenario": label,
                    "Price": best.Price,
                    "Demand": best.Adjusted_Demand,
                    "Revenue": best.Adjusted_Revenue,
                    "Contribution": best.Adjusted_Contribution,
                    "Margin_Pct": best.Margin_Pct,
                }
            )
    return pd.DataFrame(rows)


def explain_model(df, predictor):
    from demand_forecasting import BlendedDemandModel, ConstantDemandModel

    if isinstance(predictor, BlendedDemandModel):
        if predictor.model is None:
            return pd.DataFrame(columns=["Feature", "Importance"])
        predictor = predictor.model
    if isinstance(predictor, ConstantDemandModel):
        return pd.DataFrame(columns=["Feature", "Importance"])

    preprocessor = predictor.named_steps["preprocessor"]
    model = predictor.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["Feature", "Importance"])
    names = list(preprocessor.get_feature_names_out())
    values = (
        model.feature_importances_
        if hasattr(model, "feature_importances_")
        else np.abs(np.asarray(model.coef_).ravel())
    )
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )


def business_segments(df):
    data = prepare_pricing_data(df)
    summary = (
        data.groupby("Category")
        .agg(
            Avg_Price=("Selling_Price", "mean"),
            Avg_Demand=("Quantity_Sold", "mean"),
            Revenue=("Revenue", "sum"),
            Contribution=("Estimated_Contribution", "sum"),
            Promotion_Rate=("Promotion", "mean"),
        )
        .reset_index()
    )
    elasticity = calculate_price_elasticity(data, "Category", True)
    result = summary.merge(elasticity, on="Category", how="left")

    def classify(row):
        if pd.isna(row.Elasticity):
            return "Insufficient elasticity evidence"
        if abs(row.Elasticity) >= 1 and row.Promotion_Rate >= 0.2:
            return "Price-sensitive & promotion-sensitive"
        if abs(row.Elasticity) >= 1:
            return "Price-sensitive"
        if row.Promotion_Rate >= 0.2:
            return "Promotion-sensitive"
        return "Relatively lower price sensitivity"

    result["Business_Segment"] = result.apply(classify, axis=1)
    return result.sort_values("Revenue", ascending=False)


def generate_business_insights(df, segments):
    return [
        "Evaluate pricing using demand, revenue and contribution together rather than a single KPI.",
        "Price-sensitive categories require caution with price increases because their demand changes more strongly with price.",
        "Promotion-sensitive categories should be evaluated on contribution, not unit sales alone.",
        "Observed elasticity is not automatically causal because price changes can co-move with promotion, seasonality, inventory and management decisions.",
        "Real deployment should add inventory, competitor pricing, price floors, customer fairness and governance constraints.",
    ]