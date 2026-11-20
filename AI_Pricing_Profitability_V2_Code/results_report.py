from datetime import datetime
import numbers
from pathlib import Path

import pandas as pd

from pricing_analysis import (
    build_price_response_curve,
    calculate_price_elasticity,
    calculate_promotion_lift,
    explain_model,
    prepare_demand_data,
    prepare_pricing_data,
    run_sensitivity_analysis,
    split_for_backtest,
)


def _format_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, numbers.Integral):
        return f"{value:,}"
    if isinstance(value, numbers.Real):
        return f"{value:,.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(columns, rows):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_latest_results_report(
    df,
    source,
    source_path,
    model_results,
    trained_models,
    output_path,
    min_margin=15,
    max_discount=30,
    max_price_change=15,
    usd_to_inr=83.0,
    display_currency="INR",
):
    data = prepare_pricing_data(df)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    model_results = model_results.sort_values("RMSE").reset_index(drop=True)
    selected_row = model_results.loc[model_results["Selected"]].iloc[0]
    best_model_name = selected_row["Model"]
    best_model = trained_models[best_model_name]

    forecast_data = prepare_demand_data(data)
    train, validation, train_validation, test = split_for_backtest(forecast_data)
    train_count = len(train)
    validation_count = len(validation)
    refit_count = len(train_validation)
    test_count = len(test)
    train_end = train["Date"].max().date()
    validation_start = validation["Date"].min().date()
    validation_end = validation["Date"].max().date()
    test_start = test["Date"].min().date()
    test_end = test["Date"].max().date()

    elasticity = calculate_price_elasticity(data)
    category_elasticity = calculate_price_elasticity(data, "Category", True)
    category_elasticity = category_elasticity.dropna(subset=["Elasticity"])
    category_elasticity = category_elasticity.reindex(
        category_elasticity["Elasticity"].abs().sort_values(ascending=False).index
    ).head(10)

    promotion = calculate_promotion_lift(data)
    importance = explain_model(data, best_model).head(10)
    importance_table = (
        "The selected model is a constant-mean baseline and has no feature-importance values."
        if importance.empty
        else _markdown_table(
            ["Feature", "Importance"],
            importance[["Feature", "Importance"]].itertuples(index=False, name=None),
        )
    )
    selection_detail = (
        f"The validation-selected method blends {best_model.alpha:.0%} model demand with "
        f"{1-best_model.alpha:.0%} previous-day demand."
        if hasattr(best_model, "alpha") and best_model.alpha < 1
        else "The validation-selected method is used without a previous-day blend."
    )

    pair_counts = data.groupby(["Product_ID", "Store_ID"]).size()
    product_id, store_id = pair_counts.idxmax()
    curve = build_price_response_curve(
        data,
        best_model,
        product_id,
        store_id,
        max_price_change=max_price_change,
        points=31,
    )
    feasible = curve[
        (curve["Margin_Pct"] >= min_margin)
        & (curve["Discount_Pct"] <= max_discount)
    ]
    scenario_rows = []
    if not feasible.empty:
        for label, metric in (
            ("Highest revenue", "Revenue"),
            ("Highest contribution", "Contribution"),
        ):
            scenario = feasible.loc[feasible[metric].idxmax()]
            scenario_rows.append(
                [
                    label,
                    scenario["Price_Change_Pct"],
                    scenario["Price"],
                    scenario["Predicted_Demand"],
                    scenario["Revenue"] * usd_to_inr,
                    scenario["Contribution"] * usd_to_inr,
                    scenario["Margin_Pct"],
                ]
            )

    sensitivity = run_sensitivity_analysis(
        data,
        best_model,
        str(product_id),
        str(store_id),
        min_margin=min_margin,
        max_discount=max_discount,
        max_price_change=max_price_change,
    )

    total_revenue = data["Revenue"].sum() * usd_to_inr
    total_contribution = data["Estimated_Contribution"].sum() * usd_to_inr
    series_count = data[["Product_ID", "Store_ID"]].drop_duplicates().shape[0]
    first_date = data["Date"].min().date()
    last_date = data["Date"].max().date()
    if source == "M5 Walmart dataset":
        cost_limitation = "- M5 does not provide actual product-level cost; the prototype estimates unit cost as 65% of selling price, so contribution is not actual profit."
        data_limitation = "- Results are based on the sampled M5 item-store series listed above and do not represent the complete M5 dataset."
        promotion_limitation = "- M5 does not provide an explicit promotion label or base price; discounts and promotions are inferred relative to each sampled item-store median price and are not verified Walmart promotions or discounts."
    elif source == "Hugging Face Online Retail dataset":
        cost_limitation = "- The source has no product costs. Unit cost is estimated as 65% of selling price, so estimated contribution is not actual profit."
        data_limitation = "- This is a separate UCI Online Retail benchmark of UK transactions, not Walmart data; it must not be used to fill missing M5 observations."
        promotion_limitation = "- The source has no explicit promotion label or list price; discount and promotion fields are inferred from each product's median transaction price."
    else:
        cost_limitation = "- Unit costs in the synthetic dataset are generated assumptions; estimated contribution is not actual retailer profit."
        data_limitation = "- Synthetic data results are generated from assumed relationships and are not empirical evidence about a real retailer."
        promotion_limitation = "- Promotion results in synthetic data reflect the dataset's generated assumptions and are not causal evidence."
    currency_limitation = (
        "- Financial amounts are reported in GBP as provided by the source; no currency conversion was applied."
        if display_currency == "GBP"
        else f"- Financial amounts are multiplied by {usd_to_inr:.1f} for the app's INR display; this is not a verified exchange-rate conversion."
    )
    report = [
        "# Latest Pricing Analysis Results",
        "",
        f"Generated: {generated_at}",
        f"Data source: {source}",
        f"Input file(s): {source_path}",
        "",
        "## Synopsis-Ready Summary",
        "",
        (
            f"Using {len(data):,} records from {first_date} to {last_date}, this analysis "
            f"compared demand predictors using chronological training, validation and "
            f"final test periods. {best_model_name} was selected on validation dates and "
            f"achieved final-test RMSE {selected_row['RMSE']:.2f} units. The estimated overall price "
            f"elasticity was {elasticity:.3f}; the observed promotion-period differences "
            f"were {promotion['demand_lift_pct']:.2f}% for demand, "
            f"{promotion['revenue_lift_pct']:.2f}% for revenue, and "
            f"{promotion['contribution_lift_pct']:.2f}% for estimated contribution. "
            "Model-based price scenarios are estimates for decision support, not causal "
            "effects or guaranteed financial outcomes."
        ),
        "",
        "## Dataset Profile",
        "",
        _markdown_table(
            ["Measure", "Result"],
            [
                ["Records", len(data)],
                ["Date range", f"{first_date} to {last_date}"],
                ["Products", data["Product_ID"].nunique()],
                ["Stores", data["Store_ID"].nunique()],
                ["Item-store series", series_count],
                ["Categories", data["Category"].nunique()],
                ["Units sold", data["Quantity_Sold"].sum()],
                [f"Revenue ({display_currency} display units)", total_revenue],
                [f"Estimated contribution ({display_currency} display units)", total_contribution],
            ],
        ),
        "",
        "## Demand Model Comparison",
        "",
        f"Daily product/store observations: model-selection training {train_count:,}; validation {validation_count:,}; final refit {refit_count:,}; untouched test {test_count:,}.",
        f"Training dates end: {train_end}; validation dates: {validation_start} through {validation_end}; final test dates: {test_start} through {test_end}.",
        "All boundaries use unique calendar dates. The selected model and blend weight are chosen on validation dates; final test dates are not used for selection.",
        selection_detail,
        "This is a rolling one-day-ahead evaluation: previous-day observed demand becomes available before each next-day prediction. It is not a fixed-origin multi-month forecast.",
        "The training-mean baseline predicts the mean demand in the final refit training period. Previous-day and seasonal-naive baselines use matching historical observations, with the training mean as fallback when a lag is unavailable.",
        "Revenue and contribution MAE use each held-out record's observed price and unit cost, with negative predicted demand floored at zero. These are financial forecast errors conditional on observed prices, not measured gains from changing prices.",
        "",
        _markdown_table(
            ["Model", "Validation MAE (units)", "Validation RMSE (units)", "Test MAE (units)", "Test RMSE (units)", "Test R2", "Revenue MAE (source currency/record)", "Contribution MAE (source currency/record)"],
            model_results[["Model", "Validation_MAE", "Validation_RMSE", "MAE", "RMSE", "R2", "Revenue_MAE", "Contribution_MAE"]].itertuples(
                index=False, name=None
            ),
        ),
        "",
        f"Method selected by validation RMSE: **{best_model_name}**. Its final-test RMSE is reported above; the final test set is not used to select the model.",
        "",
        "## Price and Promotion Findings",
        "",
        f"Overall estimated price elasticity: **{elasticity:.3f}**.",
        "Elasticity is estimated from observational historical data and should not be interpreted as causal.",
        "",
        _markdown_table(
            ["Category", "Estimated elasticity"],
            category_elasticity[["Category", "Elasticity"]].itertuples(
                index=False, name=None
            ),
        )
        if not category_elasticity.empty
        else "No category had enough variation for an elasticity estimate.",
        "",
        _markdown_table(
            ["Promotion measure", "Promotion", "Non-promotion"],
            promotion["comparison"].itertuples(index=False, name=None),
        ),
        "",
        "Observed promotion differences may reflect seasonality, product mix, or other factors, not promotion alone.",
        "",
        "## Important Model Features",
        "",
        importance_table,
        "",
        "Feature importance indicates model reliance, not causal influence.",
        "",
        "## Constrained Price Scenarios",
        "",
        (
            f"Representative product/store: {product_id} / {store_id} "
            f"(the most frequently observed pair). Constraints: minimum margin "
            f"{min_margin:.1f}%; maximum discount {max_discount:.1f}%; tested price "
            f"change range +/-{max_price_change:.1f}%."
        ),
        "",
        _markdown_table(
            [
                "Objective",
                "Price change (%)",
                "Price (source units)",
                "Predicted demand (units)",
                f"Revenue ({display_currency} display units)",
                f"Estimated contribution ({display_currency} display units)",
                "Margin (%)",
            ],
            scenario_rows,
        )
        if scenario_rows
        else "No tested scenario met the selected margin and discount constraints.",
        "",
        "### Demand Sensitivity",
        "",
        f"Sensitivity uses a +/-{max_price_change:.1f}% price grid and demand adjustments of -20%, base, and +20%.",
        "",
        _markdown_table(
            ["Demand scenario", "Price (source units)", "Demand", "Revenue", "Contribution", "Margin (%)"],
            [
                [
                    row["Scenario"],
                    row["Price"],
                    row["Demand"],
                    row["Revenue"] * usd_to_inr,
                    row["Contribution"] * usd_to_inr,
                    row["Margin_Pct"],
                ]
                for _, row in sensitivity.iterrows()
            ],
        )
        if not sensitivity.empty
        else "No sensitivity scenario met the selected constraints.",
        "",
        "## Assumptions and Limitations",
        "",
        currency_limitation,
        cost_limitation,
        data_limitation,
        promotion_limitation,
        "- Elasticity and promotion comparisons are observational; unmeasured confounding may affect them.",
        "- No counterfactual experiment is included, so this evaluation cannot establish that a recommended price change causes higher revenue or contribution.",
        "- Price scenarios depend on the trained model and cost assumptions and should be treated as decision-support estimates.",
        "",
    ]
    report_text = "\n".join(report)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text