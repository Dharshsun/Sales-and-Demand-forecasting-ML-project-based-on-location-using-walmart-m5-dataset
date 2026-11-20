# Latest Pricing Analysis Results

Generated: 2026-09-30T21:32:49+05:30
Data source: M5 Walmart dataset
Input file(s): data/m5/sales_train_validation.csv; data/m5/sell_prices.csv; data/m5/calendar.csv (deterministic sample of 100 item-store series)

## Synopsis-Ready Summary

Using 148,404 records from 2011-01-29 to 2016-04-24, this analysis compared 6 demand predictors using a strict date-held-out 80/20 split. Linear Regression achieved the lowest test RMSE (2.49 units). The estimated overall price elasticity was -0.295; the observed promotion-period differences were 3.42% for demand, -10.92% for revenue, and -10.92% for estimated contribution. Model-based price scenarios are estimates for decision support, not causal effects or guaranteed financial outcomes.

## Dataset Profile

| Measure | Result |
| --- | --- |
| Records | 148,404 |
| Date range | 2011-01-29 to 2016-04-24 |
| Products | 98 |
| Stores | 10 |
| Item-store series | 100 |
| Categories | 3 |
| Units sold | 173,201 |
| Revenue (INR display units) | 47,913,443.20 |
| Estimated contribution (INR display units) | 16,769,705.12 |

## Demand Model Comparison

Training records: 110,296; test records: 38,108.
Training dates end: 2015-04-07; test dates: 2015-04-08 through 2016-04-24.
The holdout boundary is selected from unique calendar dates, so no date appears in both training and test sets. The training-mean baseline predicts the mean training demand for every test record.
Revenue and contribution MAE use each held-out record's observed price and unit cost, with negative predicted demand floored at zero. These are financial forecast errors conditional on observed prices, not measured gains from changing prices.

| Model | MAE (units) | RMSE (units) | R2 | Revenue MAE (source currency/record) | Contribution MAE (source currency/record) |
| --- | --- | --- | --- | --- | --- |
| Linear Regression | 1.11 | 2.49 | 0.49 | 3.87 | 1.35 |
| Gradient Boosting | 1.20 | 2.64 | 0.42 | 4.28 | 1.50 |
| XGBoost | 1.16 | 2.69 | 0.40 | 4.04 | 1.41 |
| Random Forest | 1.20 | 2.75 | 0.38 | 4.25 | 1.49 |
| Decision Tree | 1.24 | 2.78 | 0.36 | 4.40 | 1.54 |
| Training Mean Baseline | 1.50 | 3.48 | -0.00 | 5.62 | 1.97 |

Best model by test RMSE: **Linear Regression**.

## Price and Promotion Findings

Overall estimated price elasticity: **-0.295**.
Elasticity is estimated from observational historical data and should not be interpreted as causal.

| Category | Estimated elasticity |
| --- | --- |
| HOBBIES | -0.37 |
| HOUSEHOLD | -0.33 |
| FOODS | -0.16 |

| Promotion measure | Promotion | Non-promotion |
| --- | --- | --- |
| Units | 1.20 | 1.16 |
| Revenue | 3.53 | 3.96 |
| Contribution | 1.24 | 1.39 |
| Avg Selling Price | 4.48 | 4.82 |

Observed promotion differences may reflect seasonality, product mix, or other factors, not promotion alone.

## Important Model Features

| Feature | Importance |
| --- | --- |
| cat__Product_ID_FOODS_3_288 | 11.00 |
| cat__Product_ID_HOBBIES_1_295 | 6.20 |
| cat__Product_ID_HOUSEHOLD_1_339 | 4.12 |
| cat__Product_ID_FOODS_3_180 | 3.63 |
| cat__Product_ID_FOODS_2_361 | 3.60 |
| cat__Product_ID_FOODS_3_307 | 3.51 |
| cat__Product_ID_HOUSEHOLD_2_004 | 3.30 |
| cat__Product_ID_HOUSEHOLD_1_537 | 3.25 |
| cat__Product_ID_HOBBIES_1_100 | 3.15 |
| cat__Product_ID_FOODS_3_118 | 3.05 |

Feature importance indicates model reliance, not causal influence.

## Constrained Price Scenarios

Representative product/store: FOODS_1_049 / TX_2 (the most frequently observed pair). Constraints: minimum margin 15.0%; maximum discount 30.0%; tested price change range +/-15.0%.

| Objective | Price change (%) | Price (source units) | Predicted demand (units) | Revenue (INR display units) | Estimated contribution (INR display units) | Margin (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Highest revenue | 15.00 | 2.58 | 0.54 | 115.23 | 50.10 | 43.48 |
| Highest contribution | 15.00 | 2.58 | 0.54 | 115.23 | 50.10 | 43.48 |

### Demand Sensitivity

Sensitivity uses a +/-20% price grid and demand adjustments of -20%, base, and +20%.

| Demand scenario | Price (source units) | Demand | Revenue | Contribution | Margin (%) |
| --- | --- | --- | --- | --- | --- |
| Conservative: demand 20% below forecast | 2.69 | 0.44 | 97.34 | 44.61 | 45.83 |
| Base: model forecast | 2.69 | 0.55 | 121.67 | 55.77 | 45.83 |
| Optimistic: demand 20% above forecast | 2.69 | 0.65 | 146.01 | 66.92 | 45.83 |

## Assumptions and Limitations

- Financial amounts are multiplied by 83.0 for the app's INR display; this is not a verified exchange-rate conversion.
- M5 does not provide actual product-level cost; the prototype estimates unit cost as 65% of selling price, so contribution is not actual profit.
- Results are based on the sampled M5 item-store series listed above and do not represent the complete M5 dataset.
- M5 does not provide an explicit promotion label or base price in this loader; the analysis flags prices below each item-store median as a promotion proxy, not a verified promotion or discount.
- Elasticity and promotion comparisons are observational; unmeasured confounding may affect them.
- No counterfactual experiment is included, so this evaluation cannot establish that a recommended price change causes higher revenue or contribution.
- Price scenarios depend on the trained model and cost assumptions and should be treated as decision-support estimates.
