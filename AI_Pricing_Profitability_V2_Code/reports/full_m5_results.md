# Full M5 Walmart Dataset: Findings, Analysis and Interpretation

## Scope and method

This run includes all 30,490 item/store series in the supplied M5 sales file. 30,490 series have at least one observed day with a matched selling price. The analysis covers 2011-01-29 through 2016-04-24. Daily observations without a matched M5 selling price are excluded; zero-sales days with a matched price are retained.

Priced observations used: 46,027,957 (training 23,301,186; validation 11,088,906; untouched test 11,637,865). Chronological date boundaries: validation starts 2014-03-21; test starts 2015-04-08. Validation is used for model selection; test dates are held out.

To process all product/store series on a memory-constrained local machine, the evaluation reads the sales matrix in bounded batches rather than creating a single 58-million-row long table. The M5-specific predictive model is an incremental global SGD regressor, trained batch by batch. It is compared with training-mean, previous-day and 7-day seasonal-naive benchmarks. This model set differs from the interactive sampled benchmark, so its metrics should not be compared as if they came from the same model experiment.

The previous-day and 7-day baselines use actual observed quantity from the corresponding prior calendar day. This is a rolling one-day-ahead evaluation, not a fixed-origin multi-month forecast.

## Full-dataset forecast results

| Model | Validation RMSE (units) | Test MAE (units) | Test RMSE (units) | Test R² | Revenue MAE (USD/record) | Estimated contribution MAE (USD/record) |
|---|---:|---:|---:|---:|---:|---:|
| Training Mean Baseline | 3.7189 | 1.8042 | 3.6217 | -0.0064 | 7.0102 | 2.4536 |
| Previous-Day Naive | 2.6529 | 1.1111 | 2.6432 | 0.4640 | 3.7104 | 1.2987 |
| Seasonal Naive (7-day) | 2.9115 | 1.1551 | 2.7918 | 0.4020 | 3.8112 | 1.3339 |
| Incremental SGDRegressor | 2.9663 | 1.1212 | 2.8903 | 0.3590 | 3.6437 | 1.2753 |
| Tuned Hybrid (Incremental SGDRegressor + previous-day, alpha=0.35) | 2.4623 | 1.0723 | 2.4465 | 0.5408 | 3.5784 | 1.2524 |

Selected by validation RMSE: **Tuned Hybrid (Incremental SGDRegressor + previous-day, alpha=0.35)** (validation RMSE 2.4623 units). On untouched test dates, its MAE was 1.0723 units, RMSE 2.4465 units, and R² 0.5408. These values are the measured output of this full-M5 run, not estimates inferred from the sampled app screenshot.

Revenue MAE and estimated contribution MAE are conditional forecast-error measures calculated from observed prices. They do not measure revenue or profit caused by a price change.

## Management interpretation

This is a management case study using secondary, observed retail data and a decision-support proof of concept. Pricing decisions require balancing demand, revenue and margin rather than maximizing sales volume alone. The benchmark illustrates how forecasts can support comparison of demand scenarios and inform pricing analysis; it is not an automated pricing recommendation or evidence that a tested price will achieve a particular financial outcome.

## Research justification and no primary survey

> I chose the M5 Walmart dataset because my research question is about using observed sales and price history to support pricing decisions. Customer surveys can provide useful information about preferences and willingness to buy, but stated intentions may differ from actual purchases. The M5 data provides a longitudinal retail case with product, store, price, demand, and calendar information, allowing me to compare forecasting approaches and demonstrate how their outputs can be used in pricing scenarios.

No polls, surveys or interviews were conducted. A survey would be appropriate if the main research question concerned customer attitudes or stated willingness to pay. This research instead examines observed sales and pricing patterns, so secondary historical retail data is more directly aligned. The chat was not used to collect user or participant data: no respondents were recruited and no survey questions were administered. A survey could complement this analysis in future research.

## Limitations

- The M5 records are observational. Forecast accuracy does not establish causal price effects or prove that a price change caused demand to change.
- M5 does not provide actual product-level cost for this prototype. Unit cost is assumed to be 65% of selling price; estimated contribution is therefore not Walmart's actual accounting profit.
- M5 does not provide a complete verified promotion label or list price. Promotion and discount interpretations based on reference prices are proxies, not verified Walmart promotions.
- The incremental SGD model is a computationally scalable global benchmark and is methodologically different from the interactive sampled XGBoost/ensemble comparison.
- Results apply to the supplied M5 files and their available observed dates. Missing-price observations are excluded; zero-sales observations with prices are included.

## Final-review conclusion

The full-series evaluation demonstrates a reproducible, chronological forecasting workflow over the supplied M5 item/store data. Its management contribution is to show how observed demand forecasts and simple benchmarks can inform structured pricing discussion. Conclusions should remain decision-support oriented: the analysis is not causal, and estimated contribution is not actual profit.
