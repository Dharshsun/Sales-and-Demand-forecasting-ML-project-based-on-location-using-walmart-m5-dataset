# AI Pricing & Profitability Optimization — Version 2

This is the deeper research-oriented prototype discussed in the conversation.

Pipeline:
Business problem → demand analysis → price elasticity → promotion impact → ML demand prediction → explainable AI → continuous price response → scenario simulation → constraints → sensitivity analysis → strategic decision support.

## Run

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
streamlit run app.py
```

## Save the latest results report

Load a dataset in the app, then select **Generate latest results report** in the sidebar. The app writes a synopsis-ready Markdown report to `reports/latest_results.md`; generating another report replaces that file, so only the latest report is retained. The report includes the selected input files, data profile, model comparison, elasticity and promotion summaries, model feature importance, constrained price scenarios, sensitivity results, and assumptions.

## M5 Walmart data

Raw M5 data is intentionally not included. Download the official M5 files and put these in `data/m5/`:

- sales_train_validation.csv
- sell_prices.csv
- calendar.csv

M5 contains historical sales/prices but not actual product-level cost. Do not claim actual Walmart profit. Revenue is observable; contribution requires an explicit cost assumption.

## Hugging Face Online Retail data

The app also supports a separate **Hugging Face Online Retail dataset** source alongside the synthetic pilot and M5 sources. The workbook is from [michaelmallari/online-retail](https://huggingface.co/datasets/michaelmallari/online-retail), a CC-BY-4.0 mirror of the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). The downloaded workbook and a normalized daily CSV are not committed; place them in `data/huggingface/`. The app loads the CSV for faster startup and falls back to the workbook if the CSV is absent. See `data/README.txt` for download details.

The loader excludes canceled invoices, returns, and non-positive prices, then aggregates transactions to daily product/country observations (country is used as a market proxy, not as a physical store). Its source currency is GBP and is displayed as GBP without conversion. The source does not include actual product costs, explicit promotion or holiday records, or a dependable category field; estimated cost and promotion/discount features are proxies. This is a separate UK retailer benchmark, not Walmart data, and must not be used to fill missing M5 observations.

The app infers an M5 reference price from the sampled item-store median selling price. Its discount and promotion indicators are analytical proxies, not verified Walmart list prices or promotion records. At the bottom of every analysis page, the constraint outcome and category charts respond to the selected minimum margin, maximum discount, and price-movement settings. Historical descriptive/model plots remain based on the complete dataset; the shared charts show which historical rows meet the current constraints. “Efficient” there means only that qualifying rows have positive estimated contribution; it is not a formal efficiency measure or actual profit.

## Main analyses

1. Descriptive business analysis
2. Price elasticity by category/store
3. Promotion lift
4. ML model comparison
5. Explainable AI
6. Continuous price-response curves
7. Revenue and contribution analysis
8. Business constraints
9. Conservative/base/optimistic sensitivity analysis
10. Cross-industry transfer

## Code architecture

- `pricing_data.py` owns input normalization, daily demand features, and date-based splits.
- `demand_forecasting.py` owns estimator construction, validation-based selection, and forecast evaluation. It accepts a `DemandEstimatorProvider`, so additional estimators can be supplied without changing the evaluation workflow.
- `model_contracts.py` defines the small `DemandPredictor` and `DemandEstimatorProvider` interfaces used at those boundaries.
- `pricing_insights.py` owns elasticity, promotion comparisons, pricing scenarios, feature explanations, and business segmentation. Scenario functions consume the predictor interface rather than depending on one concrete estimator.
- `pricing_analysis.py` is a compatibility facade, keeping the existing app and report imports stable while implementation responsibilities stay separated.
- `m5_loader.py`, `online_retail_loader.py`, `results_report.py`, and `app.py` handle M5/Online Retail ingestion, report writing, and user interface/orchestration respectively.

Run the architecture tests with `python -m unittest discover -s tests -v` from this project directory.

## Forecast benchmark and interpretation

The evaluator aggregates source rows to daily product/store observations so demand targets and lag baselines share the same grain. It uses unique calendar dates for chronological training, validation and final testing. Models and the blend weight are chosen on validation dates; the final test period is reserved for evaluation. The benchmark includes a training-mean baseline, previous-day naive, seasonal-naive (7-day), linear regression, decision tree, random forest, gradient boosting and XGBoost when available.

The selected hybrid blends the validation-selected regression model with previous-day observed demand. Evaluation is rolling one-day-ahead: each forecast may use the previous day's observed demand, but it is not a fixed-origin forecast over a long horizon. Reported metrics include demand MAE/RMSE/R2 and revenue/estimated-contribution MAE conditional on observed test prices and assumed costs.

These tests can show improvement over the included baselines on these data and dates, but do not establish superiority over every published system or prove causal profit uplift from changing prices. A defensible published-system comparison requires reproducing the paper's method on the same data, horizon, features and metric. M5 results use a deterministic 100-series sample; M5 costs remain assumed, and synthetic data remains generated from assumptions.

The framework can be adapted to retail, e-commerce, hotels, airlines, restaurants, SaaS and telecom.

Elasticity is observational in this prototype and should not automatically be interpreted as causal.
