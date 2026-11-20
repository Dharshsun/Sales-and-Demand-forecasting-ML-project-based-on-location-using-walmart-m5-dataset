# A Study of AI-Driven Pricing and Profitability Optimization

## Research Methodology

### 1. Research Methodology – Overview

This study follows a quantitative, data-driven research methodology to examine how pricing decisions influence demand and how machine learning can support revenue- and profitability-oriented pricing decisions. The research combines business analysis with predictive analytics. Rather than treating price as an isolated variable, the study considers price together with product, store, and time-related factors, and converts predictive results into practical pricing scenarios.

The methodology follows a decision-support framework: historical sales and price data are processed to estimate demand; machine learning models are compared using a chronological holdout; price sensitivity and promotion-related patterns are examined; and demand predictions are translated into revenue, estimated contribution, and margin outcomes under business constraints.

### 2. Research Design

| Methodological Element | Approach Used |
| --- | --- |
| Research approach | Quantitative and analytical research |
| Research orientation | Business decision support using AI/ML |
| Research design | Descriptive + predictive + scenario-based analytical design |
| Unit of analysis | Item-store-date observation |
| Primary outcome | Demand prediction |
| Business outcomes | Revenue, estimated contribution and margin |
| Validation | Strict chronological train-test split |
| Decision layer | Constrained price scenario simulation and sensitivity analysis |
| Data source | M5 Walmart retail dataset |
| Prototype sample | Deterministic sample of 100 item-store series |

### 3. Research Objectives

- To examine the relationship between selling price and product demand using historical retail data.
- To estimate price elasticity overall and across product categories.
- To compare statistical and machine learning demand-prediction models using a date-held-out test set.
- To identify the variables and product characteristics the predictive model relies on.
- To examine observed differences between promotion-proxy and non-promotion periods.
- To translate demand predictions into revenue, estimated contribution and margin outcomes under alternative prices.
- To evaluate pricing sensitivity under conservative, base and optimistic demand assumptions.
- To develop a business-oriented AI decision-support framework transferable to other demand-sensitive industries.

### 4. Data Source and Sampling Method

The empirical analysis uses the M5 Walmart retail dataset, containing historical product sales, store context, selling prices and calendar information. The current prototype uses a deterministic sample of 100 item-store series rather than the complete M5 universe so that the full analytical workflow remains computationally practical and reproducible.

| Measure | Current Study |
| --- | --- |
| Records analyzed | 148,404 |
| Date range | 29 January 2011 – 24 April 2016 |
| Products | 98 |
| Stores | 10 |
| Item-store series | 100 |
| Categories | FOODS, HOBBIES, HOUSEHOLD |
| Units sold | 173,201 |
| Input files | sales_train_validation.csv; sell_prices.csv; calendar.csv |
| Sampling | Deterministic item-store sample |

### 5. Variables Used in the Study

| Variable Group | Examples / Role |
| --- | --- |
| Dependent variable | Quantity sold / demand |
| Core pricing variable | Selling price |
| Product variables | Product identifier and category |
| Store variables | Store identifier |
| Time variables | Date and calendar-related information |
| Promotion proxy | Price below item-store historical median |
| Financial outputs | Revenue, estimated cost, contribution and margin |
| Scenario variables | Price change and demand adjustment |

The dependent variable is quantity sold. Selling price is the principal explanatory variable, while product, store, and calendar characteristics capture systematic differences in demand. Financial measures are derived from observed or explicitly assumed inputs and are not presented as directly observed accounting profit.

### 6. Data Pre-processing and Preparation

- The M5 sales, price and calendar files are integrated using product, store and calendar keys.
- The selected item-store series are transformed into a time-indexed analytical dataset.
- Price and sales variables are aligned to the relevant observation date.
- Records are checked for missing, invalid and inconsistent values before modelling.
- Derived variables are created for revenue, estimated cost, contribution and contribution margin.
- Because M5 does not provide actual product-level cost, the prototype estimates unit cost as 65% of selling price.
- Because the selected M5 inputs do not provide an explicit promotion label, price below the item-store historical median is used as a promotion proxy.

These assumptions are explicitly disclosed so that estimated financial measures and proxy variables are not confused with actual Walmart accounting or promotion data.

### 7. Exploratory and Descriptive Analysis

Before modelling, the study describes demand, price and financial patterns. The analysis includes summary statistics, time-based trends, category performance, price-versus-quantity relationships and checks for unusual observations.

- Demand, price and financial distributions.
- Category-level units, revenue and average-price comparison.
- Time-series trends in sales and revenue.
- Price-versus-quantity relationship analysis.
- Identification of unusually high or low observations.

### 8. Machine Learning Modelling Approach

The study compares models with different levels of statistical and machine learning complexity. The objective is not to assume that the most complex model is automatically the best, but to determine which approach provides useful out-of-sample predictive performance for a business decision.

| Model | Purpose |
| --- | --- |
| Training Mean Baseline | Benchmark for a simple non-ML demand estimate |
| Linear Regression | Interpretable baseline and directional relationship analysis |
| Decision Tree | Captures non-linear decision rules |
| Random Forest | Ensemble modelling of non-linear patterns |
| Gradient Boosting | Sequential boosting for improved prediction |
| XGBoost | Advanced gradient boosting for non-linear demand patterns |

### 9. Training and Validation Strategy

Because retail demand and prices are time dependent, a random train-test split can allow future information to influence training. The study therefore uses a chronological date-held-out validation strategy.

| Stage | Period / Records |
| --- | --- |
| Training | Through 7 April 2015; 110,296 records |
| Testing | 8 April 2015 – 24 April 2016; 38,108 records |
| Split | Approximately 80/20 using unique calendar dates |
| Leakage control | No calendar date appears in both sets |

The held-out period represents a future period relative to model training, making the evaluation closer to a real business forecasting setting.

### 10. Model Evaluation Metrics

| Metric | Purpose |
| --- | --- |
| MAE | Average absolute demand prediction error in units |
| RMSE | Penalizes larger prediction errors; primary model-selection measure |
| R² | Measures variation explained relative to the baseline |
| Revenue MAE | Financial forecast error using observed held-out price |
| Contribution MAE | Financial forecast error using the assumed cost model |

Revenue and contribution MAE are conditional financial forecast measures. They are not interpreted as evidence that the model itself creates additional revenue or profit.

### 11. Price Elasticity Analysis

Price elasticity is used to quantify the historical responsiveness of demand to price. The study estimates elasticity overall and by category. The current overall estimate is -0.295; category estimates are -0.37 for HOBBIES, -0.33 for HOUSEHOLD and -0.16 for FOODS.

The negative estimates indicate an inverse historical association between price and quantity demanded. Since the data are observational, the estimates are treated as associational rather than causal. Price changes may be related to seasonality, product lifecycle, inventory, competition or other managerial decisions.

### 12. Promotion / Price-Reduction Analysis

The study compares observations classified as promotion-proxy periods with non-promotion periods. The proxy identifies prices below the historical median for the relevant item-store series.

| Measure | Promotion Proxy | Non-Promotion |
| --- | --- | --- |
| Units | 1.20 | 1.16 |
| Revenue | 3.53 | 3.96 |
| Contribution | 1.24 | 1.39 |
| Average Selling Price | 4.48 | 4.82 |

These differences are not interpreted as causal promotion effects. Product mix, seasonality, timing and other confounding factors may contribute to the observed differences.

### 13. Explainable AI / Model Interpretation

Model interpretation is included to connect prediction with managerial decision-making. Feature importance is examined to identify which inputs the model relies on most strongly. Several product identifiers have relatively high importance in the current results.

Feature importance represents model reliance, not causal influence. A high importance score does not mean that changing that feature would necessarily cause demand to change by the same amount.

### 14. Price Scenario Simulation and Decision Framework

The central business component is converting predicted demand into alternative pricing scenarios. For a representative item-store combination, the model estimates demand at a range of prices while other scenario variables are held constant. Revenue and estimated contribution are then calculated for each price.

| Constraint / Setting | Current Prototype |
| --- | --- |
| Minimum contribution margin | 15% |
| Maximum discount | 30% |
| Maximum price change | ±15% |
| Representative item-store | FOODS_1_049 / TX_2 |
| Decision objectives | Highest revenue and highest estimated contribution |

This creates a constrained optimization perspective. Instead of asking only which price produces the most demand, the framework asks which price produces the most attractive financial outcome while satisfying explicit business constraints.

### 15. Sensitivity and Risk Analysis

Because demand predictions contain uncertainty, the study evaluates conservative, base and optimistic demand scenarios. Demand is adjusted by -20%, 0% and +20% relative to the model forecast, and revenue, contribution and margin are recalculated.

| Scenario | Demand Adjustment | Purpose |
| --- | --- | --- |
| Conservative | -20% | Tests downside exposure |
| Base | Model forecast | Central estimate |
| Optimistic | +20% | Tests potential upside |

Sensitivity analysis makes the framework more useful for management because decisions can be examined as a range of possible outcomes rather than as a single deterministic recommendation.

### 16. Overall Research Workflow

1. M5 data acquisition
2. Data integration and cleaning
3. Exploratory and descriptive analysis
4. Feature engineering
5. Chronological train-test split
6. Baseline and ML model development
7. Model evaluation and comparison
8. Price elasticity estimation
9. Promotion-proxy analysis
10. Explainable AI / feature importance
11. Price response curve generation
12. Constrained revenue and contribution scenarios
13. Sensitivity and risk analysis
14. Business interpretation and cross-industry implications

### 17. Current Empirical Results

The latest prototype provides an initial empirical demonstration that the proposed methodology can be implemented using real historical retail data.

| Indicator | Result |
| --- | --- |
| Records | 148,404 |
| Overall price elasticity | -0.295 |
| Best model by test RMSE | Linear Regression |
| Test RMSE | 2.49 units |
| Test MAE | 1.11 units |
| Test R² | 0.49 |
| Revenue MAE | 3.87 per record |
| Contribution MAE | 1.35 per record |
| Training records | 110,296 |
| Test records | 38,108 |

These results demonstrate a complete path from historical observations to demand prediction, price sensitivity analysis and constrained financial scenarios. They are prototype research results and not a live Walmart pricing recommendation.

### 18. Business and MBA Data Science & AI Relevance

The project is designed as business research supported by AI rather than as a purely technical machine learning exercise. The central business question is how organizations can make pricing decisions that balance demand, revenue and profitability under uncertainty.

- Demand forecasts can support decisions about expected sales at different prices.
- Elasticity analysis can identify products or categories with different historical price sensitivity.
- Scenario simulation allows managers to compare financial outcomes before changing prices.
- Margin constraints prevent the decision from focusing only on sales volume or revenue.
- Sensitivity analysis highlights downside risk and uncertainty.
- Explainable AI provides transparency into the model's reliance on different inputs.
- The framework can be adapted conceptually to e-commerce, hotels, airlines, restaurants, telecom, and subscription businesses.

### 19. Reliability, Validity and Reproducibility

- The chronological holdout reduces future-to-past information leakage.
- A training-mean baseline provides a transparent benchmark.
- Six demand predictors are compared rather than relying on one algorithm.
- The sample, date boundary and input files are documented.
- Financial assumptions are explicitly disclosed.
- Sensitivity analysis is used to examine uncertainty.
- The Python workflow can be rerun using the same source files and deterministic sample.

### 20. Limitations

- The current analysis uses 100 item-store series rather than the complete M5 dataset.
- M5 does not provide actual product-level cost, so contribution is based on an assumed cost ratio and is not actual accounting profit.
- M5 does not provide a verified promotion label or explicit base price in the selected inputs; promotion is therefore proxied by relative price.
- Elasticity estimates are observational and may be affected by confounding.
- No randomized pricing experiment or formal causal inference method is included.
- Price-response curves depend on the predictive model and should not be interpreted as guaranteed causal responses.
- The INR display conversion factor of 83.0 is a presentation assumption, not a verified historical exchange-rate conversion.
- The scenario analysis is representative rather than a complete optimization over the entire product-store universe.

### 21. Planned Enhancements for the Final Study

- Expand the analysis to a larger M5 sample or the complete feasible item-store universe.
- Add lagged demand, rolling averages and richer seasonality features.
- Compare advanced boosting and time-series approaches where computationally feasible.
- Use stronger causal or quasi-experimental methods if suitable variables become available.
- Validate unit-cost assumptions using external or business-provided data where possible.
- Develop broader price-response optimization with explicit revenue and margin constraints.
- Add stress testing for demand uncertainty, cost changes and competitor price movements.
- Convert the research outputs into a managerial decision-support dashboard.

### 22. Methodology Conclusion

The proposed methodology integrates business research and machine learning into a pricing decision framework. Historical retail data are first used to understand demand and price behavior; predictive models are evaluated on future-period data; elasticity and promotion-related patterns provide business interpretation; and model outputs are then converted into constrained revenue and contribution scenarios. This allows the study to address not only whether demand can be predicted, but how predictive analytics can support practical pricing and profitability decisions.

The current M5-based results provide an operational proof of concept. The next stage should strengthen causal interpretation, improve the price-response and optimization components, validate assumptions and translate the findings into a robust managerial decision-support framework.

---

## 2nd Review: Why the Research Is Worthwhile

### Introduction

This study is worth pursuing because it addresses a problem that is highly relevant, economically significant, and difficult to solve using intuition alone: how to set prices in a way that improves both demand and profitability. The project is designed around a clear research pipeline: business problem → demand analysis → price elasticity → promotion impact → ML prediction → explainable AI → price response → scenario simulation → constraints → sensitivity analysis → strategic decision support. This structure demonstrates that the research is not only technical, but also decision-oriented and practically useful.

The central argument is that pricing is not a simple matter of increasing revenue; a firm must balance demand, margin, and risk simultaneously. The codebase reflects this by combining historical sales analysis, demand forecasting, price elasticity estimation, and contribution-oriented scenario testing. This makes the research relevant to retailers, e-commerce platforms, hospitality, airlines, and other industries where pricing directly influences both revenue and profitability.

### Why the Problem Matters

Pricing decisions shape customer demand, market competitiveness, and business sustainability. In most organizations, pricing is still guided by experience, manual rules, or limited historical analysis. However, these approaches often fail to capture nonlinear customer behavior, promotional effects, and category-specific differences in price sensitivity.

The project addresses this gap by analyzing demand and pricing signals from real-world retail-style data. The system uses historical sales records, price observations, category structure, and promotion indicators to estimate how changes in price affect demand and contribution. This is a meaningful problem because inaccurate pricing can lead to lost sales, lower margins, or poor strategic positioning.

The project’s own README emphasizes that the framework is built to support “business problem → demand analysis → price elasticity → promotion impact → ML demand prediction → explainable AI → continuous price response → scenario simulation → constraints → sensitivity analysis → strategic decision support.” This pipeline makes the research valuable because it transforms raw transaction data into actionable business insight rather than merely reporting historical trends.

### Research Significance and Contribution

The research is significant because it moves beyond descriptive reporting and incorporates a decision-support framework for pricing strategy. It does not only ask, “What happened in the past?” It asks, “What should be done under different pricing conditions?”

The codebase implements several analytical layers that collectively justify the study’s worthiness:

1. Demand modeling and forecasting: The study compares multiple model families, including Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost, using a time-based holdout validation process.
2. Price elasticity estimation: The project quantifies how demand changes with price and shows category-level elasticity differences.
3. Promotion analysis: It evaluates the effect of promotional periods on demand, revenue, and contribution, highlighting that higher volume does not necessarily mean higher profitability.
4. Scenario simulation: The app evaluates constrained price scenarios under minimum margin, maximum discount, and maximum price-change assumptions.
5. Sensitivity analysis: It tests conservative, base, and optimistic pricing scenarios to assess risk and decision robustness.
6. Explainability: The framework identifies influential features and helps show which variables most affect the model’s demand prediction.

These features show that the research is not a superficial analytical exercise. It is a practical framework for evidence-based pricing decisions in real business settings.

### Evidence from the Codebase

The codebase provides concrete evidence that the research is methodologically grounded and useful.

The README states that the study uses a benchmark that compares several models and a hybrid validation method with a strict temporal split. According to the generated report, the analysis uses 148,404 records spanning 2011-01-29 to 2016-04-24 from the M5 Walmart dataset. The report also notes that the dataset contains 98 products, 10 stores, 100 item-store series, and 3 categories.

The results show the model comparison clearly. The report states that Linear Regression achieved the lowest test RMSE of 2.49 units, while the training-mean baseline achieved 3.48. This matters because it shows the analytical approach improves forecasting accuracy beyond a simple average benchmark. The same report also indicates that the overall estimated price elasticity was -0.295, with category-level elasticities ranging from -0.16 in FOODS to -0.37 in HOBBIES and -0.33 in HOUSEHOLD. This is valuable because it reveals that not all product groups respond equally to price changes, which supports the case for segment-specific pricing strategies.

The project also quantifies promotion effects. The observed promotion-period differences show demand rising by 3.42%, but revenue and estimated contribution falling by 10.92%. This is a strong example of why the study is worth doing: it demonstrates that increased volume under promotion does not automatically translate to stronger financial outcomes. This is a realistic managerial concern that a pricing decision-support model can help address.

The constrained price-scenario results further strengthen the justification. Under business constraints such as minimum margin of 15%, maximum discount of 30%, and maximum price change of 15%, the model identifies scenarios that maximize revenue and contribution. The report shows a tested scenario where the model suggests a 15% price increase, yielding a revenue of 115.23 and estimated contribution of 50.10 with a margin of 43.48%. This demonstrates that the framework can produce actionable decisions under realistic constraints rather than abstract academic outputs.

### Why This Research Is Better Than Simple Reporting

Many businesses can report sales volume or revenue trends, but this project goes further by combining multiple layers of analysis into one research framework. It incorporates a direct link between pricing and profitability, which is often missing in ordinary KPI dashboards.

The project also explicitly acknowledges limitations, which makes it academically and professionally credible. The code and generated report state that elasticity is observational and should not be treated as causal, that M5 data lacks actual product-level cost, and that contribution is estimated using assumed cost assumptions. This honesty about uncertainty is an important sign of methodological maturity. A credible research study should not overclaim causality where the data does not support it.

This is especially important for a second review document because the research must demonstrate seriousness, realism, and practical relevance. The project does this by showing not only promising results, but also the boundaries of what the evidence can support.

### Conclusion

This research is worthy because it addresses a decision problem of strong economic importance, uses a structured and multi-layered analytical pipeline, and produces measurable evidence that pricing strategy can be informed by data rather than guesswork. The evidence in the codebase shows that the model can improve forecasting beyond naive baselines, identify category-specific elasticity patterns, assess promotion effects, and test profit-aware pricing scenarios under operational constraints.

In summary, the study is valuable because it bridges the gap between academic demand analysis and managerial pricing decision-making. It contributes a practical research framework for understanding how price, demand, promotion, and contribution interact. This makes the project academically relevant, managerially useful, and appropriately grounded in a realistic understanding of both data limitations and decision complexity.
