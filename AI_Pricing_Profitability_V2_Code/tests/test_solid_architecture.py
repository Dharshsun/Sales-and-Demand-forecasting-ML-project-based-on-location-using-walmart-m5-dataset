import unittest

import pandas as pd
from sklearn.linear_model import LinearRegression

from pricing_analysis import assess_pricing_constraints, model_demand, split_for_backtest
from pricing_insights import build_price_response_curve
from pricing_data import prepare_demand_data


class LinearOnlyProvider:
    def build_estimators(self):
        return {"Linear Regression": LinearRegression()}


class EmptyProvider:
    def build_estimators(self):
        return {}


def sample_data(day_count=50):
    dates = pd.date_range("2024-01-01", periods=day_count, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Product_ID": "P001",
            "Store_ID": "S001",
            "Category": "Grocery",
            "Selling_Price": [20 + day % 3 for day in range(day_count)],
            "Base_Price": 25,
            "Unit_Cost": 12,
            "Quantity_Sold": [10 + day % 5 for day in range(day_count)],
            "Promotion": [day % 4 == 0 for day in range(day_count)],
        }
    )


class SolidArchitectureTests(unittest.TestCase):
    def test_each_constraint_changes_eligibility(self):
        data = sample_data()
        baseline = assess_pricing_constraints(data, 0, 100, 30)
        margin_restricted = assess_pricing_constraints(data, 46, 100, 30)
        discount_restricted = assess_pricing_constraints(data, 0, 10, 30)
        price_change_restricted = assess_pricing_constraints(data, 0, 100, 10)

        self.assertEqual(baseline["eligible_rows"], len(data))
        self.assertLess(margin_restricted["eligible_rows"], baseline["eligible_rows"])
        self.assertLess(discount_restricted["eligible_rows"], baseline["eligible_rows"])
        self.assertLess(price_change_restricted["eligible_rows"], baseline["eligible_rows"])
        category_summary = baseline["category_summary"]
        self.assertEqual(category_summary["Eligible_Records"].sum(), len(data))
        self.assertEqual(category_summary["Excluded_Records"].sum(), 0)

    def test_temporal_splits_do_not_share_dates(self):
        data = prepare_demand_data(sample_data())
        train, validation, train_validation, test = split_for_backtest(data)

        self.assertLess(train.Date.max(), validation.Date.min())
        self.assertLess(validation.Date.max(), test.Date.min())
        self.assertLess(train_validation.Date.max(), test.Date.min())

    def test_custom_estimator_provider_uses_shared_evaluation(self):
        results, trained = model_demand(sample_data(), LinearOnlyProvider())
        selected = results.loc[results["Selected"]]

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected.iloc[0]["Model"],
            "Tuned Hybrid (Linear Regression + previous-day)",
        )
        self.assertIn(selected.iloc[0]["Model"], trained)
        self.assertGreaterEqual(selected.iloc[0]["RMSE"], 0)

    def test_price_scenarios_accept_predictor_contract(self):
        _, trained = model_demand(sample_data(), LinearOnlyProvider())
        curve = build_price_response_curve(
            sample_data(),
            trained["Tuned Hybrid (Linear Regression + previous-day)"],
            "P001",
            "S001",
        )

        self.assertEqual(len(curve), 31)
        self.assertTrue((curve["Predicted_Demand"] >= 0).all())

    def test_provider_must_include_a_demand_model(self):
        with self.assertRaisesRegex(ValueError, "at least one demand model"):
            model_demand(sample_data(), EmptyProvider())


if __name__ == "__main__":
    unittest.main()