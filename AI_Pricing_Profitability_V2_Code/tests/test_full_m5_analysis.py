import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from full_m5_analysis import (
    _BlendSelection,
    _make_features,
    _make_features_at,
    evaluate_full_m5,
    render_full_m5_report,
)


class FullM5AnalysisTests(unittest.TestCase):
    def test_split_features_keep_previous_day_lags_at_boundaries(self):
        quantity = np.arange(1, 11, dtype=np.float32)[None, :]
        price = np.full_like(quantity, 10)

        features, lag_one, lag_seven = _make_features(
            quantity,
            price,
            np.array([10], dtype=np.float32),
            np.arange(10),
            5,
            9,
        )

        self.assertEqual(features.shape, (1, 4, 8))
        np.testing.assert_array_equal(lag_one, [[5, 6, 7, 8]])
        np.testing.assert_array_equal(lag_seven, [[0, 0, 1, 2]])
        sparse_features = _make_features_at(
            quantity,
            price,
            np.array([10], dtype=np.float32),
            np.arange(10),
            np.zeros(4, dtype=np.int32),
            np.arange(5, 9),
        )
        np.testing.assert_allclose(sparse_features, features[0])

    def test_blend_selection_uses_exact_validation_rmse_minimum(self):
        selector = _BlendSelection()
        actual = np.array([1, 5, 2, 7], dtype=float)
        prediction = np.array([2, 4, 2, 8], dtype=float)
        lag = np.array([0, 6, 1, 9], dtype=float)
        selector.add(actual, prediction, lag)

        alpha = selector.best_alpha()
        observed_rmse = np.sqrt(
            np.mean((alpha * prediction + (1 - alpha) * lag - actual) ** 2)
        )

        self.assertAlmostEqual(selector.rmse(alpha), observed_rmse)
        self.assertGreaterEqual(alpha, 0)
        self.assertLessEqual(alpha, 1)

    def test_full_series_evaluation_keeps_zero_sales_and_reports_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sales_path = root / "sales.csv"
            prices_path = root / "prices.csv"
            calendar_path = root / "calendar.csv"

            sales_rows = []
            price_rows = []
            for store in ("STORE_1", "STORE_2"):
                for item_index in (1, 2):
                    item = f"ITEM_{item_index}"
                    quantities = [0, 1, 2, 0, 3, 1, 0, 2, 1, 3]
                    sales_rows.append(
                        {
                            "id": f"{item}_{store}",
                            "item_id": item,
                            "dept_id": "DEPT_1",
                            "cat_id": "CAT_1",
                            "store_id": store,
                            "state_id": "CA",
                            **{
                                f"d_{day + 1}": quantity + item_index
                                for day, quantity in enumerate(quantities)
                            },
                        }
                    )
                    price_rows.append(
                        {
                            "store_id": store,
                            "item_id": item,
                            "wm_yr_wk": 1,
                            "sell_price": 10.0 + item_index,
                        }
                    )
            pd.DataFrame(sales_rows).to_csv(sales_path, index=False)
            pd.DataFrame(price_rows).to_csv(prices_path, index=False)
            pd.DataFrame(
                {
                    "d": [f"d_{day}" for day in range(1, 11)],
                    "date": pd.date_range("2016-01-01", periods=10).astype(str),
                    "wm_yr_wk": [1] * 10,
                }
            ).to_csv(calendar_path, index=False)

            result = evaluate_full_m5(sales_path, prices_path, calendar_path)
            report = render_full_m5_report(result)

        self.assertEqual(result["series_count"], 4)
        self.assertEqual(result["effective_series_count"], 4)
        self.assertEqual(result["train_rows"], 24)
        self.assertEqual(result["model_fit_rows"], 24)
        self.assertEqual(result["training_series_count"], 4)
        self.assertEqual(result["validation_rows"], 8)
        self.assertEqual(result["test_rows"], 8)
        self.assertIn("all 4 item/store series", report)
        self.assertIn("No polls, surveys or interviews were conducted.", report)
        self.assertTrue(np.isfinite(result["selected_test"]["RMSE"]))


if __name__ == "__main__":
    unittest.main()
