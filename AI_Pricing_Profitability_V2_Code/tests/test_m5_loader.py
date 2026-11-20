import tempfile
import unittest
from pathlib import Path

import pandas as pd

from m5_loader import load_m5_for_pricing


class M5LoaderTests(unittest.TestCase):
    def test_limits_series_before_expanding_daily_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sales_path = root / "sales.csv"
            prices_path = root / "prices.csv"
            calendar_path = root / "calendar.csv"

            pd.DataFrame(
                [
                    ["ITEM_1_STORE_1", "ITEM_1", "DEPT_1", "CAT_1", "STORE_1", "CA", 1, 2],
                    ["ITEM_2_STORE_1", "ITEM_2", "DEPT_1", "CAT_1", "STORE_1", "CA", 3, 4],
                    ["ITEM_3_STORE_1", "ITEM_3", "DEPT_1", "CAT_1", "STORE_1", "CA", 5, 6],
                ],
                columns=[
                    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                    "d_1", "d_2",
                ],
            ).to_csv(sales_path, index=False)
            pd.DataFrame(
                [
                    ["STORE_1", item, week, 10]
                    for item in ["ITEM_1", "ITEM_2", "ITEM_3"]
                    for week in [1, 2]
                ],
                columns=["store_id", "item_id", "wm_yr_wk", "sell_price"],
            ).to_csv(prices_path, index=False)
            pd.DataFrame(
                {
                    "d": ["d_1", "d_2"],
                    "date": ["2016-01-01", "2016-01-02"],
                    "wm_yr_wk": [1, 2],
                }
            ).to_csv(calendar_path, index=False)

            result = load_m5_for_pricing(
                sales_path, prices_path, calendar_path, max_series=2
            )

        self.assertEqual(result[["Product_ID", "Store_ID"]].drop_duplicates().shape[0], 2)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
