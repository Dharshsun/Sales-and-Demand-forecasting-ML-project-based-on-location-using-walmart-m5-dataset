import unittest

import pandas as pd

from online_retail_loader import normalize_online_retail


class OnlineRetailLoaderTests(unittest.TestCase):
    def test_normalizes_sales_and_excludes_returns_and_cancellations(self):
        source = pd.DataFrame(
            {
                "InvoiceNo": ["10001", "10002", "10003", "C10004"],
                "StockCode": ["A1", "A1", "A1", "A1"],
                "Quantity": [2, 1, -1, 1],
                "InvoiceDate": [
                    "2010-12-01 08:30:00",
                    "2010-12-01 12:00:00",
                    "2010-12-02 09:00:00",
                    "2010-12-02 10:00:00",
                ],
                "UnitPrice": [4.0, 6.0, 4.0, 5.0],
                "Country": ["United Kingdom"] * 4,
            }
        )

        result = normalize_online_retail(source)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["Product_ID"], "A1")
        self.assertEqual(row["Store_ID"], "United Kingdom")
        self.assertEqual(row["Category"], "Online Retail")
        self.assertEqual(row["Quantity_Sold"], 3)
        self.assertEqual(row["Revenue"], 14)
        self.assertAlmostEqual(row["Selling_Price"], 14 / 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["Date"]))

    def test_reports_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            normalize_online_retail(pd.DataFrame({"InvoiceNo": ["1"]}))

    def test_rejects_workbook_without_valid_sales(self):
        source = pd.DataFrame(
            {
                "InvoiceNo": ["C10001"],
                "StockCode": ["A1"],
                "Quantity": [1],
                "InvoiceDate": ["2010-12-01"],
                "UnitPrice": [4.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "no valid positive-quantity sales"):
            normalize_online_retail(source)


if __name__ == "__main__":
    unittest.main()
