Raw M5 data is intentionally not included.

Place official M5 files in data/m5/:
sales_train_validation.csv
sell_prices.csv
calendar.csv

Optional Hugging Face Online Retail benchmark:
Download online-retail.xlsx from https://huggingface.co/datasets/michaelmallari/online-retail
and place it in data/huggingface/online-retail.xlsx.
The app prefers online-retail-daily.csv in this folder for faster loading.
This CC-BY-4.0 UCI mirror is separate UK GBP data, not a source for filling M5 gaps.
The app excludes cancellations/returns and estimates unavailable unit cost and promotions.

For faster M5 training, the app uses a fixed random sample of up to 500 product/store
series before expanding the daily sales records.
Use the app's "Run full M5 benchmark" action to evaluate every item/store series.
The full run uses an incremental global SGD model and naive baselines. To reduce
runtime, it trains on up to 256 evenly spaced priced observations per series,
then scores every eligible priced validation and test observation. Its results
are separate from the sampled interactive model comparison; reruns for unchanged
input files reuse the cached result.

For testing without M5:
python generate_synthetic_data.py
