import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from pricing_analysis import prepare_pricing_data, model_demand, calculate_price_elasticity, calculate_promotion_lift, build_price_response_curve, run_sensitivity_analysis, explain_model, business_segments, generate_business_insights, assess_pricing_constraints
from results_report import write_latest_results_report

st.set_page_config(page_title='AI Pricing & Profitability', page_icon='📊', layout='wide')
st.title('AI-Driven Pricing & Profitability Decision Support')
st.caption('Demand prediction → elasticity → scenario simulation → profitability → strategic pricing')

DISPLAY_UNITS = {
    'Date': 'Date (calendar date)',
    'Product_ID': 'Product ID (identifier)',
    'Store_ID': 'Store ID (identifier)',
    'Category': 'Category (identifier)',
    'Month': 'Month (1-12)',
    'Day_of_Week': 'Day of Week (0=Mon, 6=Sun)',
    'Weekend': 'Weekend (0/1 flag)',
    'Promotion': 'Promotion (0/1 flag)',
    'Holiday_Event': 'Holiday Event (0/1 flag)',
    'Base_Price': 'Base Price (INR/item)',
    'Selling_Price': 'Selling Price (INR/item)',
    'Average_Price': 'Average Price (INR/item)',
    'Avg_Price': 'Average Price (INR/item)',
    'Price': 'Price (INR/item)',
    'Unit_Cost': 'Unit Cost (INR/item)',
    'Revenue': 'Revenue (INR)',
    'Adjusted_Revenue': 'Adjusted Revenue (INR)',
    'Estimated_Cost': 'Estimated Cost (INR)',
    'Contribution': 'Contribution (INR)',
    'Estimated_Contribution': 'Estimated Contribution (INR)',
    'Adjusted_Contribution': 'Adjusted Contribution (INR)',
    'Quantity_Sold': 'Quantity Sold (units)',
    'Units': 'Units (items)',
    'Demand': 'Demand (units)',
    'Predicted_Demand': 'Predicted Demand (units)',
    'Adjusted_Demand': 'Adjusted Demand (units)',
    'Avg_Demand': 'Average Demand (units)',
    'Price_Change_Pct': 'Price Change (%)',
    'Discount_Pct': 'Discount (%)',
    'Margin_Pct': 'Margin (%)',
    'Promotion_Rate': 'Promotion Rate (%)',
    'Elasticity': 'Elasticity (%, quantity / % price)',
    'demand_lift_pct': 'Demand Lift (%)',
    'revenue_lift_pct': 'Revenue Lift (%)',
    'contribution_lift_pct': 'Contribution Lift (%)',
    'MAE': 'MAE (units)',
    'RMSE': 'RMSE (units)',
    'Validation_MAE': 'Validation MAE (units)',
    'Validation_RMSE': 'Validation RMSE (units)',
    'R2': 'R² (unitless)',
    'Importance': 'Importance (relative)',
    'Revenue_MAE': 'Revenue MAE (source currency/daily item-store)',
    'Contribution_MAE': 'Contribution MAE (source currency/daily item-store)',
}
DISPLAY_METRICS = {
    'Units': 'Units (items)',
    'Revenue': 'Revenue (INR)',
    'Contribution': 'Contribution (INR)',
    'Avg Selling Price': 'Average Selling Price (INR/item)',
}
MONEY_COLUMNS = {
    'Base_Price', 'Selling_Price', 'Average_Price', 'Avg_Price', 'Price',
    'Unit_Cost', 'Revenue', 'Adjusted_Revenue', 'Estimated_Cost',
    'Contribution', 'Estimated_Contribution', 'Adjusted_Contribution',
}
USD_TO_INR = 83.0
DISPLAY_FACTOR = USD_TO_INR
DISPLAY_CURRENCY = 'INR'
DISPLAY_SYMBOL = '₹'

def display_units(data):
    """Convert displayed money and add source-currency labels."""
    if isinstance(data, pd.Series):
        displayed = data.copy()
        if displayed.name in MONEY_COLUMNS:
            displayed = displayed * DISPLAY_FACTOR
        displayed.name = DISPLAY_UNITS.get(displayed.name, displayed.name).replace(
            'INR', DISPLAY_CURRENCY
        )
        if displayed.index.name in DISPLAY_UNITS:
            displayed.index.name = DISPLAY_UNITS[displayed.index.name].replace(
                'INR', DISPLAY_CURRENCY
            )
        return displayed
    displayed = data.copy()
    for column in MONEY_COLUMNS.intersection(displayed.columns):
        displayed[column] = displayed[column] * DISPLAY_FACTOR
    if 'Metric' in displayed.columns:
        metric_money_rows = displayed['Metric'].isin({'Revenue', 'Contribution', 'Avg Selling Price'})
        value_columns = displayed.columns.difference(['Metric'])
        displayed.loc[metric_money_rows, value_columns] = displayed.loc[metric_money_rows, value_columns] * DISPLAY_FACTOR
    displayed = displayed.rename(
        columns={
            column: label.replace('INR', DISPLAY_CURRENCY)
            for column, label in DISPLAY_UNITS.items()
        }
    )
    if displayed.index.name in DISPLAY_UNITS:
        displayed.index.name = DISPLAY_UNITS[displayed.index.name].replace(
            'INR', DISPLAY_CURRENCY
        )
    if 'Metric' in displayed.columns:
        displayed['Metric'] = displayed['Metric'].replace(
            {
                metric: label.replace('INR', DISPLAY_CURRENCY)
                for metric, label in DISPLAY_METRICS.items()
            }
        )
    return displayed

def display_summary(data):
    summary = data.select_dtypes('number').describe().T.reset_index(names='Metric')
    summary['Metric'] = summary['Metric'].replace(
        {
            column: label.replace('INR', DISPLAY_CURRENCY)
            for column, label in DISPLAY_UNITS.items()
        }
    )
    return summary

def render_constraint_outcome(data, min_margin, max_discount, max_price_change):
    outcome = assess_pricing_constraints(
        data,
        min_margin=min_margin,
        max_discount=max_discount,
        max_price_change=max_price_change,
    )
    st.divider()
    st.subheader('Constraint Outcome')
    if outcome['status'].startswith('Efficient'):
        st.success(outcome['status'])
    else:
        st.warning(outcome['status'])
    eligible, contribution, average = st.columns(3)
    eligible.metric(
        'Records meeting constraints',
        f"{outcome['eligible_rows']:,} / {outcome['total_rows']:,}",
        f"{outcome['eligible_share_pct']:.1f}% of records",
    )
    contribution.metric(
        'Estimated contribution (not actual profit)',
        f"{DISPLAY_SYMBOL}{outcome['estimated_contribution'] * DISPLAY_FACTOR:,.2f}",
    )
    average.metric(
        'Average per eligible record',
        f"{DISPLAY_SYMBOL}{outcome['average_contribution'] * DISPLAY_FACTOR:,.2f}",
    )
    st.caption(
        f"Constraints: margin >= {min_margin}% | discount <= {max_discount}% | "
        f"price movement <= +/-{max_price_change}%. Efficiency here means only "
        "constraint-feasible historical rows with positive estimated contribution. "
        "This is not a formal efficiency measure or actual profit. Source cost, "
        "discount, and promotion fields may be analytical proxies."
    )
    eligible_chart, contribution_chart = st.columns(2)
    category_summary = outcome['category_summary'].set_index('Category')
    eligible_chart.caption('Constraint-eligible and excluded historical records by category')
    eligible_chart.bar_chart(category_summary[['Eligible_Records', 'Excluded_Records']])
    contribution_chart.caption(f'Estimated contribution of eligible records by category ({DISPLAY_CURRENCY})')
    contribution_chart.bar_chart(
        category_summary[['Eligible_Contribution']] * DISPLAY_FACTOR
    )

APP_DIR = Path(__file__).resolve().parent
REPORT_PATH = APP_DIR / 'reports' / 'latest_results.md'
FULL_M5_REPORT_PATH = APP_DIR / 'reports' / 'full_m5_results.md'
DATA_DIR = APP_DIR / 'data' / 'm5'
ONLINE_RETAIL_PATH = APP_DIR / 'data' / 'huggingface' / 'online-retail.xlsx'
ONLINE_RETAIL_CSV_PATH = APP_DIR / 'data' / 'huggingface' / 'online-retail-daily.csv'
SYNTHETIC_CANDIDATES = [
    DATA_DIR / 'retail_pricing_profitability_research_data.csv',
    DATA_DIR / 'retail_pricing_profitability_demo.csv',
]
M5_FILES = [DATA_DIR / 'sales_train_validation.csv', DATA_DIR / 'sell_prices.csv', DATA_DIR / 'calendar.csv']
M5_MAX_SERIES = 500
FULL_M5_ANALYSIS_VERSION = 'streaming-sgd-v4'

def data_file_signature():
    return tuple((str(path), path.stat().st_mtime_ns if path.exists() else None) for path in [*SYNTHETIC_CANDIDATES, *M5_FILES, ONLINE_RETAIL_PATH, ONLINE_RETAIL_CSV_PATH])

@st.cache_data(show_spinner='Loading dataset...')
def load_csv(path): return pd.read_csv(path)

@st.cache_data(show_spinner='Loading Hugging Face Online Retail dataset...')
def load_online_retail(path):
    from online_retail_loader import load_online_retail as read_online_retail
    return read_online_retail(path)

@st.cache_data(show_spinner='Preparing pricing data...')
def prepare_cached_data(_data, cache_key): return prepare_pricing_data(_data)

@st.cache_data(show_spinner='Loading selected dataset...')
def load_data(source, file_signature):
    if source == 'Synthetic pilot dataset':
        synthetic = next((path for path in SYNTHETIC_CANDIDATES if path.exists()), None)
        if synthetic is None:
            st.warning('Run generate_synthetic_data.py from the application folder first.')
            return None, source
        return load_csv(synthetic), source
    if source == 'Hugging Face Online Retail dataset':
        online_retail_path = (
            ONLINE_RETAIL_CSV_PATH
            if ONLINE_RETAIL_CSV_PATH.exists()
            else ONLINE_RETAIL_PATH
        )
        if not online_retail_path.exists():
            st.warning('Place the normalized Online Retail CSV or Hugging Face workbook in data/huggingface/.')
            return None, source
        return load_online_retail(str(online_retail_path)), source
    paths = M5_FILES
    if not all(p.exists() for p in paths):
        st.warning('Put the official M5 CSV files inside data/m5/.')
        return None, source
    from m5_loader import load_m5_for_pricing
    return load_m5_for_pricing(*map(str, paths), max_series=M5_MAX_SERIES), source

@st.cache_resource(show_spinner='Training demand models...')
def train_cached_model(_data, cache_key): return model_demand(_data)

@st.cache_data(show_spinner=False)
def run_full_m5_cached(
    sales_path,
    prices_path,
    calendar_path,
    source_signature,
    analysis_version,
    _progress_callback,
):
    from full_m5_analysis import evaluate_full_m5

    return evaluate_full_m5(
        sales_path,
        prices_path,
        calendar_path,
        progress_callback=_progress_callback,
    )

source = st.sidebar.radio(
    'Data source',
    [
        'Synthetic pilot dataset',
        'M5 Walmart dataset',
        'Hugging Face Online Retail dataset',
    ],
)
if st.sidebar.button('Load dataset', type='primary'):
    st.session_state['dataset_source'] = source

loaded_source = st.session_state.get('dataset_source')
if loaded_source is None:
    st.info('Choose a data source, then select "Load dataset" to begin.')
    st.stop()

file_signature = data_file_signature()
df, source = load_data(loaded_source, file_signature)
if df is None: st.stop()
if source == 'Hugging Face Online Retail dataset':
    DISPLAY_FACTOR = 1.0
    DISPLAY_CURRENCY = 'GBP'
    DISPLAY_SYMBOL = '£'
else:
    DISPLAY_FACTOR = USD_TO_INR
    DISPLAY_CURRENCY = 'INR'
    DISPLAY_SYMBOL = '₹'
cache_key = (
    source,
    file_signature,
    M5_MAX_SERIES if source == 'M5 Walmart dataset' else None,
)
df = prepare_cached_data(df, cache_key)
st.sidebar.success(f'{source} loaded: {len(df):,} rows')
if source == 'M5 Walmart dataset':
    st.sidebar.caption(
        f'M5 uses a fixed random sample of up to {M5_MAX_SERIES:,} product/store '
        'series for interactive exploration. Use the full-M5 benchmark below for '
        'results covering every series.'
    )
if DISPLAY_CURRENCY == 'GBP':
    st.sidebar.caption('Money is shown in GBP (£) at the source value; demand and quantity are items. Unit cost is estimated, not included in the source.')
else:
    st.sidebar.caption(f'Indian units: money = INR (₹); demand and quantity = items; rates and changes = %. Display rate: $1 = ₹{USD_TO_INR:.0f}. Large totals are shown in INR; ₹1 crore = ₹10,000,000.')

st.sidebar.header('Business constraints')
min_margin = st.sidebar.slider('Minimum contribution margin %', 0, 80, 15)
max_discount = st.sidebar.slider('Maximum discount %', 0, 50, 30)
max_price_change = st.sidebar.slider('Maximum price change %', 5, 30, 15)

if source == 'M5 Walmart dataset':
    st.sidebar.subheader('Full M5 evaluation')
    st.sidebar.caption(
        'Runs a separate, batch-based benchmark on every M5 item/store series. '
        'Model fitting uses a deterministic bounded sample across all series, '
        'while validation/test metrics use every eligible row. The incremental '
        'model differs from the interactive sample benchmark.'
    )
    if st.sidebar.button('Run full M5 benchmark'):
        from full_m5_analysis import write_full_m5_report

        progress_bar = st.sidebar.progress(0.0, text='Preparing full M5 run...')
        with st.spinner('Evaluating all M5 item/store series. Keep this app open.'):
            full_results = run_full_m5_cached(
                *map(str, M5_FILES),
                file_signature,
                FULL_M5_ANALYSIS_VERSION,
                lambda value, text: progress_bar.progress(
                    value, text=text
                ),
            )
            progress_bar.progress(1.0, text='Full M5 benchmark ready')
            full_report = write_full_m5_report(
                full_results, FULL_M5_REPORT_PATH
            )
            st.session_state['full_m5_report'] = full_report
            st.session_state['full_m5_results'] = full_results
        st.sidebar.success(
            f"Full M5 benchmark complete: {full_results['series_count']:,} series, "
            f"{full_results['eligible_rows']:,} priced records. Report saved to "
            f"{FULL_M5_REPORT_PATH.relative_to(APP_DIR).as_posix()}."
        )

if 'full_m5_report' in st.session_state:
    st.sidebar.download_button(
        'Download full M5 findings',
        data=st.session_state['full_m5_report'],
        file_name='full_m5_results.md',
        mime='text/markdown',
    )

if 'full_m5_results' in st.session_state:
    full_results = st.session_state['full_m5_results']
    selected_name = full_results['selected_model']
    selected_metrics = full_results['selected_test']
    st.header('Full M5 benchmark results')
    st.caption(
        'All item/store series are included. Model fitting uses a deterministic '
        'bounded training sample; validation and test metrics cover every '
        'eligible priced row in their respective time windows.'
    )
    scope_columns = st.columns(4)
    scope_columns[0].metric('Item/store series', f"{full_results['series_count']:,}")
    scope_columns[1].metric('Priced daily records', f"{full_results['eligible_rows']:,}")
    scope_columns[2].metric('Selected model', selected_name)
    scope_columns[3].metric('Test RMSE (units)', f"{selected_metrics['RMSE']:.4f}")
    st.write(
        f"Date range: {full_results['date_start']} to {full_results['date_end']} | "
        f"Test MAE: {selected_metrics['MAE']:.4f} units | "
        f"Test R²: {selected_metrics['R2']:.4f} | "
        f"Training fit used {full_results['model_fit_rows']:,} of "
        f"{full_results['train_rows']:,} priced training rows | "
        f"Elapsed: {full_results['elapsed_seconds'] / 60:.1f} minutes"
    )
    comparison = pd.DataFrame(
        [
            {
                'Model': model_name,
                'Validation RMSE (units)': full_results['validation_results'][model_name]['RMSE'],
                'Test MAE (units)': full_results['test_results'][model_name]['MAE'],
                'Test RMSE (units)': full_results['test_results'][model_name]['RMSE'],
                'Test R²': full_results['test_results'][model_name]['R2'],
            }
            for model_name in full_results['validation_results']
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.info(
        'The downloadable report includes the full findings, research justification, '
        'no-survey explanation, methodology and limitations.'
    )

if st.sidebar.button('Generate latest results report'):
    with st.spinner('Training models and writing the report...'):
        model_results, trained_models = train_cached_model(df, cache_key)
        if source == 'Synthetic pilot dataset':
            report_paths = [next(path for path in SYNTHETIC_CANDIDATES if path.exists())]
        elif source == 'Hugging Face Online Retail dataset':
            report_paths = [
                ONLINE_RETAIL_CSV_PATH
                if ONLINE_RETAIL_CSV_PATH.exists()
                else ONLINE_RETAIL_PATH
            ]
        else:
            report_paths = M5_FILES
        report_source_path = '; '.join(
            path.resolve().relative_to(APP_DIR).as_posix() for path in report_paths
        )
        report_text = write_latest_results_report(
            df,
            source,
            report_source_path,
            model_results,
            trained_models,
            REPORT_PATH,
            min_margin=min_margin,
            max_discount=max_discount,
            max_price_change=max_price_change,
            usd_to_inr=DISPLAY_FACTOR,
            display_currency=DISPLAY_CURRENCY,
        )
        st.session_state['latest_results_report'] = report_text
    st.sidebar.success(f'Latest report saved to {REPORT_PATH.relative_to(APP_DIR).as_posix()}')

if 'latest_results_report' in st.session_state:
    st.sidebar.download_button(
        'Download latest report',
        data=st.session_state['latest_results_report'],
        file_name='latest_results.md',
        mime='text/markdown',
    )

view = st.sidebar.radio('Analysis', ['Executive Overview','Dataset Explorer','Demand Analysis','Price Elasticity','Promotion Impact','ML Models','Explainable AI','Price Response','Sensitivity','Business Strategy'])

if view == 'Executive Overview':
    st.header('Executive Business Overview')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Records (rows)',f'{len(df):,}'); c2.metric('Products (IDs)',df.Product_ID.nunique()); c3.metric('Stores (IDs)',df.Store_ID.nunique()); c4.metric('Categories (IDs)',df.Category.nunique())
    c1,c2=st.columns(2); c1.metric(f'Total Revenue ({DISPLAY_CURRENCY})',f'{DISPLAY_SYMBOL}{df.Revenue.sum() * DISPLAY_FACTOR:,.2f}'); c2.metric('Total Units (items)',f'{df.Quantity_Sold.sum():,.0f}')
    st.markdown('''### Research logic\n**Historical data → demand estimation → price sensitivity → ML prediction → explainability → price scenarios → revenue/contribution → constraints → risk.**''')
    if source == 'M5 Walmart dataset': st.info('M5 contains historical Walmart sales/prices but not actual product-level cost. Contribution is therefore based on an explicit analytical cost assumption.')
    elif source == 'Hugging Face Online Retail dataset': st.info('Source: Hugging Face michaelmallari/online-retail (UCI Online Retail; UK transactions, GBP). This is a separate benchmark, not Walmart data and must not be used to fill M5 gaps. Cancellations/returns are excluded; product costs and explicit promotions are absent, so contribution and promotion fields are analytical estimates.')

elif view == 'Dataset Explorer':
    st.header('Dataset Explorer'); st.dataframe(display_units(df.head(100)),use_container_width=True); st.dataframe(display_summary(df),use_container_width=True); st.bar_chart(display_units(df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)))

elif view == 'Demand Analysis':
    st.header('Demand & Sales Analysis')
    daily=df.groupby('Date').agg(Quantity_Sold=('Quantity_Sold','sum'),Revenue=('Revenue','sum'),Average_Price=('Selling_Price','mean'))
    st.line_chart(display_units(daily[['Quantity_Sold','Revenue']]))
    st.subheader('Category performance')
    st.dataframe(display_units(df.groupby('Category').agg(Units=('Quantity_Sold','sum'),Revenue=('Revenue','sum'),Avg_Price=('Selling_Price','mean')).sort_values('Revenue',ascending=False)),use_container_width=True)
    st.subheader('Price vs quantity'); st.scatter_chart(display_units(df.sample(min(10000,len(df)),random_state=42)[['Selling_Price','Quantity_Sold']]))

elif view == 'Price Elasticity':
    st.header('Price Elasticity of Demand')
    st.write(
        'Absolute elasticity magnitude = '
        '|% change in quantity / % change in price|.'
    )
    st.metric(
        'Overall estimated absolute elasticity magnitude',
        f'{abs(calculate_price_elasticity(df)):.3f}',
    )
    category_elasticity = calculate_price_elasticity(df, 'Category', True)
    category_elasticity['Elasticity'] = category_elasticity['Elasticity'].abs()
    category_elasticity = category_elasticity.rename(
        columns={'Elasticity': 'Absolute Elasticity Magnitude'}
    )
    st.subheader('By category')
    st.dataframe(display_units(category_elasticity), use_container_width=True)
    store_elasticity = calculate_price_elasticity(df, 'Store_ID', True)
    store_elasticity['Elasticity'] = store_elasticity['Elasticity'].abs()
    store_elasticity = store_elasticity.rename(
        columns={'Elasticity': 'Absolute Elasticity Magnitude'}
    )
    st.subheader('By store')
    st.dataframe(display_units(store_elasticity), use_container_width=True)
    st.caption(
        'The sign is omitted: a larger magnitude means quantity historically '
        'changed more with price. This observational estimate is not necessarily '
        'causal.'
    )

elif view == 'Promotion Impact':
    st.header('Promotion Impact'); p=calculate_promotion_lift(df); a,b,c=st.columns(3); a.metric('Demand lift (%)',f"{p['demand_lift_pct']:.2f}%"); b.metric('Revenue lift (%)',f"{p['revenue_lift_pct']:.2f}%"); c.metric('Contribution lift (%)',f"{p['contribution_lift_pct']:.2f}%"); st.dataframe(display_units(p['comparison']),use_container_width=True); st.warning('Higher sales volume does not necessarily mean higher contribution.')

elif view == 'ML Models':
    st.header('Forecast Benchmark & Model Comparison'); results,trained=train_cached_model(df, cache_key); st.caption('Models are selected on validation dates and evaluated on untouched later dates. Forecasts are daily product/store totals in a rolling one-day-ahead setting.'); st.dataframe(display_units(results),use_container_width=True); st.bar_chart(display_units(results.set_index('Model')[['R2']])); st.success(f"Selected by validation RMSE: {results.iloc[0]['Model']} | untouched final-test RMSE: {results.iloc[0]['RMSE']:.2f} units")

elif view == 'Explainable AI':
    st.header('Explainable AI'); results,trained=train_cached_model(df, cache_key); name=st.selectbox('Model',list(trained.keys())); imp=explain_model(df,trained[name])
    if imp.empty: st.info('The training-mean baseline has no feature importance values.')
    else: st.dataframe(display_units(imp),use_container_width=True); st.bar_chart(display_units(imp.set_index('Feature')[['Importance']]))

elif view == 'Price Response':
    st.header('Continuous Price Response Curve'); product=st.selectbox('Product ID',sorted(df.Product_ID.unique())); store=st.selectbox('Store ID',sorted(df.Store_ID.unique())); results,trained=train_cached_model(df, cache_key); best=results.iloc[0]['Model']; curve=build_price_response_curve(df,trained[best],product,store,max_price_change); st.dataframe(display_units(curve),use_container_width=True); st.subheader('Predicted demand (units)'); st.line_chart(display_units(curve.set_index('Price')[['Predicted_Demand']])); st.subheader(f'Revenue and contribution ({DISPLAY_CURRENCY})'); st.line_chart(display_units(curve.set_index('Price')[['Revenue','Contribution']])); feasible=curve[(curve.Margin_Pct>=min_margin)&(curve.Discount_Pct<=max_discount)]
    if len(feasible):
        st.write(f'Revenue-maximizing feasible scenario ({DISPLAY_CURRENCY})'); st.dataframe(display_units(feasible.loc[[feasible.Revenue.idxmax()]]),use_container_width=True); st.write(f'Contribution-maximizing feasible scenario ({DISPLAY_CURRENCY})'); st.dataframe(display_units(feasible.loc[[feasible.Contribution.idxmax()]]),use_container_width=True)
    else: st.warning('No scenario satisfies the selected constraints.')

elif view == 'Sensitivity':
    st.header('Sensitivity & Risk Analysis'); results,trained=train_cached_model(df, cache_key); best=trained[results.iloc[0]['Model']]; sensitivity=run_sensitivity_analysis(df,best,str(df.Product_ID.iloc[0]),str(df.Store_ID.iloc[0]),min_margin,max_discount,max_price_change=max_price_change); st.dataframe(display_units(sensitivity),use_container_width=True); st.write(f'Scenarios test conservative, base and optimistic demand assumptions. Prices and financial outcomes are in {DISPLAY_CURRENCY}; demand is in units.')

elif view == 'Business Strategy':
    st.header('Business Strategy & Cross-Industry Transfer'); segments=business_segments(df); st.dataframe(display_units(segments),use_container_width=True)
    for x in generate_business_insights(df,segments): st.markdown(f'- {x}')
    transfer=pd.DataFrame({'Industry':['Retail','E-commerce','Hotels','Airlines','Restaurants','SaaS','Telecom'],f'Price measure ({DISPLAY_CURRENCY}/item or service)':['Product','Product','Room','Fare','Menu/delivery','Subscription','Plan'],'Demand measure (count/rate)':['Units','Orders/conversion','Bookings','Bookings','Orders','Subscriptions','Adoption'],f'Financial outcome ({DISPLAY_CURRENCY}/KPI)':['Revenue + contribution','Revenue + contribution','Room revenue + margin','Revenue + contribution','Revenue + contribution','MRR/LTV/contribution','Revenue/LTV/contribution']})
    st.dataframe(transfer,use_container_width=True)

render_constraint_outcome(df, min_margin, max_discount, max_price_change)
st.divider(); st.caption('Research prototype. Validate costs, constraints, causality and governance before real deployment.')
