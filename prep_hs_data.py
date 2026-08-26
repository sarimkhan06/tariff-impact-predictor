import pandas as pd

min_total_value = 500_000_000

df = pd.read_csv('hs_filtered.csv', dtype={'commodity_code': str})
print(f'Loaded {len(df):,} rows')

# columns from load_hs_data.py: yearmonth, hs8, country, state, value,
# quantity, uom, heading4, heading6, commodity_code

monthly = df.groupby(['yearmonth', 'commodity_code'])[
    'value'].sum().reset_index()
print(f'After aggregating to (month, code): {len(monthly):,} rows')

totals = monthly.groupby('commodity_code')['value'].sum()
keep = totals[totals >= min_total_value].index
monthly = monthly[monthly['commodity_code'].isin(keep)]
print(f'Codes kept (>= ${min_total_value:,}): {len(keep)}')
print(f'After volume filter: {len(monthly):,} rows')

# A missing month means Canada exported ZERO of that code that month -
# it's a real data point, not missing data. Fill the gap with 0 rather
# than dropping the whole series, so lumber's genuinely slow months
# don't wipe out an otherwise valid commodity.
all_months = pd.date_range('2015-01-01', '2026-06-01', freq='MS')
codes = monthly['commodity_code'].unique()

# build every (month, code) combination that SHOULD exist
full_index = pd.MultiIndex.from_product(
    [all_months.strftime('%Y%m').astype(int), codes],
    names=['yearmonth', 'commodity_code']
)
monthly = monthly.set_index(['yearmonth', 'commodity_code'])
monthly = monthly.reindex(full_index, fill_value=0).reset_index()

print(f'After filling gaps with zero: {len(monthly):,} rows')

monthly['REF_DATE'] = pd.to_datetime(
    monthly['yearmonth'].astype(str), format='%Y%m')
monthly['commodity'] = monthly['commodity_code']
monthly['VALUE'] = monthly['value']
monthly = monthly[['REF_DATE', 'commodity', 'VALUE']]

counts = monthly.groupby('commodity').size().sort_values()
print()
print('Months of data per code (should all be equal):')
print(counts.to_string())

print()
print(
    f'Date range: {monthly["REF_DATE"].min().date()} to {monthly["REF_DATE"].max().date()}')
print(f'Missing values: {monthly["VALUE"].isna().sum()}')

monthly.to_csv('hs_monthly.csv', index=False)
print()
print('Saved hs_monthly.csv')
