import pandas as pd
import glob

# Every steel + aluminum HS heading covered by Section 232 (4-digit match,
# whole headings are covered so this level of granularity is fine).
steel_headings = [
    '7206', '7207', '7224',
    '7208', '7209', '7210', '7211', '7212', '7225', '7226',
    '7213', '7214', '7215', '7227', '7228',
    '7216',
    '7217', '7229',
    '7218', '7219', '7220', '7221', '7222', '7223',
    '7301', '7302',
    '7304', '7305', '7306',
]
aluminum_headings = ['7601', '7604', '7605', '7606', '7607', '7608', '7609']

# Lumber and furniture: the US tariff only covers specific 6-digit codes
# (not whole headings), and Canadian export codes only match international
# codes up to 6 digits - so these must be matched at HS6, not HS4.
# Verified against CBP guidance (Proclamation 10976, effective Oct 14 2025).
lumber_hs6 = ['440311', '440321', '440322', '440323', '440324',
              '440325', '440326', '440399', '440611', '440691',
              '440711', '440712', '440713', '440714', '440719']
furniture_hs6 = ['940161']

# Controls: confirmed exempt from Section 232 tariffs. Whole headings
# are fine here since none of the underlying goods are tariffed.
control_headings = ['2709', '2711', '3104', '0301', '0302', '0303', '0304', '0305']

# NOT included, and why:
# - Kitchen cabinets (9403.40/60/91): these HS6 codes contain BOTH tariffed
#   cabinets/vanities AND untariffed "other" products in the same code, per
#   CBP guidance. Can't isolate the tariffed portion from export data alone.
# - Vehicles/parts (8702/8703/8704/8708): the annex only tariffs parts
#   confirmed for use in specific vehicle types - not determinable from
#   trade data.

target_headings_hs4 = steel_headings + aluminum_headings + control_headings
target_codes_hs6 = lumber_hs6 + furniture_hs6

# loading the files
files = glob.glob('cimt_exports/*.csv')
print(f'Found {len(files)} yearly files')

# reads each year's file, collects them in a list, and glues them into 1 big table
all_years = []
for f in files:
    year_df = pd.read_csv(f, encoding='latin1')
    all_years.append(year_df)
df = pd.concat(all_years, ignore_index=True)
print(f'Combined: {len(df):,} total rows')

# Rename by position - the accented column names get mangled on save/reload
df.columns = ['yearmonth', 'hs8', 'country', 'state', 'value', 'quantity', 'uom']

# filtering to US exports
df = df[df['country'] == 'US']
print(f'After US filter: {len(df):,} rows')

df['hs8'] = df['hs8'].astype(str).str.zfill(8)
df['heading4'] = df['hs8'].str[:4]
df['heading6'] = df['hs8'].str[:6]

# Two separate matches: HS4 for steel/aluminum, HS6 for lumber/furniture,
# then combine. Use heading6 as the single "commodity" label going forward
# for the HS6 group, and heading4 for the HS4 group, so each row is tagged
# with the right granularity.
match_hs4 = df[df['heading4'].isin(target_headings_hs4)].copy()
match_hs4['commodity_code'] = match_hs4['heading4']

match_hs6 = df[df['heading6'].isin(target_codes_hs6)].copy()
match_hs6['commodity_code'] = match_hs6['heading6']

combined = pd.concat([match_hs4, match_hs6], ignore_index=True)
print(f'After heading filter: {len(combined):,} rows')

# shows total export value per code across all years
by_code = combined.groupby('commodity_code')['value'].sum().sort_values(ascending=False)
print()
print('Total export value by code (all years, all months):')
print(by_code.to_string())

# saves the result for the next script
combined.to_csv('hs_filtered.csv', index=False)
print()
print('Saved hs_filtered.csv')