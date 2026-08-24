import pandas as pd

napcs_col = 'North American Product Classification System (NAPCS)'

# Tariffed commodities
treated = [
    'Lumber and other sawmill products [241]',
    'Unwrought iron, steel and ferro-alloys [311]',
    'Basic and semi-finished iron or steel products [312]',
    'Unwrought aluminum and aluminum alloys [321]',
    'Basic and semi-finished products of aluminum and aluminum alloys [327]',
    'Dairy products [173]',
]

# Confirmed exempt from tariffs
control = [
    'Crude oil and bitumen [141]',
    'Natural gas [142]',
    'Potash [161]',
    'Fish, crustaceans, shellfish and other fishery products [121]',
]

df = pd.read_csv('12100163.csv')
print(f'Loaded {len(df):,} rows')

# Exports only - tariffs hurt what Canada sells, not what it buys
df = df[df['Trade'] == 'Export']
print(f'After export filter: {len(df):,} rows')

# Keep only our commodities
df = df[df[napcs_col].isin(treated + control)]
print(f'After commodity filter: {len(df):,} rows')

# one measurement method, custom is the border-crossing record
df = df[df['Basis'] == 'Customs']

# Raw numbers, our model handles seasonality itself
df = df[df['Seasonal adjustment'] == 'Unadjusted']
print(f'After basis/adjustment filter: {len(df):,} rows')

# Drop StatCan internal ID columns we don't need
df = df[['REF_DATE', napcs_col, 'VALUE', 'Basis',
         'Seasonal adjustment', 'SCALAR_FACTOR', 'STATUS', 'UOM']].copy()

# Check these before cleaning - each one can silently break the analysis

# Multiple bases would mean double-counting the same month
print('\nBasis:')
print(df['Basis'].value_counts())

# If StatCan already removed seasonality, our model shouldn't model it again
print('\nSeasonal adjustment:')
print(df['Seasonal adjustment'].value_counts())

# Mixed units (thousands vs millions) would make values incomparable
print('\nScalar factor:')
print(df['SCALAR_FACTOR'].value_counts())

print('\nUnit of measure:')
print(df['UOM'].value_counts())

# Flags suppressed or unavailable values - blank means normal
print('\nSTATUS:')
print(df['STATUS'].value_counts(dropna=False))

# Uneven counts would mean some series are incomplete
print('\nRows per commodity:')
print(df[napcs_col].value_counts())

print(f'\nMissing VALUE entries: {df["VALUE"].isna().sum()}')

# Writes filtered table out to a new csv file, and stopping pandas from writing its row numbers as an extra first col (index=False)
df.to_csv('filtered_raw.csv', index=False)
print('\nSaved filtered_raw.csv')