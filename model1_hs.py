import pandas as pd
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# Each commodity code paired with (cutoff date, tariff rate).
# Cutoff = the month training stops - just before the tariff took effect.
# Steel/aluminum (Section 232): 25% Mar 12 2025, escalated to 50% Jun 4 2025.
#   Using the whole post-tariff window as one period, so cutoff = Feb 2025.
# Lumber and furniture (Section 232, Proclamation 10976): effective
#   Oct 14 2025, so cutoff = Sep 2025.
# --------------------------------------------------------------------------

steel_aluminum = [
    '7206', '7207', '7208', '7209', '7210', '7211', '7212', '7213', '7214',
    '7215', '7216', '7217', '7218', '7224', '7225', '7226', '7227', '7228',
    '7304', '7305', '7306',
    '7601', '7604', '7605', '7606', '7607',
]
lumber = ['440311', '440711', '440713', '440714', '440719']
furniture = ['940161']

# Controls: never tariffed. Use the same Feb 2025 cutoff as steel/aluminum
# so they're measured over a comparable window - there's no real tariff
# date for them, this just keeps the comparison fair.
controls = ['0302', '0303', '0304', '0305', '2709', '2711', '3104']

rate_info = {}
for code in steel_aluminum:
    rate_info[code] = {'cutoff': '2025-02-01', 'rate': 50, 'group': 'treated'}
for code in lumber:
    rate_info[code] = {'cutoff': '2025-09-01', 'rate': 10, 'group': 'treated'}
for code in furniture:
    rate_info[code] = {'cutoff': '2025-09-01', 'rate': 25, 'group': 'treated'}
for code in controls:
    rate_info[code] = {'cutoff': '2025-02-01', 'rate': 0, 'group': 'control'}

# Excluded: validate_hs.py showed these have MAPE 67-820% on a no-tariff
# holdout year (2024) - the model can't forecast them reliably at all,
# likely due to lumpy/small trade volumes. Their extreme gaps in earlier
# runs were model failure, not real tariff effects (same pattern as
# natural gas at the NAPCS level).
broken_codes = ['7224', '7218', '440719', '7207', '440311', '7305']
rate_info = {code: info for code, info in rate_info.items() if code not in broken_codes}

# Some codes in rate_info may have been dropped from hs_monthly.csv by the
# volume cutoff in prep_hs_data.py (e.g. 7206 was too small: $11.8M total).
# Only keep codes that actually exist in the data, and report anything skipped.
available_codes = set(
    pd.read_csv('hs_monthly.csv', dtype={'commodity': str})['commodity'].unique()
)
commodities = {code: info for code, info in rate_info.items() if code in available_codes}

skipped = set(rate_info) - available_codes
if skipped:
    print(f'Skipping codes not present in hs_monthly.csv: {sorted(skipped)}')
print(f'Running Model 1 on {len(commodities)} commodities')
print()

features = ['lag_1', 'lag_12', 'time_index']


def run_commodity(name, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)

    df = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df[df['commodity'] == name]
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    train = df[df['REF_DATE'] < cutoff]

    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

    after = df[df['REF_DATE'] >= cutoff].copy()
    history = list(train['VALUE'])

    predictions = []
    for _, row in after.iterrows():
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
        pred = model.predict(x)[0]
        predictions.append(pred)
        history.append(pred)

    after['predicted'] = predictions
    after['gap_pct'] = (after['VALUE'] - after['predicted']) / after['predicted'] * 100

    return after['gap_pct'].mean()


results = []
for code, info in commodities.items():
    avg_gap = run_commodity(code, info['cutoff'])
    results.append({
        'commodity': code,
        'group': info['group'],
        'rate': info['rate'],
        'avg_gap_pct': round(avg_gap, 1),
    })

summary = pd.DataFrame(results).sort_values('avg_gap_pct')
print(summary.to_string(index=False))

summary.to_csv('hs_results.csv', index=False)
print()
print('Saved hs_results.csv')

print()
print('Average gap by group:')
print(summary.groupby('group')['avg_gap_pct'].mean().round(1).to_string())