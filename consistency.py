import pandas as pd
from sklearn.linear_model import LinearRegression

napcs_col = 'North American Product Classification System (NAPCS)'
features = ['lag_1', 'lag_12', 'time_index']

commodities = {
    'Basic and semi-finished iron or steel products [312]': '2025-02-01',
    'Unwrought iron, steel and ferro-alloys [311]': '2025-02-01',
    'Unwrought aluminum and aluminum alloys [321]': '2025-02-01',
    'Basic and semi-finished products of aluminum and aluminum alloys [327]': '2025-02-01',
    'Lumber and other sawmill products [241]': '2025-09-01',
    'Passenger cars and light trucks [411]': '2025-03-01',
    'Medium and heavy trucks, buses, and other motor vehicles [412]': '2025-03-01',
    'Motor vehicle engines and motor vehicle parts [413]': '2025-03-01',
    'Furniture and fixtures [391]': '2025-09-01',
    'Crude oil and bitumen [141]': '2025-02-01',
    'Potash [161]': '2025-02-01',
    'Fish, crustaceans, shellfish and other fishery products [121]': '2025-02-01',
}


# same as model1, but returns every month's gap instead of just the average
def get_gaps(name, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)

    df = pd.read_csv('filtered_raw.csv')
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df[df[napcs_col] == name]
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
    return after['gap_pct']


results = []
for name, cutoff_str in commodities.items():
    gaps = get_gaps(name, cutoff_str)
    negative = (gaps < 0).sum()   # how many months came in below the counterfactual
    total = len(gaps)
    results.append({
        'commodity': name,
        'months_negative': f'{negative}/{total}',
        'pct_negative': round(negative / total * 100),
        'avg_gap': round(gaps.mean(), 1),
    })

summary = pd.DataFrame(results).sort_values('pct_negative', ascending=False)
print('A real tariff effect should be negative in MOST months, not just on average.')
print()
print(summary.to_string(index=False))