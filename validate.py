import pandas as pd
from sklearn.linear_model import LinearRegression

napcs_col = 'North American Product Classification System (NAPCS)'
features = ['lag_1', 'lag_12', 'time_index']

# Holdout test: train on everything before 2024, forecast all of 2024,
# compare to what actually happened. 2024 is pre-tariff, so any error
# here is pure model error - it tells us how accurate the forecast is.
train_end = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2025-01-01')

commodities = [
    'Basic and semi-finished iron or steel products [312]',
    'Unwrought iron, steel and ferro-alloys [311]',
    'Unwrought aluminum and aluminum alloys [321]',
    'Basic and semi-finished products of aluminum and aluminum alloys [327]',
    'Lumber and other sawmill products [241]',
    'Passenger cars and light trucks [411]',
    'Medium and heavy trucks, buses, and other motor vehicles [412]',
    'Motor vehicle engines and motor vehicle parts [413]',
    'Furniture and fixtures [391]',
    'Crude oil and bitumen [141]',
    'Natural gas [142]',
    'Potash [161]',
    'Fish, crustaceans, shellfish and other fishery products [121]',
]


def holdout_test(name):
    df = pd.read_csv('filtered_raw.csv')
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df[df[napcs_col] == name]
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    train = df[df['REF_DATE'] < train_end]
    test = df[(df['REF_DATE'] >= train_end) & (df['REF_DATE'] < test_end)]

    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

    # same recursive forecast as model1 - feed predictions forward
    history = list(train['VALUE'])
    predictions = []
    for _, row in test.iterrows():
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
        pred = model.predict(x)[0]
        predictions.append(pred)
        history.append(pred)

    test = test.copy()
    test['predicted'] = predictions

    # MAPE: average size of the error as a % of the real value
    errors = (test['predicted'] - test['VALUE']).abs() / test['VALUE'] * 100
    return errors.mean()


results = []
for name in commodities:
    mape = holdout_test(name)
    results.append({'commodity': name, 'mape_pct': round(mape, 1)})

summary = pd.DataFrame(results).sort_values('mape_pct')
print('Forecast error on 2024 holdout (no tariffs in this period):')
print(summary.to_string(index=False))
print()
print(f'Median error across commodities: {summary["mape_pct"].median():.1f}%')