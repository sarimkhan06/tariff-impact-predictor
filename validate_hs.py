import pandas as pd
from sklearn.linear_model import LinearRegression

features = ['lag_1', 'lag_12', 'time_index']
train_end = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2025-01-01')

df_all = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
codes = df_all['commodity'].unique()


def holdout_test(name):
    df = df_all[df_all['commodity'] == name].copy()
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    train = df[df['REF_DATE'] < train_end]
    test = df[(df['REF_DATE'] >= train_end) & (df['REF_DATE'] < test_end)]

    if len(train) < 20 or len(test) == 0:
        return None

    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

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
    errors = (test['predicted'] - test['VALUE']).abs() / test['VALUE'].replace(0, pd.NA) * 100
    return errors.mean()


results = []
for code in codes:
    mape = holdout_test(code)
    if mape is not None:
        results.append({'commodity': code, 'mape_pct': round(mape, 1)})

summary = pd.DataFrame(results).sort_values('mape_pct')
print('Forecast error on 2024 holdout (no tariffs in this period):')
print(summary.to_string(index=False))
print()
print(f'Median error: {summary["mape_pct"].median():.1f}%')