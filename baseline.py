import pandas as pd
from sklearn.linear_model import LinearRegression

features = ['lag_1', 'lag_12', 'time_index']
train_end = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2025-01-01')

df_all = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
df_all['REF_DATE'] = pd.to_datetime(df_all['REF_DATE'])
codes = df_all['commodity'].unique()


def mape(actual, predicted):
    errors = (predicted - actual).abs() / actual.replace(0, pd.NA) * 100
    return errors.mean()


def compare_one(code):
    df = df_all[df_all['commodity'] == code].copy()
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    train = df[df['REF_DATE'] < train_end]
    test = df[(df['REF_DATE'] >= train_end) & (df['REF_DATE'] < test_end)]

    if len(train) < 20 or len(test) == 0:
        return None

    # --- your actual model - recursive, feeds its own predictions forward ---
    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

    history = list(train['VALUE'])
    model_preds = []
    for _, row in test.iterrows():
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
        pred = model.predict(x)[0]
        model_preds.append(pred)
        history.append(pred)

    # --- naive "last month"
    # Guess month 1 = real last-known value. Guess month 2 = its OWN
    # guess for month 1. No peeking at real data once the test period starts.
    naive_month_history = list(train['VALUE'])
    naive_month_preds = []
    for _ in range(len(test)):
        naive_month_preds.append(naive_month_history[-1])
        naive_month_history.append(naive_month_history[-1])

    # --- naive "last year" - same fix, recursive using its own chain
    naive_year_history = list(train['VALUE'])
    naive_year_preds = []
    for _ in range(len(test)):
        naive_year_preds.append(naive_year_history[-12])
        naive_year_history.append(naive_year_history[-12])

    return {
        'commodity': code,
        'model_mape': round(mape(test['VALUE'], pd.Series(model_preds, index=test.index)), 1),
        'naive_month_mape': round(mape(test['VALUE'], pd.Series(naive_month_preds, index=test.index)), 1),
        'naive_year_mape': round(mape(test['VALUE'], pd.Series(naive_year_preds, index=test.index)), 1),
    }


results = []
for code in codes:
    r = compare_one(code)
    if r is not None:
        results.append(r)

summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print()
print('Medians:')
print(summary[['model_mape', 'naive_month_mape', 'naive_year_mape']].median().round(1).to_string())