import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

features = ['lag_1', 'lag_12', 'time_index']
train_end = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2025-01-01')

df_all = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
df_all['REF_DATE'] = pd.to_datetime(df_all['REF_DATE'])
codes = df_all['commodity'].unique()


def mape(actual, predicted):
    errors = (predicted - actual).abs() / actual.replace(0, pd.NA) * 100
    return errors.mean()


# Runs the same recursive forecast with whichever model is passed in,
# so linear and forest get treated identically - the only difference
# between them is the model itself.
def forecast_with(model, train, test):
    model.fit(train[features], train['VALUE'])

    history = list(train['VALUE'])
    preds = []
    for _, row in test.iterrows():
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
        pred = model.predict(x)[0]
        preds.append(pred)
        history.append(pred)
    return pd.Series(preds, index=test.index)


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

    linear_preds = forecast_with(LinearRegression(), train, test)
    # random_state=42 so the result is reproducible - forests involve
    # randomness, and without this you'd get slightly different numbers
    # every run
    forest_preds = forecast_with(
        RandomForestRegressor(n_estimators=100, random_state=42), train, test
    )

    return {
        'commodity': code,
        'linear_mape': round(mape(test['VALUE'], linear_preds), 1),
        'forest_mape': round(mape(test['VALUE'], forest_preds), 1),
    }


results = []
for code in codes:
    r = compare_one(code)
    if r is not None:
        results.append(r)

summary = pd.DataFrame(results)
summary['winner'] = summary.apply(
    lambda r: 'linear' if r['linear_mape'] < r['forest_mape'] else 'forest', axis=1
)

print(summary.to_string(index=False))
print()
print('Medians:')
print(summary[['linear_mape', 'forest_mape']].median().round(1).to_string())
print()
print('Wins:')
print(summary['winner'].value_counts().to_string())