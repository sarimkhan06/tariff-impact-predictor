import pandas as pd
from sklearn.linear_model import LinearRegression

features = ['lag_1', 'lag_12', 'time_index']

# train on everything before 2024, test on 2024, 2024 had no tariffs, so any error here is pure model, nothing to do with tariffs at all
train_end = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2025-01-01')

df_all = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
codes = df_all['commodity'].unique()

# same style as run_commodity function, but this time trains on pre-2024 data and checks its guess against 2024's real values which we do have
def holdout_test(name):
    df = df_all[df_all['commodity'] == name].copy()
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    # train everything before jan 2024 (all pre-2024 history)
    train = df[df['REF_DATE'] < train_end]
    # test = only the rows between jan 2024 and jan 2025
    test = df[(df['REF_DATE'] >= train_end) & (df['REF_DATE'] < test_end)]

    # if a commodity somehow has fewer than 20 training rows, or 0 test rows, there's not enough data to run a fair test
    if len(train) < 20 or len(test) == 0:
        return None

    # same training concept from model1 file
    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

    # same recursive forecast as run_commodity, feeding predictions forward
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
    # MAPE = Mean Absolute Percentage Error, .abs() makes every error positive ( e.g: $10 overshoot and a $10 undershoot count the same)
    # divide by the real value to make it a %, .replace(0, pd.NA) avoids dividing by 0 if a month had no real exports at all
    # other words, % = average of |actual - predicted| / actual * 100
    errors = (test['predicted'] - test['VALUE']).abs() / test['VALUE'].replace(0, pd.NA) * 100
    return errors.mean()

# run the holdout test on every commodity in the file, collect the results
results = []
for code in codes:
    mape = holdout_test(code)
    if mape is not None:
        results.append({'commodity': code, 'mape_pct': round(mape, 1)})

# turns the result into a table, sorted most accurate to least accurate, and prints it out
summary = pd.DataFrame(results).sort_values('mape_pct')
print('Forecast error on 2024 holdout (no tariffs in this period):')
print(summary.to_string(index=False))
print()
print(f'Median error: {summary["mape_pct"].median():.1f}%')