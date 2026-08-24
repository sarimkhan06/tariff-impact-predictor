import pandas as pd
from sklearn.linear_model import LinearRegression

napcs_col = 'North American Product Classification System (NAPCS)'

# Each commodity paired with the month its tariff took effect.
# Steel/aluminum: Mar 2025 (cutoff Feb). Lumber: Oct 2025 (cutoff Sep).
# Controls were never tariffed - we use Feb 2025 just so they're measured
# over the same window as steel, for a fair comparison.
commodities = {
    'Basic and semi-finished iron or steel products [312]': '2025-02-01',
    'Unwrought iron, steel and ferro-alloys [311]': '2025-02-01',
    'Unwrought aluminum and aluminum alloys [321]': '2025-02-01',
    'Basic and semi-finished products of aluminum and aluminum alloys [327]': '2025-02-01',
    'Lumber and other sawmill products [241]': '2025-09-01',
    'Crude oil and bitumen [141]': '2025-02-01',
    'Natural gas [142]': '2025-02-01',
    'Potash [161]': '2025-02-01',
    'Fish, crustaceans, shellfish and other fishery products [121]': '2025-02-01'
}

features = ['lag_1', 'lag_12', 'time_index']


# Runs the whole Model 1 process for ONE commodity, returns its average gap.
def run_commodity(name, cutoff_str):
     # turn the date text into a real timestamp so we can compare dates below
    cutoff = pd.Timestamp(cutoff_str)
 
    # load the clean data and turn the date column into real dates
    df = pd.read_csv('filtered_raw.csv')
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
 
    # keep only this one commodity's rows
    df = df[df[napcs_col] == name]
 
    # oldest to newest, so "last month" actually means the row above
    df = df.sort_values('REF_DATE').reset_index(drop=True)
 
    # build the three features the model learns from
    df['lag_1'] = df['VALUE'].shift(1)      # last month's exports
    df['lag_12'] = df['VALUE'].shift(12)    # exports 12 months ago (seasonality)
    df['time_index'] = range(len(df))       # 0,1,2... counter for the trend
 
    # the first 12 rows have blank lags (nothing that far back) - drop them
    df = df.dropna(subset=['lag_1', 'lag_12'])
 
    # training data = everything before the tariff (what "normal" looks like)
    train = df[df['REF_DATE'] < cutoff]
 
    # fit the regression on the pre-tariff data only
    model = LinearRegression()
    model.fit(train[features], train['VALUE'])
 
    # the tariff-period rows we want to forecast a "no tariff" line for
    after = df[df['REF_DATE'] >= cutoff].copy()
 
    # running list of values the forecast uses for its lags.
    # starts as all the real pre-tariff values, then grows with each prediction.
    history = list(train['VALUE'])
 
    # forecast each tariff-period month one at a time
    predictions = []
    for _, row in after.iterrows():
        # features for this month: last month and same-month-last-year come
        # from history (which holds predictions once we're past the start),
        # time_index comes straight off the row since a tariff can't change it
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
 
        # predict returns a list, [0] pulls out the single number
        pred = model.predict(x)[0]
 
        predictions.append(pred)   # store it for the results
        history.append(pred)       # feed it forward so next month builds on it
 
    # attach the counterfactual and compute the % gap vs what really happened
    after['predicted'] = predictions
    after['gap_pct'] = (after['VALUE'] - after['predicted']) / after['predicted'] * 100
 
    # hand back one number: the average gap over the whole tariff period
    return after['gap_pct'].mean()


# Run every commodity and collect the results
results = []
# iterate thru commodities hashmap
for name, cutoff_str in commodities.items():
    avg_gap = run_commodity(name, cutoff_str)
    results.append({'commodity': name, 'avg_gap_pct': round(avg_gap, 1)})

# turn the results into a small table, sorted most-hurt to least
summary = pd.DataFrame(results).sort_values('avg_gap_pct')
print(summary.to_string(index=False))
 
# save it as a real output file
summary.to_csv('results.csv', index=False)
print('\nSaved results.csv')