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

# nested hashmap, key of rate_info would be the code, and that code has its own hashmap with keys cutoff, rate, and group
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

 # pulls out the commodity column from the csv file, and gets the list of every distinct code thats actually in there then puts it into a set
available_codes = set(pd.read_csv('hs_monthly.csv', dtype={'commodity': str})['commodity'].unique())

# builds a new hashmap goes thru every entry in rate_info and only keeps the ones where the code is also found in avaliable_codes
commodities = {code: info for code, info in rate_info.items() if code in available_codes}

skipped = set(rate_info) - available_codes
# if anything got skipped, print exactly which codes
if skipped:
    print(f'Skipping codes not present in hs_monthly.csv: {sorted(skipped)}')
print(f'Running Model 1 on {len(commodities)} commodities')
print()

features = ['lag_1', 'lag_12', 'time_index']


# Runs the whole Model 1 process for ONE commodity code, returns its average gap over the tariff period
def run_commodity(name, cutoff_str):
    # turn the date text into a real date object so it can be compared against other dates
    cutoff = pd.Timestamp(cutoff_str)

    # load ALL commodities' data - dtype=str keeps codes like '0302' from losing their leading zero
    df = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})

    # dates in the file are just text - convert to real dates, same reason as the cutoff line above
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

    # keep only this one function call's specific code, e.g. just 7208
    df = df[df['commodity'] == name]

    # oldest to newest - required for the lag features below to mean
    # "the row above" instead of some random order
    df = df.sort_values('REF_DATE').reset_index(drop=True)

    # three features: last month's value, same month last year (seasonality),
    # and a 0,1,2... counter capturing the long-run trend
    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))

    # first 12 rows have no lag_12 yet (nothing exists that far back) - drop them
    df = df.dropna(subset=['lag_1', 'lag_12'])

    # training data = only months BEFORE the tariff took effect -
    # this is what the model learns "normal" behaviour from
    train = df[df['REF_DATE'] < cutoff]

    model = LinearRegression()
    
    # training the model using the features, 1st parameter represents the clues/features, 2nd parameters represent the answer, what actually happened
    model.fit(train[features], train['VALUE'])

    # the tariff-period rows we want to build a "no tariff" forecast for
    after = df[df['REF_DATE'] >= cutoff].copy()

    # running history starts as the REAL pre-tariff values.
    # from here on we only add PREDICTIONS to it, never actuals, that's what keeps the forecast living in a world where the tariff never happened
    history = list(train['VALUE'])

    predictions = []
    for _, row in after.iterrows():
        # build this month's feature row: last two values come from
        # history (predictions once we're past the first step),
        # time_index comes straight from the row since a tariff
        # can't change what month number it is
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)

        # predict() returns a list even for one row - [0] pulls out the single number
        # It will use the 3 inputs (x), and then will pull out a single number using [0]
        pred = model.predict(x)[0]

        predictions.append(pred)   # store it for the results table
        history.append(pred)       # feed it forward for next month's lag, next time thru the loop history[-1] will be using this guess

    # attach the counterfactual, then compute how far actual reality fell from that counterfactual, as a percentage
    after['predicted'] = predictions
    after['gap_pct'] = (after['VALUE'] - after['predicted']) / after['predicted'] * 100

    # collapse 17ish months of gaps into a single average number
    return after['gap_pct'].mean()


results = []
# iterate thru the hashmap of all commodities, and call the function to get back each of its avg gap % and put them the commodity alongside its gap
# % into a hashmap, and add it to a list
for code, info in commodities.items():
    avg_gap = run_commodity(code, info['cutoff'])
    results.append({
        'commodity': code,
        'group': info['group'],
        'rate': info['rate'],
        'avg_gap_pct': round(avg_gap, 1),
    })

summary = pd.DataFrame(results).sort_values('avg_gap_pct')
# prints the whole table without row-number clutter
print(summary.to_string(index=False))

# save it as a real file
summary.to_csv('hs_results.csv', index=False)
print()
print('Saved hs_results.csv')

print()
print('Average gap by group:')
print(summary.groupby('group')['avg_gap_pct'].mean().round(1).to_string())