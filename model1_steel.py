import pandas as pd

napcs_col = 'North American Product Classification System (NAPCS)'
steel = 'Potash [161]'

df = pd.read_csv('filtered_raw.csv')
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

# Keep only semi-finished steel
df = df[df[napcs_col] == steel]

# Sort oldest to newest so lag features line up correctly
df = df.sort_values('REF_DATE').reset_index(drop=True)

# Feature 1: last month's exports (short-term level)
df['lag_1'] = df['VALUE'].shift(1)

# Feature 2: exports 12 months ago (yearly seasonality)
df['lag_12'] = df['VALUE'].shift(12)

# Feature 3: a simple counter for the overall trend
df['time_index'] = range(len(df))

# First 12 rows have blank lags (nothing exists that far back) - drop them
df = df.dropna(subset=['lag_1', 'lag_12'])

# Split: everything before the tariff is what the model learns "normal" from
cutoff = pd.Timestamp('2025-02-01')
train = df[df['REF_DATE'] < cutoff]

print(f'Steel months after building features: {len(df)}')
print(f'Pre-tariff (training) months: {len(train)}')
print(f'Training range: {train["REF_DATE"].min().date()} to {train["REF_DATE"].max().date()}')
print()
print('First few training rows:')
print(train[['REF_DATE', 'VALUE', 'lag_1', 'lag_12', 'time_index']].head())

# Train a simple regression on the pre-tariff data
from sklearn.linear_model import LinearRegression

features = ['lag_1', 'lag_12', 'time_index']
# list of names inside [...] picks those columns
X_train = train[features]
y_train = train['VALUE']

model = LinearRegression()
# train the model using linear regression
model.fit(X_train, y_train)

print()
print('Model coefficients:')
for name, coef in zip(features, model.coef_):
    print(f'  {name}: {coef:.3f}')
print(f'  intercept: {model.intercept_:.3f}')

# Forecast the counterfactual: what exports would have been with no tariff.
# Feed the model its own predictions, not actuals, or the real collapse
# would leak in through lag_1 and cancel out the effect we're measuring.

# represents the tariff period rows
after = df[df['REF_DATE'] >= cutoff].copy()
 
# Running history of values the forecast uses for its lags.
# Starts as pre-tariff actuals, then grows with each prediction, we append it to the list
history = list(train['VALUE'])
 
predictions = []
for _, row in after.iterrows():
    lag_1 = history[-1]        # previous month (predicted, once we're past the first step)
    lag_12 = history[-12]      # same month last year
    time_index = row['time_index']
 
    # builds one row table holding those 3 numbers
    x = pd.DataFrame([[lag_1, lag_12, time_index]], columns=features)
    # the predict function always returns a list so [0] pulls out the single number
    pred = model.predict(x)[0]
 
    predictions.append(pred)
    history.append(pred)       # feed this prediction forward, appending it into the history list
    

# attaches the predictions as a new column 
after['predicted'] = predictions

# Calculate the gap between the predicted vs the actual/theoretical export value
# Formula: gap % = (actual − predicted) / predicted × 100
after['gap_pct'] = (after['VALUE'] - after['predicted']) / after['predicted'] * 100
 
print()
print('Counterfactual vs actual:')
print(after[['REF_DATE', 'VALUE', 'predicted', 'gap_pct']].to_string(index=False))
 
print()
print(f'Average gap over tariff period: {after["gap_pct"].mean():.1f}%')