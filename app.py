import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

features = ['lag_1', 'lag_12', 'time_index']

# Plain-English names for each HS code, so the dropdown isn't just numbers, hashmap
names = {
    '7208': 'Hot-rolled steel, flat, wide',
    '7209': 'Cold-rolled steel, flat, wide',
    '7210': 'Steel, flat, clad or coated',
    '7211': 'Steel, flat, narrow',
    '7212': 'Steel, flat, narrow, clad or coated',
    '7213': 'Steel bars and rods, hot-rolled coils',
    '7214': 'Steel bars and rods, forged or hot-rolled',
    '7215': 'Steel bars and rods, other',
    '7216': 'Steel angles, shapes and sections',
    '7217': 'Steel wire',
    '7225': 'Alloy steel, flat, wide',
    '7226': 'Alloy steel, flat, narrow',
    '7227': 'Alloy steel bars and rods, hot-rolled',
    '7228': 'Alloy steel bars and rods, other',
    '7304': 'Steel tubes and pipes, seamless',
    '7306': 'Steel tubes and pipes, other',
    '7601': 'Unwrought aluminum',
    '7604': 'Aluminum bars, rods and profiles',
    '7605': 'Aluminum wire',
    '7606': 'Aluminum plates and sheets',
    '7607': 'Aluminum foil',
    '440711': 'Pine lumber, sawn',
    '440713': 'Spruce-pine-fir lumber, sawn',
    '440714': 'Hem-fir lumber, sawn',
    '940161': 'Upholstered seats, wooden frame',
    '0302': 'Fish, fresh or chilled',
    '0303': 'Fish, frozen',
    '0304': 'Fish fillets',
    '0305': 'Fish, dried or smoked',
    '2709': 'Crude oil',
    '2711': 'Natural gas',
    '3104': 'Potash fertilizer',
}


# Cached so the CSVs are only read once, not on every interaction.
# these things are expensive work, so we don't want it to run/repeat every single time you touch a dropdown
@st.cache_data
def load_data():
    results = pd.read_csv('hs_results.csv', dtype={'commodity': str})
    monthly = pd.read_csv('hs_monthly.csv', dtype={'commodity': str})
    monthly['REF_DATE'] = pd.to_datetime(monthly['REF_DATE'])
    # returns them into a tuple, short for return (results, monthly)
    return results, monthly


# Rebuilds the counterfactual for one commodity so we can chart it.
# Same logic as model1_hs.py - only the average gap was saved to CSV,
# not the month-by-month predictions, so they're recomputed here.
def build_forecast(monthly, code, cutoff_str):
    cutoff = pd.Timestamp(cutoff_str)

    # keeps only this 1 commodity's rows, sorted oldest to newest so the lag features line up correctly
    df = monthly[monthly['commodity'] == code].sort_values('REF_DATE').reset_index(drop=True)
    
    # the 3 features
    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    
    # drop the first 12 rows, which don't have a full lag_12 yet
    df = df.dropna(subset=['lag_1', 'lag_12'])

    # slit into pre-tariff (train) and tariff period (after) using the commodity's own cutoff date
    train = df[df['REF_DATE'] < cutoff]
    after = df[df['REF_DATE'] >= cutoff].copy()

    # train on pre-tariff data only
    model = LinearRegression()
    model.fit(train[features], train['VALUE'])

    # recursive forecast, feed predictions forward, never actuals
    history = list(train['VALUE'])
    predictions = []
    for _, row in after.iterrows():
        # build this month's inputs, last two values from history
        x = pd.DataFrame([[history[-1], history[-12], row['time_index']]],
                         columns=features)
        pred = model.predict(x)[0]
        predictions.append(pred)
        history.append(pred)

    # attach the counterfactual as a new column, so the caller can plot actual (after['VALUE']) against predicted (after['predicted'])
    after['predicted'] = predictions
    return train, after


# Holdout accuracy on 2024 (a no-tariff year), so each result comes with
# a "how much should you trust this" number. Same as validate_hs.py.
def holdout_mape(monthly, code):
    train_end = pd.Timestamp('2024-01-01')
    test_end = pd.Timestamp('2025-01-01')

    df = monthly[monthly['commodity'] == code].sort_values('REF_DATE').reset_index(drop=True)
    df['lag_1'] = df['VALUE'].shift(1)
    df['lag_12'] = df['VALUE'].shift(12)
    df['time_index'] = range(len(df))
    df = df.dropna(subset=['lag_1', 'lag_12'])

    train = df[df['REF_DATE'] < train_end]
    test = df[(df['REF_DATE'] >= train_end) & (df['REF_DATE'] < test_end)]

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

# actual code begins here
st.set_page_config(page_title='Tariff Impact', layout='wide')
st.title('US Tariff Impact on Canadian Exports')
st.caption(
    'Each product gets its own model trained on pre-tariff data. The model '
    'forecasts what exports would have been with no tariff; the gap between '
    'that forecast and reality is the estimated impact.'
)

results, monthly = load_data()

# Cutoff dates per group - steel/aluminum tariffs hit Mar 2025,
# lumber and furniture Oct 2025.
cutoffs = {}
for _, row in results.iterrows():
    if row['commodity'].startswith('44') or row['commodity'] == '940161':
        cutoffs[row['commodity']] = '2025-09-01'
    else:
        cutoffs[row['commodity']] = '2025-02-01'

# Build dropdown labels as "code - name"
options = {}
for code in results['commodity']:
    label = f"{code} - {names.get(code, 'Unknown')}"
    options[label] = code

choice = st.selectbox('Select a product', sorted(options.keys()))
code = options[choice]

row = results[results['commodity'] == code].iloc[0]
mape = holdout_mape(monthly, code)

col1, col2, col3 = st.columns(3)
col1.metric('Measured gap', f"{row['avg_gap_pct']}%")
col2.metric('Tariff rate', f"{row['rate']}%" if row['rate'] > 0 else 'Exempt')
col3.metric('Model error (2024 holdout)', f"{mape:.1f}%")

# A gap is only meaningful if it's clearly bigger than the model's own
# typical error - otherwise it can't be told apart from noise.
if abs(row['avg_gap_pct']) > mape * 1.5:
    st.success(
        f"The gap ({row['avg_gap_pct']}%) is well outside this model's normal "
        f"error ({mape:.1f}%), so it likely reflects a real effect."
    )
else:
    st.warning(
        f"The gap ({row['avg_gap_pct']}%) is close to this model's normal "
        f"error ({mape:.1f}%), so it can't be clearly separated from noise."
    )

# rebuild the actual month-by-month forecast for this one commodity,
# so we have real numbers to plot (results.csv only stored the average gap)
train, after = build_forecast(monthly, code, cutoffs[code])

# --- the chart ---
fig, ax = plt.subplots(figsize=(11, 4.5))

# grey line: the real pre-tariff history (what "normal" looked like)
ax.plot(train['REF_DATE'], train['VALUE'], color='#4a4a4a', linewidth=1,
        label='Actual (pre-tariff)')

# red line: what REALLY happened during the tariff period
ax.plot(after['REF_DATE'], after['VALUE'], color='#c0392b', linewidth=2,
        label='Actual (tariff period)')

# blue dashed line: what the model GUESSED would have happened with no tariff
ax.plot(after['REF_DATE'], after['predicted'], color='#2a78d6', linewidth=2,
        linestyle='--', label='Counterfactual (no tariff)')

# vertical dotted line marking exactly where the tariff took effect
ax.axvline(pd.Timestamp(cutoffs[code]), color='gray', linestyle=':', linewidth=1)

ax.set_ylabel('Monthly exports to US ($)')
ax.legend()                # shows the little box explaining what each line/color means
ax.set_title(choice)       # use the pretty dropdown label (e.g. "7601 - Unwrought aluminum")
fig.tight_layout()         # tidies up spacing so labels don't get cut off
st.pyplot(fig)              # actually draws this matplotlib figure onto the Streamlit page

# --- the full results table, shown below the single-product view ---
st.subheader('All products')
st.caption(
    'Treated products were tariffed. Controls were confirmed exempt and act '
    'as a check that the method is not making up effects.'
)

# copy() so we don't accidentally modify the original results table
table = results.copy()

# add a readable name column, using the same names dict as the dropdown
table['product'] = table['commodity'].map(names)

# keep + reorder only the columns worth displaying
table = table[['commodity', 'product', 'group', 'rate', 'avg_gap_pct']]

# render as a scrollable, sortable table.
# use_container_width makes it fill the page width instead of a fixed size.
# hide_index removes pandas' default 0,1,2... row numbers, which aren't
# meaningful here
st.dataframe(table.sort_values('avg_gap_pct'), use_container_width=True,
             hide_index=True)

# the key headline comparison from the whole project: treated vs control
treated_avg = results[results['group'] == 'treated']['avg_gap_pct'].mean()
control_avg = results[results['group'] == 'control']['avg_gap_pct'].mean()

col1, col2 = st.columns(2)
col1.metric('Treated average', f'{treated_avg:.1f}%')
col2.metric('Control average', f'{control_avg:.1f}%')
