# Tariff Impact Predictor

Measures how much the 2025-26 US tariffs actually cost specific Canadian export products, separating the tariff's effect from trends that were already underway.

**Live app: https://tariff-impact-predictor.streamlit.app/**

## The problem

The obvious way to measure tariff damage is to compare exports before and after. It doesn't work.

Canadian lumber exports fell sharply after tariffs hit, but lumber had already been declining since 2021 for reasons unrelated to trade policy. A raw before/after comparison blames the tariff for a decline that was already happening.

The actual question is how much of the drop is the tariff, and how much would have happened anyway.

## Approach

For each product, train a model on pre-tariff export data to forecast what exports would have been if nothing changed. Compare that forecast to what actually happened. The gap is the estimated tariff impact.

Then measure how accurate that forecast is, so the gap can be trusted rather than just eyeballed.

## Data

Statistics Canada's CIMT database, HS8-level monthly exports to the US, 2015 to 2026 (~12.5M rows). Filtered to products covered by Section 232 tariffs (steel, aluminum, lumber, furniture) plus four confirmed tariff-exempt controls (fish, crude oil, natural gas, potash).

Steel and aluminum are matched at 4-digit HS headings since whole headings are covered by the tariff. Lumber and furniture are matched at 6 digits, because their tariffs cover only specific subcodes and Canadian export codes only align with US import codes to 6 digits. Kitchen cabinets and vehicles were excluded: their codes mix tariffed and untariffed goods, or the tariff depends on end-use that trade data can't determine.

An earlier version of this analysis ran on broad commodity categories (StatCan NAPCS, 9 commodities). At that granularity only 2 of 9 effects were distinguishable from noise, because the coarse categories averaged badly-hit products together with barely-affected ones. Rebuilding at HS product level surfaced 22 of 25.

## Model

A separate `LinearRegression` per product, with three features:

- `lag_1`, last month's exports
- `lag_12`, the same month a year earlier, capturing seasonality
- `time_index`, a counter capturing long-run trend

Trained only on pre-tariff months. Cutoffs differ by product: February 2025 for steel and aluminum (tariff effective March 12), September 2025 for lumber and furniture (effective October 14).

The model feeds its own predictions forward rather than using real values. This is necessary because `lag_1` carries a coefficient around 0.83, so feeding real tariff-period values would leak the collapse back into the "no tariff" forecast and the measured gap would collapse toward zero.

Impact measure:
```
gap % = (actual - predicted) / predicted * 100
```

## Validation

**Holdout accuracy.** Trained on pre-2024 data, tested against all of 2024, a year with no tariffs, so any error is pure model error. Each product gets its own accuracy score (MAPE) that its reported gap is weighed against.

**Naive baselines.** Compared against two model-free rules, both applied recursively for a fair comparison. Median MAPE: model 17.4%, naive-last-year 17.6%, naive-last-month 24.1%. The model clearly beats pure persistence and slightly edges a seasonal-naive forecast.

**Control group.** The same pipeline run on tariff-exempt products. If those showed large gaps, the method would be manufacturing effects.

**Failure diagnosis.** Six products failed the accuracy check badly (67% to 820% error) and were excluded, mostly small, lumpy trade series the model couldn't hold.

## Model selection

A `RandomForestRegressor` was tested through identical code. It performs comparably on typical products (median MAPE 16.9% vs 17.4%) and is more robust on volatile series, since it cannot extrapolate beyond its training range.

That same property makes it unsuitable here. The counterfactual requires projecting a trend forward, and a model that structurally can't predict above its historical maximum would systematically under-predict counterfactuals for growing products, shrinking the measured gaps. `LinearRegression` was kept for that reason, not because it scored better.

## Results

| | Value |
|---|---|
| Treated products, average gap | -38.4% |
| Control products, average gap | -4.7% |
| Products with effects outside control noise | 22 of 25 |
| Typical forecast accuracy | 8-27% MAPE |

At an identical 50% tariff rate, individual products ranged from **+6.5% to -75.9%**.

## Relationship to published work

RBC Economics published essentially the same substitutability conclusion using HS6-level exposure data. Bank of Canada performs counterfactual comparison against its own pre-trade-war forecast, but at an aggregate level rather than per product. This analysis reaches the same conclusion independently, and adds a per-product counterfactual model with a documented accuracy score and control-group test for each individual product.

## Running it locally

```
pip install -r requirements.txt
streamlit run app.py
```

To rebuild from raw data, run in order:

```
load_hs_data.py    combines CIMT yearly files, filters to US + tariff codes
prep_hs_data.py    reshapes to one row per (month, product), applies volume cutoff
model1_hs.py       trains and forecasts per product, writes hs_results.csv
validate_hs.py     holdout accuracy check per product
baseline.py        naive baseline comparison
compare_models.py  LinearRegression vs RandomForest
```

Raw CIMT files aren't included in this repo. Download the HS8 export files from Statistics Canada and place them in `cimt_exports/`.

## Limitations

- Correlation, not causation. Confounds can't be fully ruled out.
- Controls are raw resources; treated products are mostly manufactured goods, so they aren't perfect economic twins.
- Crude oil, a control, showed a -5.8% average with 82% of months negative, likely its own price-driven decline. The control band is noisier than it first appears.
- The steel and aluminum tariff regime changed twice mid-window (25% to 50% to a 15-50% metal-content basis) and is treated here as one period.
- Front-running ahead of tariff deadlines likely inflates some pre-tariff months.
- Recursive forecasting compounds error over a 12-month horizon, which is the known cause of the six excluded products.
- Data begins in 2015, giving less training history than the earlier category-level version.

## Sources

- Statistics Canada, Canadian International Merchandise Trade (CIMT)
- US Section 232 proclamations and CBP tariff guidance
- Trade Commissioner Service, "Answers to common questions about U.S. tariffs"
- RBC Economics; Bank of Canada Monetary Policy Reports
