# Model card: dxb-prices apartment sale price model

## Model details

- **What it does:** estimates the sale price of one apartment in Dubai from its
  community, size, rooms, off-plan or ready status, date, and (optionally) its
  project. Each estimate comes with an 80% range and the five factors that
  moved it most (SHAP values).
- **Type:** LightGBM gradient-boosted trees predicting the natural log of price
  per square metre; the estimate is `exp(prediction) * size`.
- **Features:** size, off-plan flag, freehold flag, month of sale, community,
  project, rooms, flat or hotel apartment, nearest metro, mall and landmark
  (as recorded by DLD). Details in [data.md](data.md#columns).
- **Baseline:** the community's median price per square metre over the
  training months, times the size. Communities without training sales use the
  median of all training sales.
- **Author:** Farah Sharif, as a portfolio project. Not affiliated with DLD.
- **Licence:** code under MIT. The data is not redistributed; see
  [data.md](data.md#terms-of-use).

## Intended use

- Exploring how Dubai apartment prices relate to location, size and status.
- A starting point for a price discussion, alongside real comparables.
- Demonstrating a leak-free temporal evaluation and per-estimate explanations.

## Not intended for

- Mortgage, insurance, legal or tax valuations. The model has not been
  validated for any of these and gives no guarantee.
- Villas, townhouses, land, offices, shops or whole buildings. It only saw
  apartments (flats and hotel apartments).
- Dates well beyond the training period. It does not forecast market movement;
  a date past the newest training month is treated like that month, and the
  API warns when the gap exceeds three months.

## Training and evaluation data

- DLD open data transaction export, sales registered in the current calendar
  year (the DLD page offers no earlier dates). Cleaning rules are in
  [data.md](data.md#cleaning-rules); the exact row counts of the published run
  are in [reports/metrics.md](../reports/metrics.md).
- Split by month: training months, then one validation month for model
  selection, then the newest month for testing. After selection the model is
  refitted on training and validation months; the test month is scored once.
- Validation and test rows are not trimmed; only the fixed validity rules
  apply to them, so luxury sales stay in the evaluation.

## Results

Generated from `reports/metrics.json` by `dxb-prices render-docs`.

<!-- results:start -->
Data: DLD open data transaction export, sales registered 2026-01-01 to 2026-08-31, downloaded 2026-09-25 (UTC) (147,845 raw rows in 8 monthly files).

| Period | Estimator | Rows | MdAPE | Within 10% | MAE (AED) |
|---|---|---:|---:|---:|---:|
| Validation (2026-07) | LightGBM | 11,274 | 5.4% | 67.9% | 249,522 |
| Validation (2026-07) | Community median baseline | 11,274 | 11.9% | 44.5% | 353,510 |
| Test (2026-08) | LightGBM | 9,480 | 5.3% | 70.4% | 220,774 |
| Test (2026-08) | Community median baseline | 9,480 | 10.9% | 46.9% | 344,661 |

The model beats the baseline on all three test metrics. The 80% estimate range covered 82.7% of test prices (80% nominal).

Test month by registration:

| Segment | Rows | Model MdAPE | Baseline MdAPE | Model within 10% | Baseline within 10% | In 80% range |
|---|---:|---:|---:|---:|---:|---:|
| off-plan | 7,093 | 4.3% | 9.2% | 76.5% | 53.1% | 87.5% |
| ready | 2,387 | 9.3% | 20.4% | 52.2% | 28.4% | 68.2% |

Test month by price band:

| Segment | Rows | Model MdAPE | Baseline MdAPE | Model within 10% | Baseline within 10% | In 80% range |
|---|---:|---:|---:|---:|---:|---:|
| under 1.0M AED | 4,295 | 4.5% | 10.7% | 74.8% | 47.7% | 85.5% |
| 1.0M-2.0M AED | 3,273 | 5.5% | 10.4% | 69.7% | 47.9% | 84.1% |
| 2.0M-5.0M AED | 1,647 | 6.9% | 11.5% | 63.0% | 45.7% | 75.9% |
| 5.0M AED and over | 265 | 9.1% | 16.6% | 52.8% | 27.9% | 61.1% |

Test month by community data:

| Segment | Rows | Model MdAPE | Baseline MdAPE | Model within 10% | Baseline within 10% | In 80% range |
|---|---:|---:|---:|---:|---:|---:|
| new (no training sales) | 6 | 81.1% | 81.7% | 0.0% | 0.0% | 0.0% |
| thin (1-49 training sales) | 149 | 2.9% | 2.9% | 81.9% | 79.2% | 85.9% |
| established (50+ training sales) | 9,325 | 5.4% | 10.9% | 70.3% | 46.4% | 82.7% |

Drift (2026-08 against the training months): flagged for area_sqm, log_price_per_sqm, community.

<!-- results:end -->

## Limitations

The tables above come from the published run; the statements below refer to
them rather than repeat their numbers.

- **Luxury sales.** The top price band has the largest errors and the lowest
  share of prices inside the 80% range. The most expensive branded penthouses
  reach about 184,000 AED per square metre (`python scripts/data_audit.py`),
  several times their community's median, and there are few of them. The range
  has the same relative width for every estimate, so it is too narrow here.
- **Ready units.** Resale of ready units is harder than off-plan: the median
  error is about twice as large and fewer prices fall inside the range. Condition,
  view and floor, which the data does not record, probably explain part of this.
- **New and thin communities.** Sales in communities with no training sales are
  rare but estimated badly; the API refuses a community it has never seen rather
  than guess. The "thin" segment is small, so its figures are noisy. A project
  with few training sales is grouped with other rare projects, so its estimate
  rests on the community and the unit's own features.
- **Short history.** The DLD page only offers the current calendar year, so the
  model sees at most eleven months. Seasonal effects and turning points cannot
  be learned, and early in each year there is too little data to retrain.
- **No forecasting.** A date after the training period is treated like the last
  training month; the API warns when the gap is more than three months.
- **Registered values.** DLD records the registered value of each sale, which
  may leave out incentives such as payment plans or waived fees.
- **Categorical landmarks.** Nearest metro, mall and landmark are empty for
  roughly 40% to 55% of the sales in the 2026 export (`scripts/data_audit.py`)
  and cannot express distance.
- **Dependence on the source.** If DLD changes the export format or its
  categories, the downloader or the cleaning rules need updating.

## Ethical considerations

The data describes transactions, not people: the export has no names or
identifiers of buyers or sellers. Price estimates can still influence
decisions about homes; the estimate is presented with a range, the factors
behind it and a warning where data is thin, and the interface states that it is
not a valuation.

## Monitoring

`.github/workflows/retrain.yml` retrains monthly on the newest data, runs an
Evidently drift report comparing the newest month with the training months, and
uploads the metrics report for a person to review. It does not deploy. In the
published run the test month already differed from the training months in size,
price per square metre and community mix (the drift line above), which is a
reason to retrain monthly rather than less often.
