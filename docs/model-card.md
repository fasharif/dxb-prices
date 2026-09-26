# Model card: dxb-prices apartment sale price models

## Model details

- **What it does:** estimates the sale price of one apartment in Dubai from its
  community, size, rooms, off-plan or ready status, date and, optionally, its
  project (building). Each estimate comes with an 80% range, the five factors
  that moved it most (SHAP values) and the combined effect of the others.
- **Two models**, both LightGBM gradient-boosted trees predicting the natural
  log of price per square metre; the estimate is `exp(prediction) * size`.
  - The **full model** answers when the project has training sales. Features:
    size, off-plan flag, freehold flag, month of sale, community, project,
    rooms, flat or hotel apartment, and DLD's nearest metro, mall and landmark.
    The API fills the location labels and a missing freehold flag from the
    project's training sales.
  - The **community-level model** answers when the project is not given or has
    no training sales. It uses the same features without the project and the
    location labels.
  Details in [data.md](data.md#columns) and [decisions.md](decisions.md) (14, 15).
- **Baselines:** the community's median price per square metre over the
  training months, times the size (the brief's baseline), and the project's
  median when it has at least five training sales, otherwise the community's.
  Communities without training sales use the median of all training sales.
- **Author:** Farah Sharif, as a portfolio project. Not affiliated with DLD.
- **Licence:** code under MIT. The data is not redistributed; see
  [data.md](data.md#terms-of-use).

## Intended use

- Exploring how Dubai apartment prices relate to location, building, size and
  status.
- A starting point for a price discussion, alongside real comparables.
- Demonstrating a leak-free temporal evaluation of what an API actually returns,
  with per-estimate explanations.

## Not intended for

- Mortgage, insurance, legal or tax valuations. The models have not been
  validated for any of these and give no guarantee.
- Villas, townhouses, land, offices, shops or whole buildings. They only saw
  apartments (flats and hotel apartments).
- Dates well beyond the training period. The models do not forecast market
  movement; a date past the newest training month is treated like that month,
  and the API warns when the gap exceeds three months.

## Training and evaluation data

- DLD open data transaction export, sales registered in the current calendar
  year (the DLD page offers no earlier dates). Cleaning rules are in
  [data.md](data.md#cleaning-rules); the exact row counts of the published run
  are in [reports/metrics.md](../reports/metrics.md).
- Split by month: training months, then one validation month for model
  selection, early stopping and the 80% ranges, then the newest month for
  testing. After selection both models are refitted on training and validation
  months; the test month is scored once.
- Each test sale is scored as the request a user would send, with its project
  and without it, through the same code the API uses. A score with DLD's
  recorded location labels, which users cannot supply, is kept as a reference.
- Validation and test rows are not trimmed; only the fixed validity rules
  apply to them, so luxury sales stay in the evaluation.
- A rolling-origin backtest scores each month from the fourth onwards with
  models trained only on the months before it.

## Results

Generated from `reports/metrics.json` by `dxb-prices render-docs`.

<!-- results:start -->
Data: DLD open data transaction export, sales registered 2026-01-01 to 2026-08-31, downloaded 2026-09-25 (UTC) (147,845 raw rows in 8 monthly files).

Test month 2026-08 (9,480 sales, scored once, after the models were refitted on 2026-01 to 2026-07), each sale scored as the API would answer it:

| Estimator | Rows | MdAPE | Within 10% | MAE (AED) | Median error | In 80% range |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, project given | 9,480 | 5.6% | 69.8% | 209,340 | -0.1% | 79.1% |
| LightGBM, no project (community-level model) | 9,480 | 6.5% | 63.7% | 238,899 | -0.1% | 81.1% |
| Baseline: project median | 9,480 | 7.3% | 61.2% | 241,895 | +1.0% | n/a |
| Baseline: community median | 9,480 | 10.8% | 46.9% | 327,789 | +0.8% | n/a |
| LightGBM with DLD's recorded location labels (reference) | 9,480 | 5.4% | 70.8% | 198,400 | -0.1% | n/a |

*Project given*: the request a user sends (community, project, size, rooms, off-plan or ready, date); the nearest metro, mall and landmark and the freehold flag are filled in from the training data, as the API does. Sales whose project had no training sales (1,193 of 9,480) get the community-level model, as they would from the API. *No project*: the same request without the project, answered by the community-level model, which was trained without the project and the location labels. *Project median*: the project's training median price per sqm when it has at least 5 training sales, otherwise the community's, times the size. *Community median*: the community's training median price per sqm times the size (the baseline the brief asks for). *Recorded location labels*: the full model given DLD's own nearest metro, mall, landmark and freehold flag for each sale, which an API user cannot supply. Median error below zero means estimates run low.

With the project, LightGBM beats both the community-median and the project-median baseline on all three test metrics. Without the project, the community-level model beats the community-median baseline on all three test metrics. Most of the gain over the community median comes from knowing the building: the project median alone moves MdAPE from 10.8% to 7.3%, and the model with the project reaches 5.6%. The 80% range contained 79.1% of test prices with the project and 81.1% without it (80% nominal).

Rolling-origin backtest over 5 test months (2026-04 to 2026-08, default settings, each month scored by models trained only on earlier months): MdAPE 5.4% to 6.9% with the project and 7.0% to 8.2% without it, against 7.0% to 7.6% for the project median and 10.8% to 12.3% for the community median.

Test month by registration:

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| off-plan | 7,093 | 4.5% | 5.3% | 6.1% | 9.2% | 76.5% | 84.4% |
| ready | 2,387 | 10.1% | 11.2% | 11.5% | 20.4% | 49.8% | 63.4% |

Test month by price band:

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| under 1.0M AED | 4,295 | 4.7% | 5.6% | 6.3% | 10.7% | 73.3% | 83.2% |
| 1.0M-2.0M AED | 3,273 | 5.6% | 6.4% | 8.0% | 10.4% | 71.3% | 79.4% |
| 2.0M-5.0M AED | 1,647 | 7.6% | 8.6% | 8.1% | 11.6% | 60.6% | 70.7% |
| 5.0M AED and over | 265 | 9.4% | 12.4% | 10.6% | 16.2% | 51.3% | 60.0% |

Test month by community data:

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| thin (1-49 training sales) | 145 | 2.9% | 2.9% | 2.9% | 2.8% | 84.8% | 89.7% |
| established (50+ training sales) | 9,335 | 5.6% | 6.6% | 7.4% | 11.0% | 69.5% | 78.9% |

What the community-data segments contain:

| Segment | Rows | Communities | Largest single project | Median price (AED) |
|---|---:|---:|---:|---:|
| thin (1-49 training sales) | 145 | 18 | 70.3% of rows | 1,042,046 |
| established (50+ training sales) | 9,335 | 93 | 5.8% of rows | 1,093,950 |

The DLD page only offers dates in the current calendar year, so this 2026 snapshot cannot be downloaded again with this tool after 31 December 2026. The file sizes and SHA-256 checksums in [reports/metrics.md](../reports/metrics.md) identify it.

Drift (2026-08 against the training months): flagged for area_sqm, log_price_per_sqm, community.

<!-- results:end -->

## Limitations

The tables above come from the published run; the statements below refer to
them rather than repeat their numbers.

- **No project, less accuracy.** Without the project the community-level model
  answers. It beats both baselines but is clearly behind the full model, and
  its 80% range is wider. The API says which model answered and warns when the
  project is missing or unknown; the Streamlit page offers the community's
  projects in a list.
- **Luxury sales.** The top price band has the largest errors and the lowest
  share of prices inside the 80% range. The most expensive branded penthouses
  reach about 184,000 AED per square metre (`python scripts/data_audit.py`),
  several times their community's median, and there are few of them. The range
  has the same relative width for every estimate, so it is too narrow here.
  Penthouses and units with five or more rooms have only a handful of test
  sales, and for penthouses the project median did better than the model.
- **Ready units.** Resale of ready units is harder than off-plan: the median
  error is more than twice as large and fewer prices fall inside the range.
  Condition, view and floor, which the data does not record, probably explain
  part of this.
- **New and thin communities.** The API refuses a community it has no training
  sales for rather than guess. In the published run no test sale fell in such a
  community. The "thin" segment (fewer than 50 training sales) is dominated by
  one project (see the table above), so its low error says little about thin
  communities in general. A project with fewer than 40 training sales is grouped
  with the other rare projects in the full model; its own location labels still
  reach the model.
- **Short history, and a snapshot that expires.** The DLD page only offers the
  current calendar year, so the models see at most eleven months. Seasonal
  effects and turning points cannot be learned, from January to April there is
  too little data to retrain, and the published snapshot cannot be downloaded
  again with this tool after the year ends; its checksums identify it.
- **One test month.** The headline rests on a single month. The backtest shows
  how much the scores move between months (see the range above).
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
behind it and a warning where data is thin or the project is missing, and the
interface states that it is not a valuation.

## Monitoring

`.github/workflows/retrain.yml` runs on the 10th of each month, once the
previous month is final. It retrains on the newest data, runs an Evidently
drift report comparing the newest month with the training months, and uploads
the metrics report for a person to review. It does not deploy. In the published
run the test month already differed from the training months in size, price per
square metre and community mix (the drift line above), which is a reason to
retrain monthly rather than less often.
