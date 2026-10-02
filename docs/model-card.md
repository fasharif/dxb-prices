# Model card: dxb-prices apartment sale price models

## Model details

- **What it does:** estimates the sale price of one apartment in Dubai from its
  community, size, rooms, off-plan or ready status, date and, optionally, its
  project (building). Each estimate comes with an 80% range, the five factors
  that moved it most (SHAP values) and the combined effect of the others.
- **Two models**, both LightGBM gradient-boosted trees predicting the natural
  log of price per square metre; the estimate is `exp(prediction) * size`.
  - The **full model** answers when the project has training sales in the
    requested community (a project is its community and its name together).
    Features: size, off-plan flag, freehold flag, month of sale, community,
    project, rooms, flat or hotel apartment, and DLD's nearest metro, mall and
    landmark. The API fills the location labels and a missing freehold flag
    from the project's training sales. A project with fewer than 40 training
    sales shares one level with the other rare projects, and the API says so.
  - The **community-level model** answers when the project is not given or has
    no training sales in that community. It uses the same features without the
    project and the location labels.
  Details in [data.md](data.md#columns) and [decisions.md](decisions.md) (14,
  15, 17).
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
- A simulated cold start refits both models and both baselines with each
  community in turn cut to 10 training sales, or none, and scores that
  community's test sales (decision 18).
- The error analysis breaks the test month down by off-plan or ready, price
  band (by the recorded price and by the estimate), how much data the
  community and the building have, and rooms.

## Results

Generated from `reports/metrics.json` by `dxb-prices render-docs`.

<!-- results:start -->
Data: DLD open data transaction export, sales registered 2026-01-01 to 2026-08-31, downloaded 2026-09-25 (UTC) (147,845 raw rows in 8 monthly files).

Test month 2026-08 (9,480 sales, scored once, after the models were refitted on 2026-01 to 2026-07), each sale scored as the API would answer it:

| Estimator | Rows | MdAPE | Within 10% | MAE (AED) | Median error | In 80% range |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, project given | 9,480 | 5.4% | 69.6% | 209,530 | -0.2% | 79.0% |
| LightGBM, no project (community-level model) | 9,480 | 6.5% | 63.7% | 238,899 | -0.1% | 81.1% |
| Baseline: project median | 9,480 | 7.3% | 61.2% | 241,895 | +1.0% | n/a |
| Baseline: community median | 9,480 | 10.8% | 46.9% | 327,789 | +0.8% | n/a |
| LightGBM with DLD's recorded location labels (reference) | 9,480 | 5.2% | 70.4% | 198,826 | -0.1% | n/a |

*Project given*: the request a user sends (community, project, size, rooms, off-plan or ready, date); the nearest metro, mall and landmark and the freehold flag are filled in from the training data, as the API does. Sales whose project had no training sales in its community, or none recorded (1,193 of 9,480), get the community-level model, as they would from the API. *No project*: the same request without the project, answered by the community-level model, which was trained without the project and the location labels. *Project median*: the project's training median price per sqm when it has at least 5 training sales, otherwise the community's, times the size. *Community median*: the community's training median price per sqm times the size (the baseline the brief asks for). *Recorded location labels*: the full model given DLD's own nearest metro, mall, landmark and freehold flag for each sale, which an API user cannot supply. Median error below zero means estimates run low. The API would have answered every test sale.

With the project, LightGBM beats both the community-median and the project-median baseline on all three test metrics. Without the project, the community-level model beats the community-median baseline on all three test metrics. Most of the gain over the community median comes from knowing the building: the project median alone moves MdAPE from 10.8% to 7.3%, and the model with the project reaches 5.4%. The 80% range contained 79.0% of test prices with the project and 81.1% without it (80% nominal).

Rolling-origin backtest over 5 test months (2026-04 to 2026-08, default settings, each month scored by models trained only on earlier months): MdAPE 5.2% to 6.9% with the project and 7.0% to 8.2% without it, against 7.0% to 7.6% for the project median and 10.8% to 12.3% for the community median.

Test month by registration:

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| off-plan | 7,093 | 4.4% | 5.3% | 6.1% | 9.2% | 76.2% | 84.2% |
| ready | 2,387 | 10.0% | 11.2% | 11.5% | 20.4% | 50.1% | 63.5% |

Test month by price band:

Bands by the recorded sale price. Banding by the outcome moves sales that sold above their estimate into higher bands and can make the top band look harder than it is, so the next table bands by the estimate instead.

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| under 1.0M AED | 4,295 | 4.7% | 5.6% | 6.3% | 10.7% | 73.3% | 83.1% |
| 1.0M-2.0M AED | 3,273 | 5.4% | 6.4% | 8.0% | 10.4% | 70.9% | 79.6% |
| 2.0M-5.0M AED | 1,647 | 7.6% | 8.6% | 8.1% | 11.6% | 60.4% | 70.4% |
| 5.0M AED and over | 265 | 8.9% | 12.4% | 10.6% | 16.2% | 50.6% | 58.5% |

Test month by estimated price band:

Bands by the estimate with the project, which is what a user sees before a sale.

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| under 1.0M AED | 4,141 | 4.4% | 5.3% | 5.9% | 9.9% | 75.1% | 84.7% |
| 1.0M-2.0M AED | 3,619 | 6.0% | 7.1% | 8.9% | 11.3% | 67.1% | 75.8% |
| 2.0M-5.0M AED | 1,458 | 6.7% | 7.9% | 7.9% | 11.6% | 63.9% | 74.6% |
| 5.0M AED and over | 262 | 9.6% | 12.5% | 10.8% | 17.8% | 50.0% | 58.0% |

Test month by community data:

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| thin (1-49 training sales) | 145 | 2.9% | 2.9% | 2.9% | 2.8% | 86.2% | 91.0% |
| established (50+ training sales) | 9,335 | 5.5% | 6.6% | 7.4% | 11.0% | 69.4% | 78.8% |

What the community data segments contain:

| Segment | Rows | Communities | Largest single project |
|---|---:|---:|---:|
| thin (1-49 training sales) | 145 | 18 | 70.3% of rows |
| established (50+ training sales) | 9,335 | 93 | 5.8% of rows |

Test month by project data:

How well the full model knows each sale's building. A project needs a minimum number of training sales in its community for a level of its own; rarer projects share one level, and a project with no training sales in its community (or none recorded) goes to the community-level model, so its two model columns are equal.

| Segment | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| own level (40+ training sales) | 5,374 | 3.7% | 4.7% | 6.1% | 9.3% | 83.5% | 90.5% |
| grouped as other (1-39 training sales) | 2,913 | 9.5% | 10.5% | 8.3% | 15.1% | 52.0% | 60.7% |
| new (no training sales in its community) | 639 | 9.9% | 9.9% | 8.7% | 8.7% | 51.3% | 74.2% |
| not recorded in the sale | 554 | 10.3% | 10.3% | 23.0% | 23.0% | 48.2% | 69.5% |

What the project data segments contain:

| Segment | Rows | Communities | Largest single project |
|---|---:|---:|---:|
| own level (40+ training sales) | 5,374 | 72 | 7.8% of rows |
| grouped as other (1-39 training sales) | 2,913 | 95 | 8.9% of rows |
| new (no training sales in its community) | 639 | 22 | 28.3% of rows |
| not recorded in the sale | 554 | 40 | 100.0% of rows |

Simulated cold start:

The 111 communities of the test month were split at random into 5 groups. For each group, both models and both baselines were refitted on the final training rows with that group's communities cut to the stated number of randomly chosen sales (every other community kept all of its sales), with the chosen settings and 80% ranges, and the group's test sales were scored as served. A request for a community without training sales gets a 404 from the API; the last row shows what the models would have answered.

| Training sales kept per community | Rows | MdAPE with project | MdAPE no project | Project median | Community median | Within 10% with project | In 80% range with project |
|---|---:|---:|---:|---:|---:|---:|---:|
| all (the published models) | 9,480 | 5.4% | 6.5% | 7.3% | 10.8% | 69.6% | 79.0% |
| 10 | 9,480 | 14.0% | 16.0% | 11.3% | 11.3% | 39.2% | 53.2% |
| 0: a new community (the API refuses these) | 9,480 | 18.8% | 18.8% | 16.4% | 16.4% | 31.5% | 46.8% |

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
- **The gain comes from buildings the model knows well.** For buildings with a
  level of their own (40 or more training sales) the error is far below the
  baselines'. For rare buildings grouped with the others, and for buildings
  with no training sales in their community, the project or community median
  did better than the model (the "project data" table). The API warns in both
  cases. Falling back to, or blending with, those medians where the model
  knows little is the obvious next step; it has to be chosen on the
  validation month, not on these test results.
- **Thin and new communities.** In the simulated cold start, a community cut
  to 10 training sales was estimated worse by the model than by the median of
  those 10 sales, and far worse than with all its data. With no training sales
  the models would do worse still, which supports the API's choice to refuse
  such a community (404) rather than guess; the API also warns when one has
  fewer than 50 training sales. The test month's own "thin" segment is mostly
  one project (see the table above), so the cold start is the better guide.
- **Luxury sales.** The top price band has the largest errors and the lowest
  share of prices inside the 80% range, whether sales are banded by their
  recorded price or by their estimate. The most expensive branded penthouses
  reach about 184,000 AED per square metre (`python scripts/data_audit.py`),
  and there are few of them. The range has the same relative width for every
  estimate, so it is too narrow here. Penthouses and units with five or more
  rooms have only a handful of test sales, and for penthouses the project
  median did better than the model.
- **Ready units.** Resale of ready units is harder than off-plan: the median
  error is more than twice as large and fewer prices fall inside the range.
  Condition, view and floor, which the data does not record, probably explain
  part of this.
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
retrain monthly rather than less often. GitHub switches off scheduled workflows
in a public repository after 60 days without activity; in a quiet repository
the workflow has to be re-enabled or run by hand (`workflow_dispatch`).
