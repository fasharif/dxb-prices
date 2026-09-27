# Data: sources, terms of use and cleaning

Source: Dubai Land Department (DLD) open data. This project is not affiliated
with, or endorsed by, DLD, Digital Dubai or the Government of Dubai.

## Sources

### DLD open data page (used for training)

- Page: <https://dubailand.gov.ae/en/open-data/real-estate-data/>, section
  "Transactions", button "Download as CSV".
- The button posts a JSON filter to
  `https://gateway.dubailand.gov.ae/open-data/transactions/export/csv` and
  receives a UTF-8 CSV (with a byte-order mark). `dxb-prices download` fetches
  the same CSV export a visitor gets from that button, with all groups, usages
  and property types selected, one calendar month per request, three seconds
  apart. The request body is in `dxb_prices.download.export_body`. The endpoint
  is undocumented and may change.
- Requests carry only a User-Agent that names this project. They do not send
  the DLD page's Origin or Referer headers; a one-day request without them on
  26 September 2026 (UTC) returned the normal CSV.
- Neither `dubailand.gov.ae` nor `gateway.dubailand.gov.ae` publishes a
  `robots.txt`: both answered HTTP 404 on 27 September 2026 (UTC).
- The page's date picker only offers dates in the current calendar year, so the
  downloader only asks for months of the current year. The January to August
  2026 snapshot used for the published results therefore cannot be downloaded
  again with this tool after 31 December 2026; the file sizes and SHA-256
  checksums in [reports/metrics.md](../reports/metrics.md) identify it.
- Each download is cached in `data/raw/dld/transactions_YYYY-MM.csv`, with
  `data/raw/dld/manifest.json` recording the source URL, period, download time,
  size in bytes, SHA-256 and row count. A month downloaded less than seven days
  after it ended is marked provisional and fetched again on the next run, since
  late registrations can still arrive.
- Checked on 25 September 2026 (UTC); the published run's files were
  downloaded the same day.

### Dubai Pulse `dld_transactions-open` (checked, not used for training)

- Every `dubaipulse.gov.ae` URL, including the dataset page and the old
  `transactions.csv` download link, now returns a permanent redirect (HTTP 301)
  to <https://data.dubai>.
- On data.dubai the dataset is "Real Estate Transactions"
  (<https://data.dubai/en/l/470061>), issued by DLD, tagged Open. The page says
  the full table is available through an API key that is granted on request and
  subject to approval.
- `dxb-prices download --source dubai-data-sample` saves the public sample to
  `data/raw/dubai_data/`, and `python scripts/data_audit.py --dubai-data-sample`
  checks it. On 26 September 2026 (UTC) the sample held 7,000 rows from all
  years, 556 of them dated 2026; it is refreshed, so the counts change. Its
  column layout (lower-case names such as `actual_worth`, `procedure_area`,
  `meter_sale_price`) was read by `dxb_prices.schema` with the date, price,
  size, community and off-plan columns all filled and the three price-per-area
  columns dropped, and the cleaning rules kept 2,030 apartment sales. It is too
  small, and spread over too many years, to train on.

## Terms of use

### DLD website terms and conditions

<https://dubailand.gov.ae/en/terms-conditions/>, read on 25 September 2026 (UTC).
In summary:

- Material on the site may be viewed, downloaded or printed for personal,
  non-commercial use only.
- It may not be sold, modified, reproduced, displayed, publicly performed or
  distributed beyond that without a written agreement with DLD.
- Copyright and other notices must be kept on any copy.
- DLD owns or licenses the intellectual property in the site's content and in
  its compilation.

### Dubai Data Open Data Licence

Linked from the data.dubai dataset page as "Open Data License - English (4).pdf"
(SHA-256 `f380f9fe755c9f298bba40f94bf2638abd06de851dc813a0c35ad8a9adacbadc`),
read on 25 September 2026 (UTC). In summary:

- A worldwide, royalty-free, non-exclusive licence to use the information for
  commercial and non-commercial purposes (clause 2.1).
- Works that use the information must name its originator, keep any copyright
  notice and link to the source where practical (3.2.1), and must not claim or
  imply any association with the publisher, the Dubai Government or any other
  entity (3.2.2).
- Adapted versions of the information must be made available under the same
  licence (3.3); stand-alone work products need not be (3.3.1).
- The original information may not be sold (3.6); personal data may not be
  disclosed (2.3.2); everything is provided "as is" (4.1).

### What this repository does as a result

- No raw data, cleaned data, per-community or per-project tables, and no
  trained model are committed. `data/`, `artifacts/` and `mlruns/` are
  git-ignored. Everyone downloads their own copy.
- The drift report's HTML (`reports/drift/`) is also ignored, because it embeds
  distributions of the source data.
- Committed results in `reports/` are this project's own evaluation output:
  accuracy metrics, error analysis (row counts and errors per segment), SHAP
  importance and drift test statistics. They contain no transaction records
  and no price statistics of the data, such as a median price per community,
  project or segment.
- The README shows one illustrative API response: the model's estimate, its
  range and factors, and the model's starting price per square metre, which
  belongs to the fitted model rather than to any community. The response's
  community median price per square metre is left out of the README for the
  reason above; `scripts/readme_sample.py` drops it.
- The docs quote a few single figures about the data where a cleaning rule or
  a design decision rests on them, such as the highest price per square metre
  that passes the cleaning rules and the row counts of each cleaning step.
- A running API serves values derived from the data it was trained on: each
  community's training median price per square metre, and the number of
  training sales per community and per project (`/communities`, `/projects`).
  They come from the model directory, which is not committed.
- The test fixture (`tests/fixtures/transactions_synthetic.csv`) is generated
  by `scripts/make_fixture.py` from fixed rules and a fixed seed. It uses
  public place names but no DLD values.
- This is a personal, non-commercial portfolio project. The README and the
  Streamlit page state the source and that there is no affiliation with DLD.

## Columns

The export returns exactly the columns requested (see
`dxb_prices.schema.EXPORT_COLUMNS`). The model uses:

| Canonical name | Export column | Use |
|---|---|---|
| `transaction_date`, `month` | `INSTANCE_DATE` | month of sale; temporal split |
| `price_aed` | `TRANS_VALUE` | target (as log price per sqm) |
| `area_sqm` | `ACTUAL_AREA`, else `PROCEDURE_AREA` | size |
| `community` | `AREA_EN` with `AREA_AR` | DLD area, normalised |
| `project` | `PROJECT_EN` | project or building |
| `rooms` | `ROOMS_EN` | studio, 1-4, 5+, penthouse, unknown |
| `is_off_plan` | `IS_OFFPLAN_EN` | off-plan or ready |
| `is_freehold` | `IS_FREE_HOLD_EN` | freehold or not |
| `sub_type` | `PROP_SB_TYPE_EN` | Flat or Hotel Apartment |
| `nearest_metro`, `nearest_mall`, `nearest_landmark` | `NEAREST_*_EN` | often empty |

Not used: `PARKING` (mixes counts such as `1` with bay identifiers such as
`G-16`), `MASTER_PROJECT_*` (almost always empty), `TOTAL_BUYER` and
`TOTAL_SELLER`.

## Cleaning rules

Applied in this order by `dxb_prices.clean.clean`; each step's row counts for
the published run are in [reports/metrics.md](../reports/metrics.md#cleaning).
The thresholds live in `dxb_prices.config.CleaningRules`.

1. **Parse.** Drop rows without a transaction id, date, price or size.
2. **Exact duplicates.** Keep one copy of identical rows.
3. **Sales only.** Keep the Sales group; drop mortgages and gifts. This comes
   before any check on repeated transaction numbers, because lease-to-own
   contracts are listed under one number in both the Sales and the Mortgage
   group.
4. **Location duplicates.** The export sometimes lists one sale twice, with
   the same unit, price and date but a different nearest metro (or mall, or
   landmark). One row is kept: the one whose labels come first alphabetically.
5. **Multi-unit deals.** A transaction number that still appears on several
   rows records one value against several units; the value cannot be split, so
   all its rows are dropped. In the 2026 export this only happens in mortgage
   records (portfolio mortgages), which step 3 has already removed, so the rule
   removes nothing there; it stays as a guard for other snapshots.
6. **Market sales.** Keep sale procedures (Sale, Sell - Pre registration,
   Delayed Sell, Sale On Payment Plan and their non-freehold "Development"
   equivalents); drop lease-to-own contracts.
7. **Apartments.** Units with residential usage and sub-type Flat or Hotel
   Apartment; drops villas, land, buildings, offices, shops and hotel rooms.
8. **Size bounds.** 18 to 3,000 sqm.
9. **Price bounds.** AED 100,000 to AED 500 million.
10. **Price per sqm bounds.** 2,500 to 250,000 AED per sqm. Below about 230 AED
    per sq ft the records are part-share transfers or keying errors; the most
    expensive genuine branded penthouses in the 2026 data are around 184,000 AED
    per sqm.
11. **Names.** English and Arabic normalisation (below).

Training rows alone then lose the 0.5% tails of price per square metre within
each community that has at least 200 training rows. Smaller communities are not
trimmed: their percentiles are unreliable, and the overall percentiles would
remove whole luxury communities (every sale in Jumeirah Second, Jumeira Bay and
Island 2 lies above the overall 99.5th percentile; `scripts/data_audit.py`
lists them). Validation and test rows are not trimmed.

## Name normalisation

- English: Unicode NFKC, case-folded, punctuation and repeated spaces removed,
  `&` read as "and". Names in capitals are shown in title case.
- Arabic: diacritics and tatweel removed; أ إ آ ٱ become ا, ى becomes ي, ة
  becomes ه, ؤ becomes و, ئ becomes ي; Arabic-Indic digits become ASCII.
- Communities: English spellings that share an Arabic name are merged and
  labelled with the most frequent spelling. The mapping is built from every
  downloaded month, including the validation and test months: it is a lookup of
  DLD's registry names, not a feature learned from prices, so it carries no
  price information. A retrain on newer months can change a community's label. In the January to August 2026
  export, one Arabic name carries two different English names ("DUBAI MARITIME
  CITY" and "Madinat Dubai Almelaheyah"), several names differ only in capitals
  ("BUSINESS BAY" and "Business Bay"), and no English name carries two
  different Arabic names.
- `python scripts/data_audit.py` re-checks these facts on any download, along
  with the freehold split per procedure, the transaction numbers shared between
  groups, how often optional columns are empty, and the most expensive sales per
  square metre that pass the cleaning rules.
- Projects: English key only, because different projects can share generic
  Arabic names. A project is identified by its community and its name
  together: in the January to July 2026 export, four names belong to
  buildings in two communities each (Botanica in Dubai Marina and in Jumeirah
  Village Circle, Imperial Residence, Indigo Tower and Living Legends Phase 7;
  `python scripts/data_audit.py` lists them).
