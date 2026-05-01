# Road Accidents in France 2023 - Exploration Report 1

## Source

BAAC dataset (Bulletin d'Analyse des Accidents Corporels de la Circulation), produced by the French Ministry of the Interior and published as open data at data.gouv.fr. The 2023 release covers all bodily-injury road accidents reported by police forces in metropolitan France and the overseas departments. Records are anonymised at the user level.

## Schema (4 tables, joined on `Num_Acc`)

| Table | Rows | Cols | Grain | Purpose |
|-------|------|------|-------|---------|
| `caract-2023.csv` | 54,822 | 15 | one row per accident | date, time, light, weather, location, department, commune |
| `lieux-2023.csv` | 70,860 | 18 | one row per accident-location segment | road category, lanes, surface, infra, speed limit `vma` |
| `vehicules-2023.csv` | 93,585 | 11 | one row per vehicle | category, manoeuvre, obstacle, motorisation |
| `usagers-2023.csv` | 125,789 | 16 | one row per person | role, sex, year of birth, safety equipment, target `grav` |

`lieux` has more rows than `caract` because some accidents span multiple road segments. `vehicules` and `usagers` link via `Num_Acc + id_vehicule + num_veh`. A 1,000-row sample joined cleanly to a 1,293 x 55 frame, confirming the keys behave as expected.

## Missing values

- `caract`: `adr` (free-text address) missing in 1,389 rows (~2.5%). Latitude and longitude are typed as text (comma decimal separator) and need parsing rather than imputation.
- `lieux`: `lartpc` (median strip width) missing in 70,829 of 70,860 rows (effectively unusable). `v2` missing in 64,976 rows. `voie` missing in 12,747 rows.
- `usagers`: `an_nais` (year of birth) missing in 2,598 rows (~2.1%).
- `vehicules`: `occutc` (public transport occupants) missing in 92,747 rows (only relevant for buses/trams).

The bulk of structured fields are coded integers with no nulls. Most "missing" values in BAAC are encoded as `-1` rather than `NaN`, which will need a separate sweep before modelling.

## Target distribution: `grav`

Code book: 1 = unscathed, 2 = killed, 3 = hospitalised, 4 = light injury, -1 = unknown.

| grav | label | count | pct |
|------|-------|-------|-----|
| 1 | unscathed | 53,399 | 42.45% |
| 4 | light injury | 49,603 | 39.43% |
| 3 | hospitalised | 19,271 | 15.32% |
| 2 | killed | 3,398 | 2.70% |
| -1 | unknown | 118 | 0.09% |

Class imbalance ratio (max/min, excluding the 118 unknowns): **15.7x** between `unscathed` and `killed`. If the unknown bucket is included as a real class, the ratio jumps to **452.5x**. Practical takeaway: drop or impute `-1`, then train with class weights or a stratified resampler. A binary "severe vs not" target (grav in {2,3}) gives ~18% positives, which is a more tractable framing for a baseline.

## Seasonality and day-of-week

Accidents per month:

| Month | Count |
|-------|-------|
| Jan | 4,053 |
| Feb | 3,682 |
| Mar | 3,998 |
| Apr | 4,162 |
| May | 4,767 |
| Jun | 5,452 |
| Jul | 4,754 |
| Aug | 4,121 |
| Sep | 5,161 |
| Oct | 5,389 |
| Nov | 4,833 |
| Dec | 4,450 |

Volume peaks in June and again in September-October, with a soft trough in February and August (holiday effect). Severity (share of users killed or hospitalised) peaks in July and August at 20-21%, versus 16-17% in winter; summer driving is more lethal even though raw volume is mid-range.

Accidents per day-of-week: Friday is the highest-volume day (8,873), Sunday the lowest (6,701). The Mon-Thu band sits in a tight 7.3-8.1k range.

## Key observations

1. The four-table star schema joins cleanly on `Num_Acc`. The per-person table `usagers` is the right grain for a severity classifier; `caract`, `lieux`, and `vehicules` join in as feature sources.
2. `grav` is materially imbalanced. Modelling needs class weights, ordinal regression, or a binary reframe (severe vs not).
3. BAAC encodes missing as `-1`, not `NaN`. Standard `df.isna()` undercounts true missingness; a dedicated audit by column is needed before feature engineering.
4. `lartpc`, `v2`, and `occutc` are too sparse to use directly; drop or treat as binary "present/absent".
5. Latitude and longitude in `caract` are stored as strings with comma decimals. Convert with `str.replace(',', '.').astype(float)` before any spatial work.
6. Seasonality is present but mild for raw counts; severity seasonality (summer spike) is the more interesting signal.
7. Friday is the highest-risk weekday by volume; Sunday volume is lowest but warrants a severity check (open question for the next iteration).

## Next steps

- Build a feature engineering pass that decodes BAAC integer codes (lum, atm, col, catr, catv, secu1...) into readable categories.
- Sweep `-1` codes per column and decide impute vs drop policy.
- Convert lat/long to floats and bin by department for spatial features.
- Fit a baseline logistic / gradient boosted model on a binary severe-target with stratified split and class weights.
