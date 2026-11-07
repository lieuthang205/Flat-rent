# What Will This Flat Rent For? — Team Quantum

**Microburbs Hackathon (7 November 2026) • Task 4: Weekly Rent With and Without a Car Space (City of Sydney)**

---

## 1. What We Built

We built an end-to-end rental valuation and parking-premium estimation pipeline ([`rents_city_of_sydney_model.py`](rents_city_of_sydney_model.py)) that predicts the expected weekly asking rent **with a car space** (`rent_with_car_space`) and **without a car space** (`rent_without_car_space`) for all **65,777 flats** across the 33 suburbs of the **City of Sydney** (as at **31 July 2026**, projected to the August–September 2026 scoring window).

### Key Deliverables in This Repository
* **[`rents_quantum.csv`](rents_quantum.csv)** : Official submission file containing all **65,777 `gnaf_id` rows**, zero blanks, rounded to `$0.50` increments.
* **[`report.html`](report.html)**: Standalone interactive HTML report presenting key empirical findings, held-out validation metrics, $R^2$, temporal split design, data sources, and methodology.
* **[`rents_city_of_sydney_model.py`](rents_city_of_sydney_model.py)**: Reproducible training, validation, and inference script supporting both local execution and Microsoft Fabric Lakehouse / PySpark execution.

### Held-Out Validation Performance (Strict Out-of-Time Check <= 31 July 2026)

| Validation Split | Adverts Scored (N) | Median Absolute Error ($/wk) | Share Within $100/wk | MAE ($/wk) | RMSE ($/wk) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train <= 31 May 2026 -> Test Jun–Jul 2026** | 2,653 | **$42.50** | **80.96%** | $68.67 | $124.84 | **0.906** |
| **Train <= 30 Apr 2026 -> Test May–Jul 2026** | 3,866 | **$45.50** | **79.98%** | $70.57 | $129.17 | **0.898** |
| *Suburb x Bedroom Median (Naive Benchmark)* | *2,653* | *$180.00* | *28.99%* | *$227.81* | *$327.25* | *0.355* |
| *Microburbs Reference Valuation Benchmark* | — | *$86.00* | *55.21%* | — | — | *0.645* |

---

## 2. Who It Helps

* **Tenants & Renters:** Provides a transparent, building-specific benchmark before signing a lease—showing both the fair weekly rent for a given apartment and whether an asking price for a car space is reasonable (across the City of Sydney, a car space adds a median of **$36.50/week** or **~$1,898/year**, scaling from **+$20.50/week** for studios to **+$78.00/week** for 3-bedroom apartments).
* **Landlords & Property Managers:** Quantifies the exact marginal rental value of bundling vs. unbundling a car space in a specific apartment building and suburb before advertising a listing.
* **Property Analysts & Investors:** Separates macro market rent inflation from unit-level and building-level quality premiums across all 65,777 flats in the City of Sydney.

---

## 3. The Data
All data processing strictly enforces the **31 July 2026 temporal cutoff** (no records dated after `2026-07-31` are used anywhere in indexing, feature engineering, or training):

1. **`rents-input-flats-city-of-sydney.csv` (65,777 flats):** Target flats across the City of Sydney with `gnaf_id`, `building_address`, `suburb`, meshblock (`mb`), `latitude`, `longitude`, `property_type`, `bedrooms`, `bathrooms`, `car_space`, `car_spaces`, `floor_level`, `flats_in_building`, 2021–2026 block home growth, and listing history metadata.
2. **`rents-past-leases-city-of-sydney.csv` (112,457 historical rental adverts):** Cleaned to keep adverts dated <= `2026-07-31` with plausible weekly asking rents (`$150`–`$5,000/week`), de-duplicated, and linked to **105,594** input flat records spanning `2012-01-04` to `2026-07-31`.
3. **`blocks-homes-and-people-2021-2026.csv` (2,281 meshblocks):** Joined by meshblock (`mb`) to capture neighbourhood population density and new housing supply growth between 2021 and 2026.

*(Note: Per hackathon rules, supplied CSV files and API keys are excluded from this repository via `.gitignore`.)*

---

## 4. The Clever Step

Raw rental adverts span more than a decade (2012–2026), meaning a naive model confuses **when** a flat was leased with **how good** the flat is, and risks data leakage if historical averages include future adverts. Our pipeline solves this with a three-part design:

1. **De-Trending via L-BFGS-B Tuned Holt-Winters Monthly Rent Index:**
   We construct a bedroom-adjusted monthly median log-rent index and optimise additive Holt-Winters smoothing parameters (`alpha = 0.848, beta = 0.010, gamma = 0.990`) via `L-BFGS-B`. Every historical lease price is converted into a **time-invariant relative quality residual** (`lp_dt = log(price) - index_month`) before feature engineering, and the index is projected forward to the **August–September 2026** scoring window.
2. **Strictly Causal (Leakage-Safe) Multi-Level Historical Anchors:**
   Using de-trended log rents sorted chronologically, each training advert only sees **strictly earlier adverts** (`shift(1)` / cumulative sums excluding the current row) at three granularities:
   * Exact flat history (`own_prev_lp`, `own_prev_n`)
   * Same building x same bedroom count history (`bld_bed_prev_lp`, `bld_bed_prev_n`)
   * Same building overall history (`bld_prev_lp`, `bld_prev_n`)
   This single step reduced held-out median error from **$180.00/wk** down to **$42.50/wk**.
3. **L1-Loss Gradient Boosting with Monotonic Counterfactual Pairing:**
   We train `HistGradientBoostingRegressor(loss="absolute_error")`—directly optimising the hackathon's primary **Median Absolute Error** metric—with monotonic constraints (`+1`) on `car_flag`, `car_spaces_f`, `bedrooms_f`, and `bathrooms_f`. Every flat is scored twice (`car_space = 1` vs `car_space = 0`), guaranteeing a non-negative, building-aware parking premium rounded to `$0.50`.

---

## 5. How to Run It

### Prerequisites
* Python 3.10+ with `numpy`, `pandas`, `scipy`, and `scikit-learn`:
  ```bash
  pip install numpy pandas scipy scikit-learn
  ```

### Step 1: Place the Supplied Data Files
Place the three supplied hackathon CSV files in the same directory as `rents_city_of_sydney_model.py` (or point `RENTS_DATA_DIR` to their folder):
* `rents-input-flats-city-of-sydney.csv`
* `rents-past-leases-city-of-sydney.csv`
* `blocks-homes-and-people-2021-2026.csv`

*(Optional: If using the Microburbs API enrichment script, export your key as an environment variable `export MICROBURBS_API_KEY="your_key_here"` — never hardcode it.)*

### Step 2: Run the Model
```bash
python rents_city_of_sydney_model.py
```

### Step 3: Outputs Generated
Running the script executes the out-of-time validation check, prints the held-out accuracy table, runs contract assertions (65,777 rows, exact `gnaf_id` order, zero NaNs, `$0.50` rounding), and writes:
* `rents_quantum.csv` (and `rents_team-35.csv`)
* `report.html`
