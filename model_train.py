#!/usr/bin/env python
# coding: utf-8

# ## model_train
# 
# null

# In[3]:


"""
================================================================================
MICROBURBS HACKATHON - "WHAT WILL THIS FLAT RENT FOR?"
Weekly rent with / without a car space for all 65,777 City of Sydney flats.
================================================================================
Task contract (https://www.microburbs.com.au/p/hackathon-task-rents):
  * As-at date 31 July 2026: nothing dated after it is used anywhere.
  * Own check: build on earlier adverts, test on LATER adverts dated by 31 July 2026.
    The held-out adverts are never used for building or tuning.
  * Ranking metric: median absolute $ error; tie-break: share within $100.
  * Also report R squared on the held-out adverts.
  * Output rents_<team>.csv: gnaf_id, rent_with_car_space, rent_without_car_space
    - all 65,777 gnaf_ids, no extras, no blanks, values rounded to $0.50.

Method:
  1. Clean past leases (<= 31 Jul 2026, plausible weekly prices, de-duplicated).
  2. Monthly rent index (median log rent, bedroom-adjusted) -> Holt / Holt-Winters
     trend with alpha, beta, gamma tuned by L-BFGS-B, projected to Aug-Sep 2026
     (the scoring window). Prices are de-trended with this index before modelling.
  3. Gradient boosting (HistGradientBoostingRegressor, absolute-error loss = the
     median-error metric) on flat, building, block and location features plus
     leakage-safe history features (the flat's own earlier rent, its building's
     earlier rents for the same bedroom count).
  4. Parking premium: predict each flat twice, with car_space = 1 and = 0, with a
     monotonic constraint so rent_with >= rent_without.
================================================================================
"""

import sys, os, json, base64, warnings, subprocess
warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Auto-switch to Python 3.13 if VS Code launched this with Python 3.12 (which lacks numpy/sklearn)
try:
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    py313 = r"C:\Users\QCAO\AppData\Local\Programs\Python\Python313\python.exe"
    if os.path.exists(py313) and os.path.normcase(sys.executable) != os.path.normcase(py313):
        print("[INFO] Switching from Python 3.12 to Python 3.13...", flush=True)
        sys.exit(subprocess.call([py313, "-u"] + sys.argv))
    raise

# ------------------------------------------------------------------------------
# CONFIG - auto-detects data (Local CSV -> Fabric Lakehouse Files/Subfolders/Tables/OneLake ABFSS)
# ------------------------------------------------------------------------------
TEAM = "FantRent"
FABRIC_WS_ID = "66d24098-d04a-4662-97da-bf9c17e14f9f"
FABRIC_LH_ID = "3576468c-07c8-4ffc-b7b2-871015a889b0"
ABFSS_ROOT = f"abfss://{FABRIC_WS_ID}@onelake.dfs.fabric.microsoft.com/{FABRIC_LH_ID}"

_CANDIDATE_DIRS = [
    os.environ.get("RENTS_DATA_DIR", ""),
    "/lakehouse/default/Files",
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
    ".",
    r"C:\Users\QCAO\.gemini\antigravity\brain\51f2ba3a-4dc6-4251-baaf-409115842d2f\scratch\mock",
]
_DEFAULT_OUT = "/lakehouse/default/Files" if os.path.exists("/lakehouse/default/Files") else (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
)
OUT_DIR = os.environ.get("RENTS_OUT_DIR", _DEFAULT_OUT)


def _cast_df(df, dtype):
    if dtype == str:
        return df.astype(str)
    if isinstance(dtype, dict):
        for k, v in dtype.items():
            if k in df.columns:
                df[k] = df[k].astype(v)
    return df


def _norm_name(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _get_fs_utils():
    for mod_name in ("notebookutils", "mssparkutils"):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "fs"):
                return mod.fs
        except Exception:
            pass
    try:
        from notebookutils import mssparkutils
        return mssparkutils.fs
    except Exception:
        return None


def read_dataset(stem, dtype=None, required=True):
    """Read dataset from CSV file, Fabric subfolder, Lakehouse Table, or OneLake ABFSS via Spark."""
    target_norm = _norm_name(stem)
    for d in _CANDIDATE_DIRS:
        if not d or not os.path.exists(d):
            continue
        # 1. Direct CSV file: <d>/<stem>.csv
        p_csv = os.path.join(d, f"{stem}.csv")
        if os.path.isfile(p_csv):
            return pd.read_csv(p_csv, dtype=dtype)
        # 2. Subfolder: <d>/<stem>/ containing CSV or Parquet files
        p_dir = os.path.join(d, stem)
        if os.path.isdir(p_dir):
            csvs = [os.path.join(r, f) for r, _, fs in os.walk(p_dir) for f in fs if f.lower().endswith(".csv")]
            if csvs:
                return pd.concat([pd.read_csv(f, dtype=dtype) for f in csvs], ignore_index=True)
            pqs = [os.path.join(r, f) for r, _, fs in os.walk(p_dir) for f in fs if f.lower().endswith(".parquet") and not f.startswith("_")]
            if pqs:
                return _cast_df(pd.concat([pd.read_parquet(f) for f in pqs], ignore_index=True), dtype)
        # 3. Any matching CSV or folder anywhere under <d>
        for r, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(".csv") and target_norm in _norm_name(f):
                    return pd.read_csv(os.path.join(r, f), dtype=dtype)
    # 4. Mounted Fabric Lakehouse Tables directory (/lakehouse/default/Tables/...)
    for tbl_root in ["/lakehouse/default/Tables/dbo", "/lakehouse/default/Tables"]:
        if os.path.isdir(tbl_root):
            for entry in os.listdir(tbl_root):
                if target_norm in _norm_name(entry):
                    p_tbl = os.path.join(tbl_root, entry)
                    pqs = [os.path.join(r, f) for r, _, fs in os.walk(p_tbl) for f in fs if f.lower().endswith(".parquet") and not f.startswith("_")]
                    if pqs:
                        return _cast_df(pd.concat([pd.read_parquet(f) for f in pqs], ignore_index=True), dtype)
    # 5. PySpark / OneLake ABFSS discovery (works even if Lakehouse is not attached in Explorer!)
    discovered = []
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        fsu = _get_fs_utils()

        roots = [ABFSS_ROOT]
        try:
            import notebookutils
            for lh in notebookutils.lakehouse.list():
                abfs = lh.get("properties", {}).get("abfsPath") or lh.get("abfsPath")
                if abfs and abfs not in roots:
                    roots.append(abfs.rstrip("/"))
        except Exception:
            pass

        candidate_uris = []
        for rt in roots:
            # Direct known locations
            candidate_uris.extend([
                (f"{rt}/Tables/dbo/{stem}", "delta"),
                (f"{rt}/Tables/{stem}", "delta"),
                (f"{rt}/Tables/dbo/{stem.replace('-', '_')}", "delta"),
                (f"{rt}/Tables/{stem.replace('-', '_')}", "delta"),
                (f"{rt}/Files/{stem}", "csv"),
                (f"{rt}/Files/{stem}.csv", "csv"),
                (f"{rt}/Files/{stem}", "parquet"),
                (f"{rt}/Files/{stem}", "delta"),
            ])
            # Crawl OneLake directories using mssparkutils/notebookutils
            if fsu is not None:
                queue = [f"{rt}/Tables/dbo", f"{rt}/Tables", f"{rt}/Files"]
                visited = set()
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    try:
                        items = fsu.ls(curr)
                    except Exception:
                        continue
                    for it in items:
                        it_path = getattr(it, "path", "").rstrip("/")
                        it_name = getattr(it, "name", "")
                        is_dir = getattr(it, "isDir", False)
                        discovered.append(it_path)
                        if target_norm in _norm_name(it_name):
                            if it_name.lower().endswith(".csv"):
                                candidate_uris.insert(0, (it_path, "csv"))
                            elif it_name.lower().endswith(".parquet"):
                                candidate_uris.insert(0, (it_path, "parquet"))
                            else:
                                candidate_uris.insert(0, (it_path, "delta"))
                                candidate_uris.insert(1, (it_path, "csv"))
                                candidate_uris.insert(2, (it_path, "parquet"))
                        if is_dir and len(visited) < 40 and not it_name.startswith("_"):
                            queue.append(it_path)

        seen_try = set()
        for uri, fmt in candidate_uris:
            if (uri, fmt) in seen_try:
                continue
            seen_try.add((uri, fmt))
            try:
                if fmt == "csv":
                    sdf = spark.read.option("header", "true").option("recursiveFileLookup", "true").csv(uri)
                elif fmt == "parquet":
                    sdf = spark.read.option("recursiveFileLookup", "true").parquet(uri)
                else:
                    sdf = spark.read.format(fmt).load(uri)
                if len(sdf.columns) > 1:
                    pdf = sdf.toPandas()
                    if len(pdf) > 0:
                        log(f"[SPARK] Loaded '{stem}' ({len(pdf):,} rows) from {uri} [{fmt}]")
                        return _cast_df(pdf, dtype)
            except Exception:
                continue
    except Exception as e:
        log(f"[SPARK WARN] {e}")
    if required:
        raise FileNotFoundError(
            f"Could not find non-empty dataset '{stem}'. Discovered OneLake paths: {discovered[:25]}"
        )
    return None

AS_AT = pd.Timestamp("2026-07-31")
HOLDOUT_START = pd.Timestamp("2026-05-01")      # own check: May-Jul 2026 adverts
SCORE_MONTHS = [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")]
PRICE_MIN, PRICE_MAX = 150, 5000                # plausible weekly asking rents
EXPECTED_ROWS = 65777
SEED = 42


def log(msg=""):
    print(msg, flush=True)


# ------------------------------------------------------------------------------
# 1. LOAD + COLUMN DETECTION (past-lease columns are not documented publicly)
# ------------------------------------------------------------------------------
def pick(df, candidates, required=True, name=""):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    for cand in candidates:                      # fuzzy contains-match
        for lc, orig in cols.items():
            if cand in lc:
                return orig
    if required:
        raise KeyError(f"Cannot find column for '{name}'. Columns are: {list(df.columns)}")
    return None


def load_data():
    flats = read_dataset("rents-input-flats-city-of-sydney", dtype={"gnaf_id": str, "mb": str})
    flats["gnaf_id"] = flats["gnaf_id"].astype(str)
    if "mb" in flats.columns:
        flats["mb"] = flats["mb"].astype(str)
    leases_raw = read_dataset("rents-past-leases-city-of-sydney", dtype=str)
    blocks = read_dataset("blocks-homes-and-people-2021-2026", dtype=str, required=False)
    log(f"[LOAD] input flats: {len(flats):,} rows | past leases: {len(leases_raw):,} rows"
        + (f" | blocks: {len(blocks):,} rows" if blocks is not None else ""))
    log(f"[LOAD] past-lease columns: {list(leases_raw.columns)}")

    c_date = pick(leases_raw, ["date", "listing_date", "advert_date", "lease_date", "listed"], name="date")
    c_price = pick(leases_raw, ["price", "weekly_rent", "rent", "asking_rent"], name="price")
    c_gnaf = pick(leases_raw, ["gnaf_id", "gnaf_pid", "gnaf"], required=False)
    c_bldg = pick(leases_raw, ["building_address"], required=False)
    c_bed = pick(leases_raw, ["bedrooms", "beds"], required=False)
    c_bath = pick(leases_raw, ["bathrooms", "baths"], required=False)
    c_car = pick(leases_raw, ["car_spaces", "garage_spaces", "parking", "car_space", "cars"], required=False)
    if c_gnaf is None and c_bldg is None:
        raise KeyError("Past leases need gnaf_id or building_address to link to flats.")

    L = pd.DataFrame({
        "date": pd.to_datetime(leases_raw[c_date], errors="coerce"),
        "price": pd.to_numeric(leases_raw[c_price].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                               errors="coerce"),
    })
    L["gnaf_id"] = leases_raw[c_gnaf].astype(str) if c_gnaf else np.nan
    L["building_address"] = leases_raw[c_bldg].astype(str).str.upper().str.strip() if c_bldg else np.nan
    L["adv_bedrooms"] = pd.to_numeric(leases_raw[c_bed], errors="coerce") if c_bed else np.nan
    L["adv_bathrooms"] = pd.to_numeric(leases_raw[c_bath], errors="coerce") if c_bath else np.nan
    L["adv_car_spaces"] = pd.to_numeric(leases_raw[c_car], errors="coerce") if c_car else np.nan

    # --- cleaning -------------------------------------------------------------
    n0 = len(L)
    L = L.dropna(subset=["date", "price"])
    L = L[L["date"] <= AS_AT]                          # as-at rule
    L = L[(L["price"] >= PRICE_MIN) & (L["price"] <= PRICE_MAX)]
    L = L.drop_duplicates(subset=["gnaf_id", "building_address", "date", "price"])
    L = L.sort_values("date").reset_index(drop=True)
    log(f"[CLEAN] kept {len(L):,} of {n0:,} adverts (dated <= {AS_AT.date()}, "
        f"${PRICE_MIN}-${PRICE_MAX}/wk, de-duplicated). "
        f"Range {L['date'].min().date()} -> {L['date'].max().date()}")

    flats["building_address"] = flats["building_address"].astype(str).str.upper().str.strip()
    return flats, L, blocks


# ------------------------------------------------------------------------------
# 2. FEATURES FROM THE INPUT FLAT TABLE
# ------------------------------------------------------------------------------
NUM_FLAT_COLS = ["latitude", "longitude", "bedrooms", "bathrooms", "car_space", "car_spaces",
                 "floor_level", "flats_in_building", "block_new_homes_2021_2026",
                 "block_homes_2026", "ever_advertised_for_lease"]


def prepare_flats(flats, blocks):
    F = flats.copy()
    for c in NUM_FLAT_COLS:
        if c in F.columns:
            F[c] = pd.to_numeric(F[c], errors="coerce")
    F["first_listed_year"] = pd.to_datetime(F.get("first_ever_listed"), errors="coerce").dt.year
    F["attrs_year"] = pd.to_datetime(F.get("attrs_last_seen"), errors="coerce").dt.year
    F.loc[F["attrs_year"] < 1900, "attrs_year"] = np.nan
    F["suburb_code"] = F["suburb"].astype("category").cat.codes
    F["ptype_code"] = F["property_type"].astype("category").cat.codes
    F["block_growth"] = F["block_new_homes_2021_2026"] / F["block_homes_2026"].replace(0, np.nan)
    # distance to Sydney CBD (Town Hall) in km - simple location gradient
    F["dist_cbd_km"] = np.sqrt(((F["latitude"] + 33.8731) * 111.0) ** 2 +
                               ((F["longitude"] - 151.2065) * 92.5) ** 2)

    # optional block census join: numeric columns of the block file, keyed on mb
    if blocks is not None:
        kb = pick(blocks, ["mb", "mb_code", "mb_code_2021", "block"], required=False)
        if kb:
            B = blocks.copy()
            B["mb"] = B[kb].astype(str)
            num = {}
            for c in B.columns:
                if c in (kb, "mb"):
                    continue
                v = pd.to_numeric(B[c], errors="coerce")
                if v.notna().mean() > 0.8:
                    num["blk_" + c] = v
            if num:
                B2 = pd.concat([B[["mb"]], pd.DataFrame(num)], axis=1).drop_duplicates("mb")
                F = F.merge(B2, on="mb", how="left")
    return F


# ------------------------------------------------------------------------------
# 3. MONTHLY RENT INDEX + HOLT / HOLT-WINTERS PROJECTION (alpha, beta, gamma by L-BFGS-B)
# ------------------------------------------------------------------------------
def monthly_index(L):
    """Bedroom-adjusted monthly median of log rent (index in log points)."""
    d = L.copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["lp"] = np.log(d["price"])
    bed = d["bedrooms_f"].fillna(-1).clip(upper=4)
    d["lp_adj"] = d["lp"] - d.groupby(bed)["lp"].transform("median")
    s = d.groupby("month")["lp_adj"].median()
    cnt = d.groupby("month").size()
    s = s[cnt >= 20]                                     # drop thin months
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full).interpolate().ffill().bfill()


def holt_winters_fit(y, m=12):
    """Additive Holt-Winters on log index if >= 2 seasons, else Holt linear.
    Smoothing weights are tuned by minimising one-step-ahead MSE with L-BFGS-B."""
    y = np.asarray(y, float)
    n = len(y)
    seasonal = n >= 2 * m

    def run(p):
        a, b = p[0], p[1]
        g = p[2] if seasonal else 0.0
        L0 = y[:m].mean() if seasonal else y[0]
        T0 = (y[m:2 * m].mean() - y[:m].mean()) / m if seasonal else (y[1] - y[0] if n > 1 else 0.0)
        S = list(y[:m] - y[:m].mean()) if seasonal else [0.0] * m
        Lv, Tv, err = L0, T0, []
        for t in range(n):
            s = S[t % m] if seasonal else 0.0
            fc = Lv + Tv + s
            err.append(y[t] - fc)
            Ln = a * (y[t] - s) + (1 - a) * (Lv + Tv)
            Tv = b * (Ln - Lv) + (1 - b) * Tv
            if seasonal:
                S[t % m] = g * (y[t] - Ln) + (1 - g) * s
            Lv = Ln
        return np.array(err), Lv, Tv, S

    x0 = [0.3, 0.05, 0.1] if seasonal else [0.3, 0.05]
    bounds = [(0.01, 0.99)] * len(x0)
    burn = m if seasonal else 2
    res = minimize(lambda p: np.mean(run(p)[0][burn:] ** 2), x0, bounds=bounds, method="L-BFGS-B")
    err, Lv, Tv, S = run(res.x)
    params = dict(zip(["alpha", "beta", "gamma"], [float(v) for v in res.x]))

    def forecast(h):
        return float(Lv + h * Tv + (S[(n - 1 + h) % m] if seasonal else 0.0))
    return params, forecast, seasonal


def index_value_for(idx_series, forecast_fn, month_ts):
    if month_ts in idx_series.index:
        return float(idx_series.loc[month_ts])
    last = idx_series.index.max()
    h = (month_ts.year - last.year) * 12 + (month_ts.month - last.month)
    return forecast_fn(max(h, 1))


# ------------------------------------------------------------------------------
# 4. LEAKAGE-SAFE HISTORY FEATURES
# ------------------------------------------------------------------------------
def add_prior_history(T):
    """For each training advert use only STRICTLY EARLIER adverts."""
    T = T.sort_values("date").reset_index(drop=True)
    # same flat: last earlier de-trended log price
    g = T.groupby("gnaf_id", sort=False)["lp_dt"]
    T["own_prev_lp"] = g.shift(1)
    T["own_prev_n"] = T.groupby("gnaf_id", sort=False).cumcount()
    # same building + bedroom count: mean of earlier adverts
    key = T["building_address"].astype(str) + "|" + T["bedrooms_f"].fillna(-1).astype(int).astype(str)
    cs = T.groupby(key, sort=False)["lp_dt"].cumsum() - T["lp_dt"]
    cn = T.groupby(key, sort=False).cumcount()
    T["bld_bed_prev_lp"] = np.where(cn > 0, cs / cn.replace(0, np.nan), np.nan)
    T["bld_bed_prev_n"] = cn
    # same building any bedroom
    kb = T["building_address"].astype(str)
    cs2 = T.groupby(kb, sort=False)["lp_dt"].cumsum() - T["lp_dt"]
    cn2 = T.groupby(kb, sort=False).cumcount()
    T["bld_prev_lp"] = np.where(cn2 > 0, cs2 / cn2.replace(0, np.nan), np.nan)
    T["bld_prev_n"] = cn2
    return T


def history_for_targets(T_hist, targets):
    """For prediction rows: use ALL adverts in T_hist (all dated before the target period)."""
    out = targets.copy()
    own = T_hist.sort_values("date").groupby("gnaf_id")["lp_dt"].last()
    own_n = T_hist.groupby("gnaf_id").size()
    out["own_prev_lp"] = out["gnaf_id"].map(own)
    out["own_prev_n"] = out["gnaf_id"].map(own_n).fillna(0)
    kh = T_hist["building_address"].astype(str) + "|" + T_hist["bedrooms_f"].fillna(-1).astype(int).astype(str)
    kt = out["building_address"].astype(str) + "|" + out["bedrooms_f"].fillna(-1).astype(int).astype(str)
    bb = T_hist.groupby(kh)["lp_dt"].agg(["mean", "size"])
    out["bld_bed_prev_lp"] = kt.map(bb["mean"])
    out["bld_bed_prev_n"] = kt.map(bb["size"]).fillna(0)
    b = T_hist.groupby(T_hist["building_address"].astype(str))["lp_dt"].agg(["mean", "size"])
    out["bld_prev_lp"] = out["building_address"].astype(str).map(b["mean"])
    out["bld_prev_n"] = out["building_address"].astype(str).map(b["size"]).fillna(0)
    return out


# ------------------------------------------------------------------------------
# 5. MODEL
# ------------------------------------------------------------------------------
HIST_COLS = ["own_prev_lp", "own_prev_n", "bld_bed_prev_lp", "bld_bed_prev_n", "bld_prev_lp", "bld_prev_n"]


def feature_list(F):
    base = ["latitude", "longitude", "dist_cbd_km", "bedrooms_f", "bathrooms_f", "car_flag",
            "car_spaces_f", "floor_level", "flats_in_building", "block_new_homes_2021_2026",
            "block_homes_2026", "block_growth", "ever_advertised_for_lease", "first_listed_year",
            "attrs_year", "suburb_code", "ptype_code"]
    base += [c for c in F.columns if c.startswith("blk_")]
    return [c for c in base if c in F.columns] + HIST_COLS


def make_model(features):
    mono = [0] * len(features)
    for name, sign in [("car_flag", 1), ("car_spaces_f", 1), ("bedrooms_f", 1), ("bathrooms_f", 1)]:
        if name in features:
            mono[features.index(name)] = sign
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=1500, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=1.0, monotonic_cst=mono,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=SEED)


def build_training_table(L, F):
    """Attach flat attributes to each advert. Advert-level attributes override when present."""
    if L["gnaf_id"].notna().any() and (L["gnaf_id"] != "nan").any():
        T = L.drop(columns=["building_address"]).merge(F, on="gnaf_id", how="inner")
    else:
        T = L.merge(F.drop_duplicates("building_address"), on="building_address", how="inner")
    T["bedrooms_f"] = T["adv_bedrooms"].fillna(T["bedrooms"])
    T["bathrooms_f"] = T["adv_bathrooms"].fillna(T["bathrooms"])
    T["car_spaces_f"] = T["adv_car_spaces"].fillna(T["car_spaces"])
    T["car_flag"] = np.where(T["adv_car_spaces"].notna(), (T["adv_car_spaces"] > 0).astype(float),
                             T["car_space"])
    return T


def detrend(T, idx, fc):
    month = T["date"].dt.to_period("M").dt.to_timestamp()
    T["idx"] = [index_value_for(idx, fc, mth) for mth in month]
    T["lp"] = np.log(T["price"])
    T["lp_dt"] = T["lp"] - T["idx"]
    return T


def evaluate(y_true, y_pred):
    e = np.abs(y_true - y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {"median_abs_error": float(np.median(e)), "share_within_100": float(np.mean(e <= 100)),
            "MAE": float(e.mean()), "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "R2": float(1 - ss_res / ss_tot), "n": int(len(e))}


def fit_pipeline(T_train, F, features):
    """Fit index + model using ONLY T_train. Returns objects needed to predict."""
    idx = monthly_index(T_train)
    hw_params, fc, seasonal = holt_winters_fit(idx.values)
    T_train = detrend(T_train, idx, fc)
    T_train = add_prior_history(T_train)
    model = make_model(features)
    model.fit(T_train[features], T_train["lp_dt"])
    return {"idx": idx, "fc": fc, "hw": hw_params, "seasonal": seasonal, "model": model, "T": T_train}


def predict_rows(fitted, rows, features, target_months):
    lvl = np.mean([index_value_for(fitted["idx"], fitted["fc"], m) for m in target_months])
    X = history_for_targets(fitted["T"], rows)
    return np.exp(fitted["model"].predict(X[features]) + lvl), X


def round_half(x):
    return np.round(np.asarray(x, float) * 2.0) / 2.0


# ------------------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------------------
def main():
    log("=" * 88)
    log(" MICROBURBS HACKATHON - WEEKLY RENT WITH / WITHOUT CAR SPACE (City of Sydney)")
    log("=" * 88)
    flats, L, blocks = load_data()
    F = prepare_flats(flats, blocks)
    T_all = build_training_table(L, F)
    log(f"[LINK] {len(T_all):,} adverts linked to input flats")
    features = feature_list(F.assign(bedrooms_f=0, bathrooms_f=0, car_flag=0, car_spaces_f=0))

    # ---------------- held-out check: earlier adverts -> later adverts -----------------
    tr = T_all[T_all["date"] < HOLDOUT_START].copy()
    te = T_all[(T_all["date"] >= HOLDOUT_START) & (T_all["date"] <= AS_AT)].copy()
    log(f"\n[SPLIT] build: {len(tr):,} adverts dated < {HOLDOUT_START.date()} | "
        f"held-out: {len(te):,} adverts {HOLDOUT_START.date()} -> {AS_AT.date()}")
    fit_v = fit_pipeline(tr, F, features)
    te_months = sorted(te["date"].dt.to_period("M").dt.to_timestamp().unique())
    # predict each held-out advert at its own month level
    preds = np.zeros(len(te))
    for mth in te_months:
        mask = (te["date"].dt.to_period("M").dt.to_timestamp() == mth).values
        preds[mask], _ = predict_rows(fit_v, te[mask], features, [mth])
    preds = round_half(preds)
    hold = evaluate(te["price"].values, preds)

    # naive benchmark: suburb x bedroom median of the build set
    key_tr = tr["suburb"].astype(str) + "|" + tr["bedrooms_f"].fillna(-1).astype(int).astype(str)
    key_te = te["suburb"].astype(str) + "|" + te["bedrooms_f"].fillna(-1).astype(int).astype(str)
    med = tr.groupby(key_tr)["price"].median()
    bench_pred = key_te.map(med).fillna(tr["price"].median()).values
    bench = evaluate(te["price"].values, bench_pred)

    log(f"[HOLT-WINTERS] seasonal={fit_v['seasonal']} params={ {k: round(v, 3) for k, v in fit_v['hw'].items()} }")
    log("\n[HELD-OUT ACCURACY]                     model      suburb x bed median")
    log(f"   median absolute error ($/wk)    {hold['median_abs_error']:>9.2f}   {bench['median_abs_error']:>9.2f}")
    log(f"   share within $100               {hold['share_within_100']:>9.2%}   {bench['share_within_100']:>9.2%}")
    log(f"   MAE ($/wk)                      {hold['MAE']:>9.2f}   {bench['MAE']:>9.2f}")
    log(f"   R squared                       {hold['R2']:>9.3f}   {bench['R2']:>9.3f}")
    log(f"   adverts scored                  {hold['n']:>9,}")
    log(f"   (Microburbs reference: their valuation $86 median error, 55.21% within $100)")

    # ---------------- final model on ALL adverts <= 31 Jul 2026 ------------------------
    fit_f = fit_pipeline(T_all.copy(), F, features)
    P = F.copy()
    P["bedrooms_f"] = P["bedrooms"]
    P["bathrooms_f"] = P["bathrooms"]
    P_with = P.assign(car_flag=1.0, car_spaces_f=P["car_spaces"].where(P["car_spaces"] > 0, 1.0))
    P_without = P.assign(car_flag=0.0, car_spaces_f=0.0)
    r_with, _ = predict_rows(fit_f, P_with, features, SCORE_MONTHS)
    r_without, _ = predict_rows(fit_f, P_without, features, SCORE_MONTHS)
    r_with = np.maximum(r_with, r_without)              # premium never negative

    # safety fallback for any non-finite value
    fb = P["suburb"].astype(str).map(T_all.groupby("suburb")["price"].median()).fillna(T_all["price"].median())
    r_without = np.where(np.isfinite(r_without), r_without, fb)
    r_with = np.where(np.isfinite(r_with), r_with, r_without + 50)

    sub = pd.DataFrame({"gnaf_id": F["gnaf_id"].astype(str),
                        "rent_with_car_space": round_half(r_with),
                        "rent_without_car_space": round_half(r_without)})

    # ---------------- contract checks ---------------------------------------------------
    assert len(sub) == len(flats), "row count changed"
    assert sub["gnaf_id"].tolist() == flats["gnaf_id"].astype(str).tolist(), "gnaf_id order/values changed"
    assert sub.isna().sum().sum() == 0, "blank values present"
    assert ((sub[["rent_with_car_space", "rent_without_car_space"]] * 2) % 1 == 0).all().all(), "not $0.50 steps"
    if len(sub) != EXPECTED_ROWS:
        log(f"[WARN] input has {len(sub):,} rows, task says {EXPECTED_ROWS:,}")

    out_dirs = [OUT_DIR]
    kqdb_dir = "/lakehouse/default/Files/kqdb"
    if os.path.isdir(kqdb_dir) and kqdb_dir not in out_dirs:
        out_dirs.append(kqdb_dir)

    out_csv = os.path.join(OUT_DIR, f"rents_{TEAM}.csv")
    for d_out in out_dirs:
        for t_slug in [TEAM, "fant-rent", "team-35"]:
            sub.to_csv(os.path.join(d_out, f"rents_{t_slug}.csv"), index=False)

    prem = sub["rent_with_car_space"] - sub["rent_without_car_space"]
    log(f"\n[OUTPUT] {out_csv} (+ rents_fant-rent.csv, rents_team-35.csv): {len(sub):,} rows, 0 blanks")
    if len(out_dirs) > 1:
        log(f"[OUTPUT] Also saved into {kqdb_dir}/")
    log(f"   median rent without car ${sub['rent_without_car_space'].median():.0f}/wk | "
        f"median parking premium ${prem.median():.0f}/wk")
    log(sub.head(5).to_string(index=False))

    write_report(hold, bench, fit_v, fit_f, tr, te, T_all, sub, prem, features, out_dirs)
    return sub, hold


# ------------------------------------------------------------------------------
# 7. report.html (required hand-in: insights, held-out score, R squared, split, method)
# ------------------------------------------------------------------------------
def write_report(hold, bench, fit_v, fit_f, tr, te, T_all, sub, prem, features, out_dirs=None):
    if out_dirs is None:
        out_dirs = [OUT_DIR]
    F_ = sub.merge(T_all.drop_duplicates("gnaf_id")[["gnaf_id", "suburb"]], on="gnaf_id", how="left")
    by_bed = T_all.groupby(T_all["bedrooms_f"].fillna(-1).astype(int).clip(upper=4))["price"].median()
    car_eff = T_all.groupby([T_all["bedrooms_f"].fillna(-1).astype(int).clip(upper=3), "car_flag"])["price"].median().unstack()
    rows = "".join(f"<tr><td>{k}</td><td>{hold[k]:.3f}</td><td>{bench[k]:.3f}</td></tr>"
                   for k in ["median_abs_error", "share_within_100", "MAE", "RMSE", "R2"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Rents - {TEAM}</title>
<style>body{{font-family:Segoe UI,Arial;max-width:960px;margin:32px auto;color:#1b2a3a}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:6px 10px}}th{{background:#1B365D;color:#fff}}</style></head><body>
<h1>What will this flat rent for? - team {TEAM}</h1>
<h2>Insights</h2><ul>
<li>Median parking premium across all {len(sub):,} flats: <b>${prem.median():.0f}/wk</b> (IQR ${prem.quantile(.25):.0f}-${prem.quantile(.75):.0f}).</li>
<li>Median advertised rent by bedrooms (cleaned adverts &le; 31 Jul 2026): {by_bed.round(0).to_dict()}.</li>
<li>Raw median rent with vs without car space by bedrooms: {car_eff.round(0).to_dict()}.</li>
<li>Rent trend (Holt{'-Winters' if fit_f['seasonal'] else ''} on monthly bedroom-adjusted log index): params {json.dumps({k: round(v, 3) for k, v in fit_f['hw'].items()})}.</li>
</ul>
<h2>Held-out accuracy (own check)</h2>
<p>Built on {len(tr):,} adverts dated before {HOLDOUT_START.date()}; tested on {len(te):,} later adverts
dated {HOLDOUT_START.date()} to {AS_AT.date()}. The held-out adverts were never used to build or tune the method
(the index, the history features and the model were all re-fitted on the build set only).</p>
<table><tr><th>metric</th><th>our model</th><th>suburb x bedroom median</th></tr>{rows}</table>
<p><b>Median absolute error ${hold['median_abs_error']:.2f}/wk, {hold['share_within_100']:.1%} within $100, R&sup2; = {hold['R2']:.3f}.</b></p>
<h2>Data</h2><ul><li>rents-input-flats-city-of-sydney.csv (65,777 flats)</li>
<li>rents-past-leases-city-of-sydney.csv, filtered to adverts dated &le; 31 July 2026, ${PRICE_MIN}-${PRICE_MAX}/wk, de-duplicated</li>
<li>blocks-homes-and-people-2021-2026.csv (block census counts and 2026 estimates)</li></ul>
<h2>Method</h2><ol>
<li>Monthly bedroom-adjusted median log rent index; Holt / Holt-Winters smoothing weights tuned by L-BFGS-B; projected to Aug-Sep 2026.</li>
<li>Gradient boosting with absolute-error loss (matches the median-error ranking) on de-trended log rent.</li>
<li>Features: {', '.join(features)}.</li>
<li>History features use only strictly earlier adverts during training (no leakage).</li>
<li>Each flat predicted twice (car_space = 1 / 0) with a monotonic constraint, so the difference is the parking premium. Rounded to $0.50.</li>
</ol></body></html>"""
    for d_out in out_dirs:
        path = os.path.join(d_out, "report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"[OUTPUT] {path}")
    try:
        from notebookutils import mssparkutils
        csv_text = sub.to_csv(index=False)
        for rel_dir in ["Files/kqdb", "Files"]:
            for t_slug in [TEAM, "quantum"]:
                mssparkutils.fs.put(f"{ABFSS_ROOT}/{rel_dir}/rents_{t_slug}.csv", csv_text, True)
            mssparkutils.fs.put(f"{ABFSS_ROOT}/{rel_dir}/report.html", html, True)
        log(f"[OUTPUT] Saved CSVs and report.html directly to Lakehouse Data > Files/kqdb")
    except Exception:
        pass
    try:
        from IPython.display import display, HTML
        b64_rep = base64.b64encode(html.encode("utf-8")).decode("ascii")
        b64_csv = base64.b64encode(sub.to_csv(index=False).encode("utf-8")).decode("ascii")
        display(HTML(
            f'<div style="margin:12px 0;padding:12px;border:2px solid #1B365D;border-radius:8px;background:#f4f8fb">'
            f'<b> download this file:</b><br><br>'
            f'<a download="rents_quantum.csv" href="data:text/csv;base64,{b64_csv}" '
            f'style="background:#1B365D;color:#fff;padding:8px 14px;border-radius:5px;text-decoration:none;margin-right:10px">⬇ Tải rents_quantum.csv</a>'
            f'<a download="rents_team-35.csv" href="data:text/csv;base64,{b64_csv}" '
            f'style="background:#2B547E;color:#fff;padding:8px 14px;border-radius:5px;text-decoration:none;margin-right:10px">⬇ Tải rents_team-35.csv</a>'
            f'<a download="report.html" href="data:text/html;base64,{b64_rep}" '
            f'style="background:#006644;color:#fff;padding:8px 14px;border-radius:5px;text-decoration:none">⬇ Tải report.html</a>'
            f'</div>'
        ))
    except Exception:
        pass


if __name__ == "__main__":
    main()

