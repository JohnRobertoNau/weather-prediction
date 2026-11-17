"""
Construiește datasetul final din răspunsurile brute Open-Meteo (data/raw/*.json).

Pași:
  1. date zilnice din API + medii zilnice din datele orare (presiune, umiditate, nebulozitate)
  2. informații despre oraș (coordonate, zonă, altitudine)
  3. features derivate: calendar, vânt sin/cos, lag-uri, ferestre de 7 zile, variația presiunii
  4. ținte: ploua_maine (precipitații mâine >= 1 mm) și temp_max_maine
  5. elimină rândurile fără istoric complet sau fără țintă

Rezultat: data/processed/weather_dataset.csv  (+ data/processed/build_report.json)

    python src/build_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITIES_CSV = ROOT / "data" / "cities.csv"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_CSV = OUT_DIR / "weather_dataset.csv"
REPORT_JSON = OUT_DIR / "build_report.json"

RAIN_THRESHOLD_MM = 1.0   # o zi „cu ploaie” = cel puțin 1 mm

# Redenumiri: nume API -> nume scurte în dataset
RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precip",
    "precipitation_hours": "precip_hours",
    "wind_speed_10m_max": "wind_speed_max",
    "wind_gusts_10m_max": "wind_gusts_max",
    "wind_direction_10m_dominant": "wind_dir",
    "shortwave_radiation_sum": "radiation",
    "sunshine_duration": "sunshine_s",
    "pressure_msl": "pressure",
    "relative_humidity_2m": "humidity",
    "cloud_cover": "cloud_cover",
}

LAG_COLS = ["temp_max", "temp_min", "precip", "pressure", "humidity", "cloud_cover"]
LAGS = [1, 2, 3]
ROLL_MEAN_COLS = ["temp_mean", "pressure", "humidity", "cloud_cover"]
ROLL_WINDOW = 7


def load_city(path: Path) -> tuple[pd.DataFrame, dict]:
    """Citește un JSON brut și întoarce un DataFrame zilnic + metadate."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    daily = pd.DataFrame(raw["daily"])
    daily["date"] = pd.to_datetime(daily.pop("time"))

    hourly = pd.DataFrame(raw["hourly"])
    hourly["date"] = pd.to_datetime(hourly.pop("time")).dt.normalize()
    # Medie pe ziua locală; o zi se păstrează doar dacă are cel puțin 20 de ore valide
    agg = hourly.groupby("date").agg(["mean", "count"])
    hourly_daily = pd.DataFrame(index=agg.index)
    for var in hourly.columns.drop("date"):
        mean, count = agg[(var, "mean")], agg[(var, "count")]
        hourly_daily[var] = mean.where(count >= 20)
    hourly_daily = hourly_daily.reset_index()

    df = daily.merge(hourly_daily, on="date", how="left").rename(columns=RENAME)
    meta = {
        "city_id": raw["_meta"]["city_id"] if "_meta" in raw else path.stem,
        "elevation": raw.get("elevation"),
        "grid_latitude": raw.get("latitude"),
        "grid_longitude": raw.get("longitude"),
        "timezone": raw.get("timezone"),
    }
    return df, meta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivate pentru UN oraș (df sortat după dată, zile consecutive)."""
    df = df.sort_values("date").reset_index(drop=True)

    # Calendar
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Unități și transformări
    df["sunshine_hours"] = df.pop("sunshine_s") / 3600.0
    rad = np.deg2rad(df["wind_dir"])
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["rain_today"] = (df["precip"] >= RAIN_THRESHOLD_MM).astype("Int64")

    # Lag-uri: valoarea de acum k zile
    for col in LAG_COLS:
        for k in LAGS:
            df[f"{col}_lag{k}"] = df[col].shift(k)

    # Ferestre pe ultimele 7 zile (inclusiv azi) – folosesc doar trecutul
    for col in ROLL_MEAN_COLS:
        df[f"{col}_mean_7d"] = df[col].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
    df["precip_sum_7d"] = df["precip"].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).sum()
    df["rain_days_7d"] = df["rain_today"].astype(float).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).sum()

    # Variații față de ieri
    df["pressure_change_1d"] = df["pressure"] - df["pressure"].shift(1)
    df["temp_max_change_1d"] = df["temp_max"] - df["temp_max"].shift(1)

    # Ținte (ziua următoare)
    next_precip = df["precip"].shift(-1)
    df["ploua_maine"] = (next_precip >= RAIN_THRESHOLD_MM).astype("Int64").where(next_precip.notna())
    df["temp_max_maine"] = df["temp_max"].shift(-1)
    # ajută la verificare: ce precipitații au fost efectiv mâine
    df["precip_maine"] = next_precip
    return df


def main() -> None:
    cities = pd.read_csv(CITIES_CSV)
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        sys.exit(f"Nu există fișiere în {RAW_DIR}. Rulează întâi: python src/collect.py")

    frames, report_cities = [], []
    for path in files:
        df, meta = load_city(path)
        info = cities[cities["city_id"] == meta["city_id"]]
        if info.empty:
            print(f"! {path.name}: orașul nu e în cities.csv, îl sar")
            continue
        info = info.iloc[0]

        # Verificăm că zilele sunt consecutive (altfel lag-urile ar fi greșite)
        full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        if len(full_range) != len(df):
            df = df.set_index("date").reindex(full_range).rename_axis("date").reset_index()

        missing_before = int(df.drop(columns="date").isna().sum().sum())
        df = add_features(df)

        df.insert(0, "city_id", info["city_id"])
        df.insert(1, "city", info["city"])
        df.insert(2, "country", info["country"])
        df.insert(3, "zone", info["zone"])
        df.insert(4, "latitude", info["latitude"])
        df.insert(5, "longitude", info["longitude"])
        df.insert(6, "elevation", meta["elevation"])
        frames.append(df)
        report_cities.append({**meta, "rows": len(df), "missing_values_raw": missing_before})
        print(f"{info['city']:<12} {len(df)} zile, altitudine {meta['elevation']} m, "
              f"valori lipsă brute: {missing_before}")

    data = pd.concat(frames, ignore_index=True)
    rows_before = len(data)

    # Eliminăm rândurile fără istoric complet (primele zile) sau fără țintă (ultima zi)
    needed = [c for c in data.columns if c.endswith(("_lag3", "_7d"))] + ["ploua_maine", "temp_max_maine"]
    data = data.dropna(subset=needed).reset_index(drop=True)

    # Ordinea coloanelor: identificare, features, ținte
    front = ["city_id", "city", "country", "zone", "date", "year", "month", "day_of_year",
             "latitude", "longitude", "elevation"]
    targets = ["ploua_maine", "temp_max_maine", "precip_maine"]
    data = data[front + [c for c in data.columns if c not in front + targets] + targets]
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_CSV, index=False, float_format="%.6g")

    id_cols = {"city_id", "city", "country", "zone", "date", "year"}
    feature_cols = [c for c in data.columns if c not in id_cols | set(targets)]
    report = {
        "rows_before_cleaning": rows_before,
        "rows": len(data),
        "columns": data.shape[1],
        "feature_columns": len(feature_cols),
        "cities": len(frames),
        "date_min": data["date"].min(),
        "date_max": data["date"].max(),
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "rain_tomorrow_share": round(float(data["ploua_maine"].mean()), 4),
        "remaining_missing_values": {k: int(v) for k, v in data.isna().sum().items() if v > 0},
        "per_city": report_cities,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDataset: {len(data)} rânduri x {data.shape[1]} coloane "
          f"({len(feature_cols)} features) -> {OUT_CSV}")
    print(f"Zile cu ploaie mâine (>= {RAIN_THRESHOLD_MM} mm): {report['rain_tomorrow_share']:.1%}")


if __name__ == "__main__":
    main()
