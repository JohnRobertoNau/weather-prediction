"""
Colectarea datelor meteo din Open-Meteo Historical Weather API.

Pentru fiecare oraș din data/cities.csv face O cerere pentru toată perioada
(zilnic + orar) și salvează răspunsul brut în data/raw/<city_id>.json.
Orașele deja descărcate sunt sărite, deci scriptul poate fi oprit și repornit.

Exemple:
    python src/collect.py                      # toate orașele, 2020-2025
    python src/collect.py --cities bucharest   # un singur oraș (test)
    python src/collect.py --force              # descarcă din nou tot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"

ROOT = Path(__file__).resolve().parents[1]
CITIES_CSV = ROOT / "data" / "cities.csv"
RAW_DIR = ROOT / "data" / "raw"

# PAS 2: Variabile cerute zilnic (agregate de API în ziua locală a orașului)
DAILY_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "sunshine_duration",
]

# Variabile cerute orar, pe care le agregăm noi la medie zilnică
HOURLY_VARS = [
    "pressure_msl",
    "relative_humidity_2m",
    "cloud_cover",
]

PAUSE_BETWEEN_CITIES = 20     # secunde între orașe, ca să rămânem sub limita pe minut
MAX_RETRIES = 12

# Pas3: trimite cererea la API 
def fetch_city(lat: float, lon: float, start: str, end: str) -> dict:
    """Face cererea pentru un oraș; reîncearcă la erori de rețea și la limitele API-ului."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",          # zilele = zile locale ale orașului
        "wind_speed_unit": "kmh",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=120)
        except requests.RequestException as exc:
            wait = min(60, 5 * attempt)
            print(f"    eroare de rețea ({exc.__class__.__name__}), reîncerc în {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()

        reason = ""
        try:
            reason = resp.json().get("reason", "")
        except ValueError:
            reason = resp.text[:200]

        #Pas4: gestionarea erorilor, prea multe cereri, erori server, etc.
        if resp.status_code == 429:
            low = reason.lower()
            if "daily" in low:
                sys.exit(f"Limita zilnică a API-ului a fost atinsă: {reason}\n"
                         "Rulează din nou mâine; orașele deja descărcate sunt păstrate.")
            wait = 600 if "hourly" in low else 65
            print(f"    limită API ({reason or 429}), aștept {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            wait = min(120, 10 * attempt)
            print(f"    eroare server {resp.status_code}, reîncerc în {wait}s")
            time.sleep(wait)
            continue

        # 400 etc.: de obicei un nume de variabilă greșit -> nu are sens să reîncercăm
        sys.exit(f"Cerere respinsă ({resp.status_code}): {reason}")

    raise RuntimeError(f"Nu am reușit după {MAX_RETRIES} încercări")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarcă datele Open-Meteo pentru orașele din cities.csv")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--cities", nargs="*", help="city_id-uri de descărcat (implicit: toate)")
    parser.add_argument("--force", action="store_true", help="suprascrie fișierele existente")
    args = parser.parse_args()

    # PAS 1: se citeste lista de orase
    cities = pd.read_csv(CITIES_CSV)  # se citeste lista de orase
    if args.cities:
        unknown = set(args.cities) - set(cities["city_id"])
        if unknown:
            sys.exit(f"Orașe necunoscute: {', '.join(sorted(unknown))}")
        cities = cities[cities["city_id"].isin(args.cities)]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = len(cities)
    downloaded = 0

    for i, row in enumerate(cities.itertuples(index=False), start=1):
        out_path = RAW_DIR / f"{row.city_id}.json"
        if out_path.exists() and not args.force:
            print(f"[{i}/{total}] {row.city}: există deja, sar peste")
            continue

        if downloaded > 0:
            time.sleep(PAUSE_BETWEEN_CITIES)

        print(f"[{i}/{total}] {row.city}: descarc {args.start} .. {args.end}")
        data = fetch_city(row.latitude, row.longitude, args.start, args.end)


        #PAS 5: se salveaza raspunsurile
        data["_meta"] = {  
            "city_id": row.city_id,
            "requested_latitude": row.latitude,
            "requested_longitude": row.longitude,
            "start_date": args.start,
            "end_date": args.end,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": API_URL,
        }
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(out_path)
        downloaded += 1

        n_days = len(data["daily"]["time"])
        print(f"    ok: {n_days} zile, altitudine {data.get('elevation')} m, "
              f"fus orar {data.get('timezone')}")

    print(f"\nGata. Descărcate acum: {downloaded}. Fișiere în {RAW_DIR}")


if __name__ == "__main__":
    main()
