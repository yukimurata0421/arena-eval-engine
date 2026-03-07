"""
get_opensky_traffic.py

Fetch daily arrival/departure counts for Haneda/Narita from the OpenSky API.

Credentials:
  Read credentials.json (recommended).

  Default path:
    <project>/scripts/secrets/opensky_credentials.json
    (template: <project>/scripts/secrets/opensky_credentials.example.json)

  Alternative path (PowerShell):
    $env:OPENSKY_CREDENTIALS_JSON = "<path>\opensky_credentials.json"

  JSON key examples (either is OK):
    {"clientId":"xxx","clientSecret":"yyy"}
    {"client_id":"xxx","client_secret":"yyy"}

Never commit credentials.json to Git (.gitignore is recommended).
"""


import requests
import pandas as pd
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta


from arena.lib.paths import DATA_DIR, SCRIPTS_ROOT

DEFAULT_CRED_PATH = SCRIPTS_ROOT / "secrets" / "opensky_credentials.json"
CRED_PATH = Path(os.environ.get("OPENSKY_CREDENTIALS_JSON", str(DEFAULT_CRED_PATH)))


def load_opensky_credentials(cred_path: Path):
    """
    Read credentials.json and return (client_id, client_secret).
    Supports both clientId/clientSecret and client_id/client_secret.
    """
    if not cred_path.exists():
        print(f"  Error: credentials.json  not found: {cred_path}")
        print("  Action:")
        print(f"   - Place it at the default path: {DEFAULT_CRED_PATH}")
        print('   - Or set OPENSKY_CREDENTIALS_JSON to the full path')
        return "", ""

    try:
        obj = json.loads(cred_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  Error: failed to read credentials.json: {cred_path} ({e})")
        return "", ""

    client_id = (obj.get("clientId") or obj.get("client_id") or "").strip()
    client_secret = (obj.get("clientSecret") or obj.get("client_secret") or "").strip()

    if not client_id or not client_secret:
        print(f"  Error: credentials.json missing clientId/clientSecret (or client_id/client_secret): {cred_path}")
        return "", ""

    return client_id, client_secret


CLIENT_ID, CLIENT_SECRET = load_opensky_credentials(CRED_PATH)

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
BASE_URL  = "https://opensky-network.org/api/flights"
AIRPORTS  = {"HND": "RJTT", "NRT": "RJAA"}

OUTPUT_DIR  = str(DATA_DIR / "flight_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "airport_movements.csv")

START_DATE = datetime(2025, 12, 1)
END_DATE   = datetime.now()

MAX_RETRIES      = 3
RETRY_WAIT_SEC   = 10
REQUEST_INTERVAL = 2


def get_access_token():
    """Get OAuth2 token."""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("  Error: OpenSky credentials are not configured.")
        print(f"  Expected path: {CRED_PATH}")
        return None

    payload = {
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    try:
        res = requests.post(TOKEN_URL, data=payload, timeout=30)
        res.raise_for_status()
        token = res.json().get("access_token")
        if token:
            print("  Authentication success")
        return token

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        print(f"  Auth error (HTTP {status}): {e}")
        return None
    except Exception as e:
        print(f"  Auth error: {e}")
        return None


def get_flight_count(token, icao, start_ts, end_ts, mode):
    """Fetch flight count for a given airport/direction (with retries)."""
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"airport": icao, "begin": int(start_ts), "end": int(end_ts)}

    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(
                f"{BASE_URL}/{mode}",
                params=params,
                headers=headers,
                timeout=30
            )

            if res.status_code == 200:
                return len(res.json())
            elif res.status_code == 404:
                return 0
            elif res.status_code == 429:
                return "LIMIT_REACHED"
            elif res.status_code == 401:
                return "AUTH_EXPIRED"
            else:
                print(f"\n    HTTP {res.status_code} ({icao}/{mode})", end="")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_WAIT_SEC)
                    continue
                return 0

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_SEC)
                continue
            return 0
        except Exception:
            return 0

    return 0


def fetch_day_counts(token, day_date):
    """Fetch daily arrivals/departures and return as dict. Refresh token if needed."""
    # Accept both datetime.date and datetime.datetime
    if isinstance(day_date, datetime):
        day_dt = day_date
    else:
        day_dt = datetime.combine(day_date, datetime.min.time())

    date_str = day_dt.strftime("%Y-%m-%d")
    s_ts = day_dt.timestamp()
    e_ts = (day_dt + timedelta(days=1)).timestamp()

    day_results = {"date": date_str}
    total = 0

    for name, icao in AIRPORTS.items():
        for mode in ["arrival", "departure"]:
            count = get_flight_count(token, icao, s_ts, e_ts, mode)

            if count == "LIMIT_REACHED":
                return None, token, "LIMIT_REACHED"

            elif count == "AUTH_EXPIRED":
                print("\n  Token expired. Refreshing...", end=" ")
                token = get_access_token()
                if not token:
                    return None, token, "AUTH_EXPIRED"

                count = get_flight_count(token, icao, s_ts, e_ts, mode)
                if count in ("LIMIT_REACHED", "AUTH_EXPIRED"):
                    return None, token, "AUTH_EXPIRED"

            day_results[f"{name.lower()}_{mode[:3]}"] = count
            total += count
            time.sleep(REQUEST_INTERVAL)

    day_results["hnd_nrt_movements"] = total
    return day_results, token, None


def load_existing_data():
    """Load existing CSV (for resume)."""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return df, set(df["date"].tolist())
        except Exception:
            pass
    return pd.DataFrame(), set()


def day_differs(df_all, day_results):
    """Check if there are differences vs existing data."""
    date_str = day_results["date"]
    rows = df_all[df_all["date"] == date_str]
    if rows.empty:
        return True

    row = rows.iloc[-1]
    for key, val in day_results.items():
        if key == "date":
            continue
        if key not in row or pd.isna(row[key]) or int(row[key]) != int(val):
            return True
    return False


def upsert_day(df_all, day_results):
    """Update/append daily records."""
    date_str = day_results["date"]
    if "date" in df_all.columns and (df_all["date"] == date_str).any():
        for key, val in day_results.items():
            df_all.loc[df_all["date"] == date_str, key] = val
        return df_all
    return pd.concat([df_all, pd.DataFrame([day_results])], ignore_index=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not CLIENT_ID or not CLIENT_SECRET:
        if os.path.exists(OUTPUT_FILE):
            try:
                os.utime(OUTPUT_FILE, None)
                print("  Credentials missing; touched existing CSV.")
                return
            except Exception:
                pass
        print("  Error: OpenSky credentials are not configured.")
        sys.exit(1)

    token = get_access_token()
    if not token:
        sys.exit(1)

    df_existing, existing_dates = load_existing_data()
    if existing_dates:
        print(f"  Existing data: {len(existing_dates)}  days")

    current_date = START_DATE
    new_records = []

    while current_date < END_DATE:
        date_str = current_date.strftime("%Y-%m-%d")

        if date_str in existing_dates:
            current_date += timedelta(days=1)
            continue

        print(f"  {date_str} ...", end=" ", flush=True)
        day_results, token, err = fetch_day_counts(token, current_date)

        if err == "LIMIT_REACHED":
            print("\n  Rate limit reached. Aborting.")
            sys.exit(2)
        if err == "AUTH_EXPIRED":
            sys.exit(3)

        new_records.append(day_results)
        print(f"{day_results['hnd_nrt_movements']} flights")

        current_date += timedelta(days=1)

    if new_records:
        df_new = pd.DataFrame(new_records)
        if not df_existing.empty:
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
    else:
        df_all = df_existing

    if not df_all.empty:
        today_str = datetime.now().strftime("%Y-%m-%d")
        latest_date = df_all["date"].max()
        if latest_date == today_str:
            print("\n  Today is the last day. Starting update check from yesterday.")
            today_date = datetime.now().date()
            date_cursor = today_date - timedelta(days=1)
            backfill_updates = 0

            while True:
                date_str = date_cursor.strftime("%Y-%m-%d")
                print(f"  {date_str} (backfill) ...", end=" ", flush=True)

                day_results, token, err = fetch_day_counts(token, date_cursor)
                if err == "LIMIT_REACHED":
                    print("\n  Rate limit reached. Aborting.")
                    break
                if err == "AUTH_EXPIRED":
                    break

                if day_differs(df_all, day_results):
                    df_all = upsert_day(df_all, day_results)
                    backfill_updates += 1
                    print("Updates found")
                    date_cursor -= timedelta(days=1)
                    continue

                print("No updates")
                if date_cursor >= today_date:
                    break
                date_cursor += timedelta(days=1)
                if date_cursor > today_date:
                    break

            if backfill_updates:
                print(f"  Backfill updates: {backfill_updates} days")

    if not df_all.empty:
        df_all.to_csv(OUTPUT_FILE, index=False)
        added_days = len(new_records)
        print(f"\n  Saved: {OUTPUT_FILE} ({added_days} days added, total {len(df_all)} days)")
    else:
        print("\n  No new data.")
        if os.path.exists(OUTPUT_FILE):
            try:
                os.utime(OUTPUT_FILE, None)
                print("  Touched existing CSV to refresh mtime.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
