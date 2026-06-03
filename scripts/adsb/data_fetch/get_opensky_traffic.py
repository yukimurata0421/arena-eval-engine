"""
get_opensky_traffic.py

Fetch daily arrival/departure counts for Haneda/Narita from the OpenSky API.

Credentials:
  Environment variables are preferred:
    OPENSKY_CLIENT_ID
    OPENSKY_CLIENT_SECRET

  Alternative path (PowerShell):
    $env:OPENSKY_CREDENTIALS_JSON = "<path>\\opensky_credentials.json"

  JSON key examples (optional fallback file):
    {"clientId":"xxx","clientSecret":"yyy"}
    {"client_id":"xxx","client_secret":"yyy"}
"""


import requests
import pandas as pd
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


from arena.lib.paths import DATA_DIR, SCRIPTS_ROOT

from arena.log import get_script_logger


log = get_script_logger(__name__)
DEFAULT_CRED_PATH = SCRIPTS_ROOT / "secrets" / "opensky_credentials.json"
CRED_PATH = Path(os.environ.get("OPENSKY_CREDENTIALS_JSON", str(DEFAULT_CRED_PATH)))


def load_opensky_credentials(cred_path: Path):
    """
    Resolve OpenSky credentials and return (client_id, client_secret).
    Priority:
      1) OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET
      2) OPENSKY_CREDENTIALS_JSON file (clientId/clientSecret or client_id/client_secret)
    """
    env_client_id = str(os.environ.get("OPENSKY_CLIENT_ID", "")).strip()
    env_client_secret = str(os.environ.get("OPENSKY_CLIENT_SECRET", "")).strip()
    if env_client_id and env_client_secret:
        return env_client_id, env_client_secret
    if env_client_id or env_client_secret:
        log.info("Error: Please set both OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET.")
        return "", ""

    if not cred_path.exists():
        log.info("Error: OpenSky credentials not found.")
        log.info(" Action:")
        log.info(" - set OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET")
        log.info(f" - or set OPENSKY_CREDENTIALS_JSON (currently: {cred_path})")
        return "", ""

    try:
        obj = json.loads(cred_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.info(f" Error: Failed to load credentials.json: {cred_path} ({e})")
        return "", ""

    client_id = (obj.get("clientId") or obj.get("client_id") or "").strip()
    client_secret = (obj.get("clientSecret") or obj.get("client_secret") or "").strip()

    if not client_id or not client_secret:
        log.info(f" Error: clientId/clientSecret missing in credentials.json: {cred_path}")
        return "", ""

    return client_id, client_secret


CLIENT_ID, CLIENT_SECRET = load_opensky_credentials(CRED_PATH)

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
BASE_URL  = "https://opensky-network.org/api/flights"
AIRPORTS  = {"HND": "RJTT", "NRT": "RJAA"}

FLIGHT_DATA_DIR  = str(DATA_DIR / "flight_data")
OUTPUT_FILE = os.path.join(FLIGHT_DATA_DIR, "airport_movements.csv")

START_DATE = datetime(2025, 12, 1)
TODAY      = datetime.now().date()

MAX_RETRIES      = 3
RETRY_WAIT_SEC   = 10
REQUEST_INTERVAL = 2
DEFAULT_REFRESH_DAYS = 7
DEFAULT_INCLUDE_TODAY = True
DEFAULT_STABLE_DAYS_LAG = 2
DEFAULT_MIN_DAILY_MOVEMENTS = 700
DEFAULT_SHARD_HOURS = 12
DEFAULT_MAX_RUNTIME_SEC = 240
DEFAULT_REQUEST_TIMEOUT_SEC = 15
DEFAULT_AUTO_REPAIR_LOOKBACK_DAYS = 14
DEFAULT_MAX_AUTO_REPAIR_DATES = 3


def parse_int_env(name, default, min_value=0):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= min_value else default


def budget_remaining_sec(deadline_monotonic):
    if deadline_monotonic is None:
        return None
    return deadline_monotonic - time.monotonic()


def budget_exceeded(deadline_monotonic):
    rem = budget_remaining_sec(deadline_monotonic)
    return rem is not None and rem <= 0


def sleep_with_budget(seconds, deadline_monotonic):
    if seconds <= 0:
        return True

    rem = budget_remaining_sec(deadline_monotonic)
    if rem is not None and rem <= 0:
        return False

    sleep_sec = min(seconds, rem) if rem is not None else seconds
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return not budget_exceeded(deadline_monotonic)


def get_access_token():
    """Get OAuth2 token."""
    if not CLIENT_ID or not CLIENT_SECRET:
        log.info("Error: OpenSky credentials not configured.")
        log.info(f" Expected path: {CRED_PATH}")
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
            log.info("Authentication successful")
        return token

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", "unknown")
        log.info(f" Authentication error (HTTP {status}): {e}")
        return None
    except Exception as e:
        log.info(f" Authentication error: {e}")
        return None


def get_flight_records(
    token,
    icao,
    start_ts,
    end_ts,
    mode,
    *,
    max_retries,
    retry_wait_sec,
    request_timeout_sec,
    deadline_monotonic,
):
    """Fetch flights list for a given airport/direction (with retries)."""
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"airport": icao, "begin": int(start_ts), "end": int(end_ts)}

    for attempt in range(max_retries):
        if budget_exceeded(deadline_monotonic):
            return "TIME_BUDGET_EXCEEDED"

        timeout = request_timeout_sec
        rem = budget_remaining_sec(deadline_monotonic)
        if rem is not None:
            if rem <= 1:
                return "TIME_BUDGET_EXCEEDED"
            timeout = max(1, min(request_timeout_sec, int(rem)))

        try:
            res = requests.get(
                f"{BASE_URL}/{mode}",
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list):
                    return data
                return []
            elif res.status_code == 404:
                return []
            elif res.status_code == 429:
                return "LIMIT_REACHED"
            elif res.status_code == 401:
                return "AUTH_EXPIRED"
            else:
                log.info(f"\n    HTTP {res.status_code} ({icao}/{mode})")
                if attempt < max_retries - 1:
                    if not sleep_with_budget(retry_wait_sec, deadline_monotonic):
                        return "TIME_BUDGET_EXCEEDED"
                    continue
                return []

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                if not sleep_with_budget(retry_wait_sec, deadline_monotonic):
                    return "TIME_BUDGET_EXCEEDED"
                continue
            return []
        except Exception:
            return []

    return []


def unique_flight_count(records):
    """
    Count unique flights defensively.
    - Keep API-call count low by deduplicating when time-sharded queries are used.
    """
    if not records:
        return 0

    uniq = set()
    for r in records:
        if not isinstance(r, dict):
            continue
        key = (
            str(r.get("icao24", "")),
            int(r.get("firstSeen", 0) or 0),
            int(r.get("lastSeen", 0) or 0),
            str(r.get("estDepartureAirport", "")),
            str(r.get("estArrivalAirport", "")),
        )
        uniq.add(key)
    return len(uniq)


def split_time_ranges(start_ts, end_ts, shard_hours):
    step = max(1, int(shard_hours)) * 3600
    cur = int(start_ts)
    end_ts = int(end_ts)
    out = []
    while cur < end_ts:
        nxt = min(cur + step, end_ts)
        out.append((cur, nxt))
        cur = nxt
    return out


def fetch_day_counts(
    token,
    day_date,
    min_daily_movements,
    stable_days_lag,
    shard_hours,
    *,
    max_retries,
    retry_wait_sec,
    request_timeout_sec,
    request_interval_sec,
    deadline_monotonic,
):
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

    day_records = defaultdict(list)
    for name, icao in AIRPORTS.items():
        for mode in ["arrival", "departure"]:
            recs = get_flight_records(
                token,
                icao,
                s_ts,
                e_ts,
                mode,
                max_retries=max_retries,
                retry_wait_sec=retry_wait_sec,
                request_timeout_sec=request_timeout_sec,
                deadline_monotonic=deadline_monotonic,
            )

            if recs == "TIME_BUDGET_EXCEEDED":
                return None, token, "TIME_BUDGET_EXCEEDED"

            if recs == "LIMIT_REACHED":
                return None, token, "LIMIT_REACHED"

            elif recs == "AUTH_EXPIRED":
                log.info("\nToken expired. Refreshing...")
                token = get_access_token()
                if not token:
                    return None, token, "AUTH_EXPIRED"

                recs = get_flight_records(
                    token,
                    icao,
                    s_ts,
                    e_ts,
                    mode,
                    max_retries=max_retries,
                    retry_wait_sec=retry_wait_sec,
                    request_timeout_sec=request_timeout_sec,
                    deadline_monotonic=deadline_monotonic,
                )
                if recs == "LIMIT_REACHED":
                    return None, token, "LIMIT_REACHED"
                if recs == "TIME_BUDGET_EXCEEDED":
                    return None, token, "TIME_BUDGET_EXCEEDED"
                if recs == "AUTH_EXPIRED":
                    return None, token, "AUTH_EXPIRED"

            day_records[(name, mode)] = recs
            count = unique_flight_count(recs)
            day_results[f"{name.lower()}_{mode[:3]}"] = count
            total += count
            if not sleep_with_budget(request_interval_sec, deadline_monotonic):
                return None, token, "TIME_BUDGET_EXCEEDED"

    # If a stable historical day looks implausibly low, run one sharded verification.
    # This limits additional API calls only to suspicious days.
    if day_dt.date() <= (TODAY - timedelta(days=stable_days_lag)) and total < min_daily_movements:
        verified_total = 0
        for name, icao in AIRPORTS.items():
            for mode in ["arrival", "departure"]:
                merged = []
                for ss, ee in split_time_ranges(s_ts, e_ts, shard_hours):
                    recs = get_flight_records(
                        token,
                        icao,
                        ss,
                        ee,
                        mode,
                        max_retries=max_retries,
                        retry_wait_sec=retry_wait_sec,
                        request_timeout_sec=request_timeout_sec,
                        deadline_monotonic=deadline_monotonic,
                    )
                    if recs == "TIME_BUDGET_EXCEEDED":
                        return None, token, "TIME_BUDGET_EXCEEDED"
                    if recs == "LIMIT_REACHED":
                        return None, token, "LIMIT_REACHED"
                    if recs == "AUTH_EXPIRED":
                        token = get_access_token()
                        if not token:
                            return None, token, "AUTH_EXPIRED"
                        recs = get_flight_records(
                            token,
                            icao,
                            ss,
                            ee,
                            mode,
                            max_retries=max_retries,
                            retry_wait_sec=retry_wait_sec,
                            request_timeout_sec=request_timeout_sec,
                            deadline_monotonic=deadline_monotonic,
                        )
                        if recs == "LIMIT_REACHED":
                            return None, token, "LIMIT_REACHED"
                        if recs == "TIME_BUDGET_EXCEEDED":
                            return None, token, "TIME_BUDGET_EXCEEDED"
                        if recs == "AUTH_EXPIRED":
                            return None, token, "AUTH_EXPIRED"
                    merged.extend(recs if isinstance(recs, list) else [])
                    if not sleep_with_budget(request_interval_sec, deadline_monotonic):
                        return None, token, "TIME_BUDGET_EXCEEDED"

                vcount = unique_flight_count(merged)
                day_results[f"{name.lower()}_{mode[:3]}"] = vcount
                verified_total += vcount
        total = verified_total

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


def parse_force_dates():
    """Parse comma-separated force dates from OPENSKY_FORCE_DATES (YYYY-MM-DD)."""
    raw = os.environ.get("OPENSKY_FORCE_DATES", "").strip()
    if not raw:
        return set()

    parsed = set()
    for token in raw.split(","):
        date_str = token.strip()
        if not date_str:
            continue
        try:
            parsed.add(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            log.info(f" Warning: Invalid date format for OPENSKY_FORCE_DATES: {date_str}")
    return parsed


def detect_suspicious_dates(
    df_all,
    min_daily_movements,
    stable_days_lag,
    auto_repair_lookback_days,
    max_auto_repair_dates,
):
    """
    Auto-mark obviously broken days for repair.
    Keeps API usage low by only selecting low-total stable days.
    """
    if df_all.empty or "date" not in df_all.columns or "hnd_nrt_movements" not in df_all.columns:
        return set()

    if auto_repair_lookback_days <= 0:
        return set()

    out = []
    lower_bound = TODAY - timedelta(days=auto_repair_lookback_days)
    upper_bound = TODAY - timedelta(days=stable_days_lag)
    for _, r in df_all.iterrows():
        try:
            d = datetime.strptime(str(r["date"]), "%Y-%m-%d").date()
            total = int(r["hnd_nrt_movements"])
        except Exception:
            continue
        if lower_bound <= d <= upper_bound and total < min_daily_movements:
            out.append(d)

    out = sorted(set(out), reverse=True)
    if max_auto_repair_dates > 0:
        out = out[:max_auto_repair_dates]
    return set(out)


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
    os.makedirs(FLIGHT_DATA_DIR, exist_ok=True)

    if not CLIENT_ID or not CLIENT_SECRET:
        if os.path.exists(OUTPUT_FILE):
            try:
                os.utime(OUTPUT_FILE, None)
                log.info(" mtime of existing CSV has been updated because there is no authentication information.")
                return
            except Exception:
                pass
        log.info("Error: OpenSky credentials not configured.")
        sys.exit(1)

    token = get_access_token()
    if not token:
        sys.exit(1)

    df_existing, existing_dates = load_existing_data()
    if existing_dates:
        log.info(f" Existing data: {len(existing_dates)} days")

    try:
        refresh_days = int(os.environ.get("OPENSKY_REFRESH_DAYS", str(DEFAULT_REFRESH_DAYS)))
    except ValueError:
        refresh_days = DEFAULT_REFRESH_DAYS
    refresh_days = max(refresh_days, 0)

    try:
        stable_days_lag = int(os.environ.get("OPENSKY_STABLE_DAYS_LAG", str(DEFAULT_STABLE_DAYS_LAG)))
    except ValueError:
        stable_days_lag = DEFAULT_STABLE_DAYS_LAG
    stable_days_lag = max(stable_days_lag, 0)

    try:
        min_daily_movements = int(os.environ.get("OPENSKY_MIN_DAILY_MOVEMENTS", str(DEFAULT_MIN_DAILY_MOVEMENTS)))
    except ValueError:
        min_daily_movements = DEFAULT_MIN_DAILY_MOVEMENTS
    min_daily_movements = max(min_daily_movements, 0)

    try:
        shard_hours = int(os.environ.get("OPENSKY_SHARD_HOURS", str(DEFAULT_SHARD_HOURS)))
    except ValueError:
        shard_hours = DEFAULT_SHARD_HOURS
    shard_hours = max(1, min(shard_hours, 24))

    max_runtime_sec = parse_int_env("OPENSKY_MAX_RUNTIME_SEC", DEFAULT_MAX_RUNTIME_SEC, min_value=0)
    request_timeout_sec = parse_int_env("OPENSKY_REQUEST_TIMEOUT_SEC", DEFAULT_REQUEST_TIMEOUT_SEC, min_value=1)
    max_retries = parse_int_env("OPENSKY_MAX_RETRIES", MAX_RETRIES, min_value=1)
    retry_wait_sec = parse_int_env("OPENSKY_RETRY_WAIT_SEC", RETRY_WAIT_SEC, min_value=0)
    request_interval_sec = parse_int_env("OPENSKY_REQUEST_INTERVAL_SEC", REQUEST_INTERVAL, min_value=0)
    auto_repair_lookback_days = parse_int_env(
        "OPENSKY_AUTO_REPAIR_LOOKBACK_DAYS",
        DEFAULT_AUTO_REPAIR_LOOKBACK_DAYS,
        min_value=0,
    )
    max_auto_repair_dates = parse_int_env(
        "OPENSKY_MAX_AUTO_REPAIR_DATES",
        DEFAULT_MAX_AUTO_REPAIR_DATES,
        min_value=0,
    )

    include_today = os.environ.get("OPENSKY_INCLUDE_TODAY", "1" if DEFAULT_INCLUDE_TODAY else "0").strip() not in (
        "0", "false", "False"
    )

    deadline_monotonic = None
    if max_runtime_sec > 0:
        deadline_monotonic = time.monotonic() + max_runtime_sec

    refresh_from = TODAY - timedelta(days=refresh_days)
    force_dates = {d for d in parse_force_dates() if d <= TODAY}
    auto_suspicious_dates = detect_suspicious_dates(
        df_existing,
        min_daily_movements,
        stable_days_lag,
        auto_repair_lookback_days=auto_repair_lookback_days,
        max_auto_repair_dates=max_auto_repair_dates,
    )
    force_dates |= auto_suspicious_dates

    date_cursor = START_DATE.date()
    normal_dates = []
    stop_date = TODAY if include_today else (TODAY - timedelta(days=1))
    while date_cursor <= stop_date:
        normal_dates.append(date_cursor)
        date_cursor += timedelta(days=1)

    if existing_dates:
        recent_window_start = max(START_DATE.date(), refresh_from)
        recent_dates = [d for d in normal_dates if d >= recent_window_start]
        older_forced = sorted([d for d in force_dates if d < recent_window_start], reverse=True)
        all_dates = []
        seen_dates = set()
        for d in recent_dates + older_forced:
            if d in seen_dates:
                continue
            seen_dates.add(d)
            all_dates.append(d)
    else:
        all_dates = sorted(set(normal_dates) | force_dates)
    new_records = []
    updated_days = 0

    for current_date in all_dates:
        if budget_exceeded(deadline_monotonic):
            log.info("\nThe execution time budget has been reached and the remaining dates will be carried over to the next time.")
            break

        date_str = current_date.strftime("%Y-%m-%d")
        should_refresh = current_date >= refresh_from or current_date in force_dates

        if date_str in existing_dates and not should_refresh:
            continue

        log.info(f"  {date_str} ...")
        day_results, token, err = fetch_day_counts(
            token,
            current_date,
            min_daily_movements=min_daily_movements,
            stable_days_lag=stable_days_lag,
            shard_hours=shard_hours,
            max_retries=max_retries,
            retry_wait_sec=retry_wait_sec,
            request_timeout_sec=request_timeout_sec,
            request_interval_sec=request_interval_sec,
            deadline_monotonic=deadline_monotonic,
        )

        if err == "TIME_BUDGET_EXCEEDED":
            log.info("\nThe execution time budget has been reached and the remaining dates will be carried over to the next time.")
            break
        if err == "LIMIT_REACHED":
            log.info("\nRate limit reached. Interrupting.")
            sys.exit(2)
        if err == "AUTH_EXPIRED":
            sys.exit(3)

        if date_str in existing_dates:
            if day_differs(df_existing, day_results):
                df_existing = upsert_day(df_existing, day_results)
                updated_days += 1
                log.info(f"{day_results['hnd_nrt_movements']} flights (updated)")
            else:
                log.info(f"{day_results['hnd_nrt_movements']} flights (no changes)")
        else:
            new_records.append(day_results)
            log.info(f"{day_results['hnd_nrt_movements']} flights (added)")

    if new_records:
        df_new = pd.DataFrame(new_records)
        if not df_existing.empty:
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
    else:
        df_all = df_existing

    if not df_all.empty:
        df_all.to_csv(OUTPUT_FILE, index=False)
        added_days = len(new_records)
        log.info(f"\nSaved: {OUTPUT_FILE}(added {added_days} days, updated {updated_days} days, total {len(df_all)} days)")
    else:
        log.info("\nNo new data.")
        if os.path.exists(OUTPUT_FILE):
            try:
                os.utime(OUTPUT_FILE, None)
                log.info(" mtime of existing CSV has been updated.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
