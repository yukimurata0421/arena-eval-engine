from datetime import datetime

import pandas as pd


def parse_date(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def prompt_intervention_date(default_date: str):
    while True:
        target_date = input(f"Intervention date [e.g., {default_date}]: ").strip()
        if not target_date:
            target_date = default_date
        d = parse_date(target_date)
        if d:
            return pd.Timestamp(d)
        print(" Invalid date format.")


def prompt_phase_dates(default_labels=None):
    if default_labels is None:
        default_labels = {}

    print("\n" + "=" * 50 + "\n Phase setup mode\n" + "=" * 50)
    phases = []

    while True:
        base_date_in = input("1. Baseline start date (YYYY-MM-DD or YYYY/MM/DD): ").strip()
        base_dt = parse_date(base_date_in)
        if base_dt:
            phases.append({"date": base_dt.strftime("%Y-%m-%d"), "name": "Initial Baseline"})
            break
        print(" Format error.")

    print("\n2. Add intervention dates (e.g., 2026-01-14,airspy_introduce). Type 'done' to finish.")
    while True:
        entry = input("Intervention date and name (YYYY-MM-DD or YYYY/MM/DD,Name): ").strip()
        if entry.lower() == "done":
            if len(phases) > 1:
                break
            print(" At least one intervention date is required.")
            continue
        try:
            d_str, n_str = entry.split(",", 1)
            date_val = parse_date(d_str)
            if not date_val:
                raise ValueError("invalid date")
            name = n_str.strip()
            if not name:
                name = default_labels.get(date_val.strftime("%Y-%m-%d"), "unnamed_event")
            phases.append({"date": date_val.strftime("%Y-%m-%d"), "name": name})
        except Exception:
            print(" Input format error.")

    return phases
