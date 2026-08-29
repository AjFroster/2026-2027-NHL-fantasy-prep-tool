#!/usr/bin/env python3
"""
fetch_nhl.py -- pull three seasons of NHL skater data from the official NHL
Stats REST API and write one tidy long-format CSV.

Standard library only. Every network response is cached to disk, so a rerun
with a warm cache never touches the network.

Usage:
    python3 fetch_nhl.py               # full pull (uses cache where present)
    python3 fetch_nhl.py --dry-run     # one page from one season, verify API
    python3 fetch_nhl.py --refresh     # bypass cache, re-fetch everything
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration -- edit these
# ---------------------------------------------------------------------------

SEASONS = ["20232024", "20242025", "20252026"]   # oldest -> newest
PROJECTION_SEASON = "20262027"
GAME_TYPE = 2          # regular season only
MIN_GP_TOTAL = 20      # across all 3 seasons, to appear in output

PAGE_LIMIT = 100       # NHL API caps `limit` at 100
CACHE_DIR = "cache"
DATA_DIR = "data"
OUT_CSV = os.path.join(DATA_DIR, "skaters_3yr.csv")

STATS_BASE = "https://api.nhle.com/stats/rest/en/skater"
LANDING_URL = "https://api-web.nhle.com/v1/player/{pid}/landing"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

MAX_RETRIES = 5
BACKOFF_BASE = 1.5     # seconds; doubles each retry
REQUEST_PAUSE = 0.05   # polite delay between live requests

# Fields we require from each endpoint. Missing field == loud failure.
SUMMARY_FIELDS = [
    "playerId", "skaterFullName", "positionCode", "seasonId", "teamAbbrevs",
    "gamesPlayed", "goals", "assists", "points", "shots", "shootingPct",
    "timeOnIcePerGame",
]
REALTIME_FIELDS = [
    "playerId", "seasonId", "teamAbbrevs", "gamesPlayed",
    "hits", "blockedShots", "takeaways", "giveaways",
]

COUNTING_SUMMARY = ["gamesPlayed", "goals", "assists", "points", "shots"]
COUNTING_REALTIME = ["gamesPlayed", "hits", "blockedShots", "takeaways", "giveaways"]


RAW_SEASON_COUNTS: dict[int, int] = {}


class SchemaError(RuntimeError):
    """Raised when the API returns something we did not expect."""


# ---------------------------------------------------------------------------
# HTTP with retry + disk cache
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 30) -> dict:
    """GET a URL, retrying on 429 / 5xx with exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429 or 500 <= e.code < 600:
                wait = BACKOFF_BASE * (2 ** attempt) + random.random()
                print(f"  HTTP {e.code} on {url[:90]}... retry in {wait:.1f}s "
                      f"({attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            wait = BACKOFF_BASE * (2 ** attempt) + random.random()
            print(f"  {type(e).__name__} on {url[:90]}... retry in {wait:.1f}s "
                  f"({attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {MAX_RETRIES} attempts: {last_err}")


def _cached_get(cache_path: str, url: str, refresh: bool) -> dict:
    """Read from disk cache, else fetch and write the cache."""
    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    payload = _http_get_json(url)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, cache_path)
    time.sleep(REQUEST_PAUSE)
    return payload


def build_stats_url(endpoint: str, season: str, start: int) -> str:
    params = {
        "isAggregate": "false",
        "isGame": "false",
        "start": str(start),
        "limit": str(PAGE_LIMIT),
        "sort": json.dumps([{"property": "playerId", "direction": "ASC"}]),
        "cayenneExp": f"gameTypeId={GAME_TYPE} and seasonId={season}",
    }
    return f"{STATS_BASE}/{endpoint}?" + urllib.parse.urlencode(params)


def fetch_endpoint_season(endpoint: str, season: str, refresh: bool,
                          required: list[str], single_page: bool = False) -> list[dict]:
    """Paginate one endpoint for one season. Returns the raw row list."""
    rows: list[dict] = []
    start = 0
    total = None
    while True:
        cache_path = os.path.join(CACHE_DIR, f"{endpoint}_{season}_{start}.json")
        url = build_stats_url(endpoint, season, start)
        payload = _cached_get(cache_path, url, refresh)

        if "data" not in payload:
            raise SchemaError(f"{endpoint}/{season} start={start}: no 'data' key "
                              f"in response (keys={list(payload)})")
        page = payload["data"]
        if not page:
            raise SchemaError(f"{endpoint}/{season} start={start}: empty 'data' array")
        if "total" not in payload:
            raise SchemaError(f"{endpoint}/{season}: response has no 'total'")

        missing = [f for f in required if f not in page[0]]
        if missing:
            raise SchemaError(f"{endpoint}/{season}: rows missing expected fields "
                              f"{missing}; got {sorted(page[0])}")

        rows.extend(page)
        total = payload["total"]
        if single_page:
            break
        start += PAGE_LIMIT
        if start >= total:
            break

    if not single_page and len(rows) != total:
        raise SchemaError(f"{endpoint}/{season}: fetched {len(rows)} rows but "
                          f"API reported total={total}")
    print(f"  {endpoint:9s} {season}: {len(rows)} rows (total={total})")
    return rows


# ---------------------------------------------------------------------------
# Trade / duplicate-row handling
# ---------------------------------------------------------------------------

def _split_teams(value) -> list[str]:
    """teamAbbrevs may be 'STL' or a comma-joined 'LAK,TBL'."""
    if not value:
        return []
    return [t.strip() for t in str(value).replace("/", ",").split(",") if t.strip()]


def aggregate_rows(rows: list[dict], counting: list[str],
                   endpoint_name: str) -> dict[tuple[int, int], dict]:
    """
    Collapse to one record per (playerId, seasonId).

    Counting stats are summed. timeOnIcePerGame is games-weighted. Team
    abbreviations are collected in first-seen order. Idempotent: a player with
    a single row comes out unchanged.
    """
    out: dict[tuple[int, int], dict] = {}
    for r in rows:
        key = (int(r["playerId"]), int(r["seasonId"]))
        rec = out.get(key)
        if rec is None:
            rec = {
                "playerId": key[0],
                "seasonId": key[1],
                "_teams": [],
                "_toi_seconds_total": 0.0,
                "_rowcount": 0,
            }
            for c in counting:
                rec[c] = 0
            out[key] = rec

        gp = int(r.get("gamesPlayed") or 0)
        for c in counting:
            rec[c] += int(r.get(c) or 0)

        toi_pg = r.get("timeOnIcePerGame")
        if toi_pg is not None:
            rec["_toi_seconds_total"] += float(toi_pg) * gp

        for t in _split_teams(r.get("teamAbbrevs")):
            if t not in rec["_teams"]:
                rec["_teams"].append(t)

        rec["_rowcount"] += 1

        # Identity fields: keep the first non-null we see.
        for f in ("skaterFullName", "positionCode", "shootsCatches"):
            if f in r and r[f] and f not in rec:
                rec[f] = r[f]

    for rec in out.values():
        gp = rec.get("gamesPlayed", 0)
        # Weighted-average TOI/GP, seconds -> minutes.
        rec["toiPerGame"] = (rec["_toi_seconds_total"] / gp / 60.0) if gp else 0.0
        rec["teams"] = "/".join(rec["_teams"])
        rec["changedTeams"] = len(rec["_teams"]) > 1
    print(f"  {endpoint_name}: {len(rows)} raw rows -> {len(out)} player-seasons")
    return out


# ---------------------------------------------------------------------------
# Player metadata (landing endpoint)
# ---------------------------------------------------------------------------

def _name_of(value) -> str:
    if isinstance(value, dict):
        return value.get("default", "")
    return value or ""


def fetch_landing(player_ids: list[int], refresh: bool) -> tuple[dict[int, dict], list[int]]:
    """One landing call per player, cached. Returns (metadata, failed_ids)."""
    meta: dict[int, dict] = {}
    failed: list[int] = []
    total = len(player_ids)
    print(f"Fetching player metadata for {total} skaters...")
    for i, pid in enumerate(player_ids, 1):
        cache_path = os.path.join(CACHE_DIR, f"landing_{pid}.json")
        try:
            payload = _cached_get(cache_path, LANDING_URL.format(pid=pid), refresh)
            meta[pid] = {
                "birthDate": payload.get("birthDate"),
                "shootsCatches": payload.get("shootsCatches"),
                "currentTeam": payload.get("currentTeamAbbrev"),
                "position": payload.get("position"),
                "fullName": (_name_of(payload.get("firstName")) + " " +
                             _name_of(payload.get("lastName"))).strip(),
            }
            if not meta[pid]["birthDate"]:
                print(f"  WARN: player {pid} landing has no birthDate; "
                      f"age adjustment will be skipped", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 -- one bad player must not kill the run
            print(f"  WARN: landing failed for player {pid}: {e}; "
                  f"age left null, age adjustment skipped", file=sys.stderr)
            failed.append(pid)
        if i % 100 == 0 or i == total:
            print(f"  ...{i}/{total}")
    return meta, failed


def age_on_feb1(birth_date: str | None, projection_season: str) -> float | None:
    """Age in years (fractional) on Feb 1 of the projection season's calendar year."""
    if not birth_date:
        return None
    try:
        bd = dt.date.fromisoformat(birth_date[:10])
    except ValueError:
        return None
    end_year = int(projection_season[4:])   # 20262027 -> 2027
    target = dt.date(end_year, 2, 1)
    return round((target - bd).days / 365.2425, 2)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def position_group(position_code: str) -> str:
    return "D" if position_code == "D" else "F"


def per60(stat: float, toi_per_game: float, gp: int) -> float:
    minutes = toi_per_game * gp
    return round(stat / (minutes / 60.0), 3) if minutes > 0 else 0.0


CSV_HEADER = [
    "playerId", "playerName", "position", "positionGroup", "seasonId", "teams",
    "changedTeams", "gamesPlayed", "goals", "assists", "shots", "hits",
    "blockedShots", "takeaways", "giveaways", "toiPerGame", "shootingPct",
    "goalsPer60", "assistsPer60", "shotsPer60", "hitsPer60", "blocksPer60",
    "age", "shootsCatches", "currentTeam",
]


def build_rows(refresh: bool) -> list[dict]:
    summary_agg: dict = {}
    realtime_agg: dict = {}

    for season in SEASONS:
        print(f"Season {season}:")
        s_rows = fetch_endpoint_season("summary", season, refresh, SUMMARY_FIELDS)
        r_rows = fetch_endpoint_season("realtime", season, refresh, REALTIME_FIELDS)
        # Aggregate each endpoint independently BEFORE joining. The two
        # endpoints do not always split traded players the same way (realtime
        # often returns one row with 'LAK,TBL'), so collapsing first is what
        # keeps the join one-to-one.
        summary_agg.update(aggregate_rows(s_rows, COUNTING_SUMMARY, f"summary/{season}"))
        realtime_agg.update(aggregate_rows(r_rows, COUNTING_REALTIME, f"realtime/{season}"))

    missing_rt = [k for k in summary_agg if k not in realtime_agg]
    if missing_rt:
        raise SchemaError(f"{len(missing_rt)} player-seasons present in summary but "
                          f"absent from realtime, e.g. {missing_rt[:5]}")

    gp_mismatch = 0
    records = []
    for key, s in sorted(summary_agg.items()):
        rt = realtime_agg[key]
        if s["gamesPlayed"] != rt["gamesPlayed"]:
            gp_mismatch += 1
        pos = s.get("positionCode", "")
        if pos == "G":
            continue
        teams = s["teams"] or rt["teams"]
        changed = s["changedTeams"] or rt["changedTeams"]
        records.append({
            "playerId": key[0],
            "playerName": s.get("skaterFullName", ""),
            "position": pos,
            "positionGroup": position_group(pos),
            "seasonId": key[1],
            "teams": teams,
            "changedTeams": changed,
            "gamesPlayed": s["gamesPlayed"],
            "goals": s["goals"],
            "assists": s["assists"],
            "shots": s["shots"],
            "hits": rt["hits"],
            "blockedShots": rt["blockedShots"],
            "takeaways": rt["takeaways"],
            "giveaways": rt["giveaways"],
            "toiPerGame": round(s["toiPerGame"], 3),
        })
    if gp_mismatch:
        print(f"  NOTE: {gp_mismatch} player-seasons had a gamesPlayed mismatch "
              f"between summary and realtime; summary is authoritative.")

    # Per-season counts before the MIN_GP_TOTAL filter -- this is the number
    # the 850-1000 acceptance band refers to (the API's full skater pool).
    global RAW_SEASON_COUNTS
    RAW_SEASON_COUNTS = defaultdict(int)
    for r in records:
        RAW_SEASON_COUNTS[r["seasonId"]] += 1

    # Drop players below the career GP floor.
    gp_total: dict[int, int] = defaultdict(int)
    for r in records:
        gp_total[r["playerId"]] += r["gamesPlayed"]
    kept = [r for r in records if gp_total[r["playerId"]] >= MIN_GP_TOTAL]
    dropped_players = len({r["playerId"] for r in records}) - len({r["playerId"] for r in kept})
    print(f"Dropped {dropped_players} players under MIN_GP_TOTAL={MIN_GP_TOTAL}.")

    # Drop zero-game / zero-TOI rows (they carry no rate information).
    before = len(kept)
    kept = [r for r in kept if r["gamesPlayed"] > 0 and r["toiPerGame"] > 0]
    if before != len(kept):
        print(f"Dropped {before - len(kept)} rows with gamesPlayed==0 or toiPerGame==0.")

    # Player metadata + derived rates.
    pids = sorted({r["playerId"] for r in kept})
    meta, failed = fetch_landing(pids, refresh)
    if failed:
        print(f"  {len(failed)} landing lookups failed: {failed[:10]}"
              f"{' ...' if len(failed) > 10 else ''}")

    for r in kept:
        m = meta.get(r["playerId"], {})
        r["age"] = age_on_feb1(m.get("birthDate"), PROJECTION_SEASON)
        r["shootsCatches"] = m.get("shootsCatches") or ""
        r["currentTeam"] = m.get("currentTeam") or ""
        if m.get("fullName") and not r["playerName"]:
            r["playerName"] = m["fullName"]
        gp, toi = r["gamesPlayed"], r["toiPerGame"]
        # Rates are always recomputed from summed totals, never averaged.
        r["shootingPct"] = round(100.0 * r["goals"] / r["shots"], 3) if r["shots"] else 0.0
        r["goalsPer60"] = per60(r["goals"], toi, gp)
        r["assistsPer60"] = per60(r["assists"], toi, gp)
        r["shotsPer60"] = per60(r["shots"], toi, gp)
        r["hitsPer60"] = per60(r["hits"], toi, gp)
        r["blocksPer60"] = per60(r["blockedShots"], toi, gp)

    kept.sort(key=lambda r: (r["playerName"], r["seasonId"]))
    return kept


def write_csv(rows: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------

def acceptance_checks(rows: list[dict]) -> bool:
    print("\n" + "=" * 68)
    print("ACCEPTANCE CHECKS")
    print("=" * 68)
    ok = True

    seen = set()
    dupes = []
    for r in rows:
        k = (r["playerId"], r["seasonId"])
        if k in seen:
            dupes.append(k)
        seen.add(k)
    if dupes:
        ok = False
        print(f"FAIL  duplicate (playerId, seasonId) pairs: {len(dupes)} e.g. {dupes[:5]}")
    else:
        print(f"PASS  no duplicate (playerId, seasonId) pairs ({len(rows)} rows)")

    bad = [r for r in rows if not (r["gamesPlayed"] > 0 and r["toiPerGame"] > 0)]
    if bad:
        ok = False
        print(f"FAIL  {len(bad)} rows with gamesPlayed<=0 or toiPerGame<=0")
    else:
        print("PASS  every row has gamesPlayed > 0 and toiPerGame > 0")

    per_season: dict[int, int] = defaultdict(int)
    for r in rows:
        per_season[r["seasonId"]] += 1
    for season in sorted(per_season):
        n_raw = RAW_SEASON_COUNTS.get(season, per_season[season])
        n_kept = per_season[season]
        if 850 <= n_raw <= 1000:
            print(f"PASS  season {season}: {n_raw} skaters from the API "
                  f"(expected 850-1000); {n_kept} rows written after the "
                  f"MIN_GP_TOTAL={MIN_GP_TOTAL} filter")
        else:
            print(f"WARN  season {season}: {n_raw} skaters from the API "
                  f"(expected roughly 850-1000); {n_kept} rows written")

    n_changed = sum(1 for r in rows if r["changedTeams"])
    print(f"INFO  {n_changed} player-seasons flagged changedTeams=True")
    n_no_age = len({r["playerId"] for r in rows if r["age"] in (None, "")})
    print(f"INFO  {n_no_age} players have no age (landing lookup failed or no birthDate)")

    sample = [r for r in rows if r["playerName"] == "Cale Makar"]
    if not sample:
        sample = [r for r in rows if "Makar" in r["playerName"]]
    print("\nSample player (eyeball against NHL.com):")
    if sample:
        for r in sorted(sample, key=lambda x: x["seasonId"]):
            print(f"  {json.dumps({k: r.get(k) for k in CSV_HEADER}, indent=None)}")
    else:
        print("  WARN: Cale Makar not found in output")
        ok = False
    print("=" * 68)
    return ok


# ---------------------------------------------------------------------------

def dry_run(refresh: bool) -> None:
    season = SEASONS[-1]
    print(f"DRY RUN: one page of skater/summary for season {season}\n")
    rows = fetch_endpoint_season("summary", season, refresh, SUMMARY_FIELDS,
                                 single_page=True)
    print(f"\nFirst row:\n{json.dumps(rows[0], indent=2)}")
    print(f"\nAPI is responding. {len(rows)} rows in page 1.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="bypass the disk cache and re-fetch every response")
    ap.add_argument("--dry-run", action="store_true",
                    help="pull a single page from one season and stop")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        if args.dry_run:
            dry_run(args.refresh)
            return 0
        rows = build_rows(args.refresh)
        if not rows:
            raise SchemaError("no rows survived filtering -- refusing to write an empty CSV")
        write_csv(rows)
        return 0 if acceptance_checks(rows) else 1
    except SchemaError as e:
        print(f"\nSCHEMA ERROR: {e}", file=sys.stderr)
        print("Refusing to write a partial CSV.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
