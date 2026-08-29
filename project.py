#!/usr/bin/env python3
"""
project.py -- Marcel-style 2026-27 fantasy projections from data/skaters_3yr.csv.

Weighted recent seasons -> regression to a positional mean -> age curve ->
separate games-played projection -> trend metrics, tiers and VORP.

Standard library only, no network access.

Usage:
    python3 project.py
    python3 project.py --teams 12 --f-slots 9 --d-slots 4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

IN_CSV = os.path.join("data", "skaters_3yr.csv")
OUT_CSV = os.path.join("data", "projections_2026_27.csv")

SEASONS = ["20232024", "20242025", "20252026"]   # oldest -> newest
PROJECTION_SEASON = "20262027"

# Season weights, newest -> oldest.
SEASON_WEIGHT = {"20252026": 5.0, "20242025": 4.0, "20232024": 3.0}

# Default fantasy scoring. Only used for the baseline numbers written to the
# CSV (tiers, VORP, trend metrics) -- the dashboard re-applies its own weights
# to the projected stat totals, so changing scoring never needs a rerun.
DEFAULT_WEIGHTS = {
    "goals": 2.0,
    "assists": 1.5,
    "shots": 0.15,
    "hits": 0.2,
    "blocks": 0.35,
}

# Phantom games of league-average production added to each player's history.
# Lower = the stat stabilizes faster and we trust the player's own history more.
PHANTOM_GP = {
    "goals": 24.0,     # noisiest, driven by shooting % variance
    "assists": 20.0,   # depends on linemates
    "shots": 12.0,     # stabilizes quickly, largely volume/role
    "hits": 8.0,       # strongly role-driven, high year-over-year correlation
    "blocks": 8.0,     # same -- a role stat, very sticky
}

SHPCT_PRIOR_SHOTS = 250.0    # phantom shots at league sh% for the sh% prior
GOAL_BLEND = 0.5             # 50/50 blend of rate model and sh%-based estimate

GP_PROJ_ANCHOR = 72.0        # league-ish baseline games
GP_PROJ_ANCHOR_WEIGHT = 0.25
INJURY_GP_THRESHOLD = 55
CONFIDENCE_GP = 60
TIER_MIN_GP = 40             # only players projected this many games get a tier

AGE_CLAMP = (0.60, 1.25)

# Age multipliers, forwards. Ages 19-27 are measured: the year-over-year change
# in era-adjusted points per 60 across 6,912 NHL player-seasons of cached career
# history, smoothed with an n-weighted cubic, then converted into "next season
# vs the 5/4/3-weighted level of the last three seasons". Per *60*, deliberately
# -- the ice-time half of early-career growth is handled by the TOI term below,
# and counting both would double-count it.
#
# Ages 28+ keep the originally specified decline, re-anchored at 27 for
# continuity. The same measurement makes decline look far gentler (0.93 at 34
# rather than 0.83), but it cannot be trusted: the career histories come from
# players active in 2023-26, so anyone who declined and retired is structurally
# absent from the sample. Measured ascent, specified decline.
AGE_MULT_F = {
    19: 1.167, 20: 1.167, 21: 1.151, 22: 1.127, 23: 1.102,
    24: 1.079, 25: 1.057, 26: 1.037, 27: 1.019, 28: 1.004,
    29: 0.988, 30: 0.973, 31: 0.942, 32: 0.912, 33: 0.881,
    34: 0.830, 35: 0.779, 36: 0.728, 37: 0.678, 38: 0.627,
    39: 0.576, 40: 0.525,
}

# Ice-time term. A player's own TOI trend is extrapolated from the centroid of
# the 5/4/3 weights (1.83 seasons back) to the projection season, damped because
# ice-time trends regress hard. Change in TOI/GP correlates r=0.47 with change
# in production across all transitions, r=0.50 in a player's first five seasons
# -- the strongest single predictor in the history.
TOI_LAG = 22.0 / 12.0        # centroid of the 5/4/3 season weights
TOI_TREND_DAMPING = 0.5      # take half the extrapolation
TOI_MULT_CLAMP = (0.90, 1.15)
COMBINED_CLAMP = (0.55, 1.35)   # age x ice time, so the two cannot compound wildly

STATS = ["goals", "assists", "shots", "hits", "blocks"]
# stat name -> column name in skaters_3yr.csv
STAT_COL = {
    "goals": "goals", "assists": "assists", "shots": "shots",
    "hits": "hits", "blocks": "blockedShots",
}


class DataError(RuntimeError):
    """Input CSV is not what we expect."""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _num(v, cast=float, default=0.0):
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def load_players(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        raise DataError(f"{path} not found -- run fetch_nhl.py first")
    required = {"playerId", "playerName", "position", "positionGroup", "seasonId",
                "gamesPlayed", "goals", "assists", "shots", "hits", "blockedShots",
                "toiPerGame", "age", "teams", "currentTeam"}
    players: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DataError(f"{path} is missing expected columns: {sorted(missing)}")
        n = 0
        for row in reader:
            n += 1
            pid = int(row["playerId"])
            season = row["seasonId"]
            if season not in SEASON_WEIGHT:
                raise DataError(f"unexpected seasonId {season} in {path}")
            p = players.setdefault(pid, {
                "playerId": pid,
                "playerName": row["playerName"],
                "position": row["position"],
                "positionGroup": row["positionGroup"],
                "currentTeam": row.get("currentTeam") or "",
                "age": _num(row.get("age"), float, None),
                "seasons": {},
            })
            if season in p["seasons"]:
                raise DataError(f"duplicate (playerId={pid}, seasonId={season}) in {path}")
            p["seasons"][season] = {
                "gamesPlayed": _num(row["gamesPlayed"], int, 0),
                "goals": _num(row["goals"], int, 0),
                "assists": _num(row["assists"], int, 0),
                "shots": _num(row["shots"], int, 0),
                "hits": _num(row["hits"], int, 0),
                "blocks": _num(row["blockedShots"], int, 0),
                "toiPerGame": _num(row["toiPerGame"], float, 0.0),
                "teams": row.get("teams") or "",
            }
            # Most recent row wins for identity/team, seasons are read in order.
            p["position"] = row["position"] or p["position"]
            if not p["currentTeam"]:
                p["currentTeam"] = row.get("teams") or ""
    if n == 0:
        raise DataError(f"{path} has no data rows")
    print(f"Loaded {n} player-season rows for {len(players)} players from {path}")
    return players


# ---------------------------------------------------------------------------
# Step 1 -- weighted rate, per stat
# ---------------------------------------------------------------------------

def weighted_num_den(p: dict, stat: str) -> tuple[float, float]:
    """5/4/3 weighted stat total and games total. A missing season adds 0 to both."""
    num = den = 0.0
    for season, w in SEASON_WEIGHT.items():
        s = p["seasons"].get(season)
        if not s:
            continue
        num += w * s[stat]
        den += w * s["gamesPlayed"]
    return num, den


def league_rates(players: dict[int, dict]) -> tuple[dict, dict]:
    """Per-position-group league rate per game (same 5/4/3 weighting), plus league sh%."""
    num = defaultdict(float)
    den = defaultdict(float)
    g_num = defaultdict(float)
    s_num = defaultdict(float)
    for p in players.values():
        pg = p["positionGroup"]
        for stat in STATS:
            n, d = weighted_num_den(p, stat)
            num[(pg, stat)] += n
            den[(pg, stat)] += d
        gn, _ = weighted_num_den(p, "goals")
        sn, _ = weighted_num_den(p, "shots")
        g_num[pg] += gn
        s_num[pg] += sn

    rates = {}
    for (pg, stat), n in num.items():
        d = den[(pg, stat)]
        if d <= 0:
            raise DataError(f"league denominator is zero for {pg}/{stat}")
        rates[(pg, stat)] = n / d
    shpct = {pg: (g_num[pg] / s_num[pg]) for pg in g_num if s_num[pg] > 0}
    return rates, shpct


# ---------------------------------------------------------------------------
# Step 4 -- age curve
# ---------------------------------------------------------------------------

def age_multiplier(age: float | None, position_group: str, stat: str) -> float:
    """
    Age at Feb 1 of the projection season -> per-60 production multiplier.

    Linear interpolation over AGE_MULT_F. Defensemen are evaluated one year
    younger, which shifts the whole curve a year later for them, and take half
    the decline on hits/blocks, which hold up far longer than scoring does.
    Result clamped to [0.60, 1.25].
    """
    if age is None:
        return 1.0
    a = age - (1.0 if position_group == "D" else 0.0)

    lo, hi = min(AGE_MULT_F), max(AGE_MULT_F)
    a = max(float(lo), min(float(hi), a))
    floor_age = int(a)
    if floor_age >= hi:
        m = AGE_MULT_F[int(hi)]
    else:
        frac = a - floor_age
        m = (AGE_MULT_F[floor_age] * (1 - frac) + AGE_MULT_F[floor_age + 1] * frac)

    if position_group == "D" and stat in ("hits", "blocks") and m < 1.0:
        m = 1.0 - (1.0 - m) * 0.5

    return max(AGE_CLAMP[0], min(AGE_CLAMP[1], m))


# ---------------------------------------------------------------------------
# Step 7 helpers
# ---------------------------------------------------------------------------

def ols_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope; 0.0 when there is nothing to fit."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def coeff_of_variation(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / n
    return math.sqrt(var) / mean


def fantasy_points(stats: dict, weights: dict) -> float:
    return sum(weights[s] * stats.get(s, 0.0) for s in STATS)


# ---------------------------------------------------------------------------
# Step 8 -- 1-D k-means for tiers
# ---------------------------------------------------------------------------

def kmeans_1d(values: list[float], k: int = 6, iters: int = 200) -> list[int]:
    """
    Deterministic 1-D k-means. Initialized at evenly spaced quantiles of the
    sorted data, which is stable and needs no RNG. Returns a cluster index per
    input value; clusters are renumbered 0..k-1 by descending centroid, so 0 is
    always the best group.
    """
    n = len(values)
    if n == 0:
        return []
    if n <= k:
        order = sorted(range(n), key=lambda i: -values[i])
        out = [0] * n
        for rank, i in enumerate(order):
            out[i] = rank
        return out

    srt = sorted(values)
    centroids = [srt[min(n - 1, int((j + 0.5) * n / k))] for j in range(k)]

    assign = [0] * n
    for _ in range(iters):
        changed = False
        for i, v in enumerate(values):
            best = min(range(k), key=lambda j: abs(v - centroids[j]))
            if best != assign[i]:
                assign[i] = best
                changed = True
        sums = [0.0] * k
        counts = [0] * k
        for i, v in enumerate(values):
            sums[assign[i]] += v
            counts[assign[i]] += 1
        for j in range(k):
            if counts[j]:
                centroids[j] = sums[j] / counts[j]
        if not changed:
            break

    order = sorted(range(k), key=lambda j: -centroids[j])
    remap = {old: new for new, old in enumerate(order)}
    return [remap[a] for a in assign]


# Tiers produced by the k-means clustering...
KMEANS_TIER_NAMES = ["S+", "S", "A", "B", "C", "D"]
# ...plus S++, which is not a cluster: it is the top `teams` players overall by
# projected season total -- i.e. the players who must go in round one. Sized to
# the league so it stays "the first round" whatever the league size.
TIER_NAMES = ["S++"] + KMEANS_TIER_NAMES


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_player(p: dict, lg_rates: dict, lg_shpct: dict) -> dict:
    pg = p["positionGroup"]
    seasons = p["seasons"]
    present = [s for s in SEASONS if s in seasons]

    out = {
        "playerId": p["playerId"],
        "playerName": p["playerName"],
        "position": p["position"],
        "positionGroup": pg,
        "team": p["currentTeam"],
        "age": p["age"] if p["age"] is not None else "",
        "seasonsPlayed": len(present),
    }

    # --- Step 1 + 2: weighted rate regressed toward the positional mean ------
    raw_rate = {}
    for stat in STATS:
        num, den = weighted_num_den(p, stat)
        lg = lg_rates[(pg, stat)]
        ph = PHANTOM_GP[stat]
        raw_rate[stat] = (num + lg * ph) / (den + ph)

    # --- Step 3: goals get a shooting-percentage-based second opinion --------
    career_goals = sum(seasons[s]["goals"] for s in present)
    career_shots = sum(seasons[s]["shots"] for s in present)
    lg_sh = lg_shpct[pg]
    shpct_proj = ((career_goals + lg_sh * SHPCT_PRIOR_SHOTS) /
                  (career_shots + SHPCT_PRIOR_SHOTS))
    goals_alt = raw_rate["shots"] * shpct_proj
    raw_rate["goals"] = GOAL_BLEND * raw_rate["goals"] + (1 - GOAL_BLEND) * goals_alt

    last = present[-1] if present else None
    shpct_last = ((seasons[last]["goals"] / seasons[last]["shots"])
                  if last and seasons[last]["shots"] > 0 else None)

    out["shPct_last"] = round(100 * shpct_last, 2) if shpct_last is not None else ""
    out["shPct_proj"] = round(100 * shpct_proj, 2)
    if shpct_last is None:
        out["shPct_flag"] = "neutral"
    else:
        diff = 100 * (shpct_last - shpct_proj)
        if diff < -2.0:
            out["shPct_flag"] = "positive regression"
        elif diff > 2.0:
            out["shPct_flag"] = "negative regression"
        else:
            out["shPct_flag"] = "neutral"

    # --- Ice time: weighted history level and trend --------------------------
    # Computed here because Step 6 needs them; Step 7 reports the same trend.
    toi_num = sum(SEASON_WEIGHT[s] * seasons[s]["gamesPlayed"] * seasons[s]["toiPerGame"]
                  for s in present)
    toi_den = sum(SEASON_WEIGHT[s] * seasons[s]["gamesPlayed"] for s in present)
    toi_hist = toi_num / toi_den if toi_den > 0 else 0.0
    played = [s for s in SEASONS if s in seasons and seasons[s]["gamesPlayed"] > 0]
    toi_trend = ols_slope([float(SEASONS.index(s)) for s in played],
                          [seasons[s]["toiPerGame"] for s in played])

    toi_proj = toi_hist + toi_trend * TOI_LAG * TOI_TREND_DAMPING
    if toi_hist > 0:
        toi_mult = max(TOI_MULT_CLAMP[0], min(TOI_MULT_CLAMP[1], toi_proj / toi_hist))
    else:
        toi_mult = 1.0
    out["toiHist"] = round(toi_hist, 2)
    out["toiProj"] = round(toi_hist * toi_mult, 2)
    out["toiMult"] = round(toi_mult, 4)

    # --- Step 4: age adjustment ---------------------------------------------
    out["ageMult_scoring"] = round(age_multiplier(p["age"], pg, "goals"), 4)
    out["ageMult_physical"] = round(age_multiplier(p["age"], pg, "hits"), 4)

    # --- Step 5: games played ------------------------------------------------
    gp_num = sum(SEASON_WEIGHT[s] * seasons[s]["gamesPlayed"] for s in present)
    gp_weighted = gp_num / sum(SEASON_WEIGHT.values())
    gp_proj = min(82.0, (1 - GP_PROJ_ANCHOR_WEIGHT) * gp_weighted +
                  GP_PROJ_ANCHOR_WEIGHT * GP_PROJ_ANCHOR)
    out["gp_proj"] = round(gp_proj, 1)
    out["injury_risk"] = any(seasons[s]["gamesPlayed"] < INJURY_GP_THRESHOLD
                             for s in present)

    # --- Step 6: assemble ----------------------------------------------------
    proj = {}
    for stat in STATS:
        combined = age_multiplier(p["age"], pg, stat) * toi_mult
        combined = max(COMBINED_CLAMP[0], min(COMBINED_CLAMP[1], combined))
        final_rate = raw_rate[stat] * combined
        proj[stat] = final_rate * gp_proj
        out[f"proj_{stat}"] = round(proj[stat], 2)
        out[f"rate_{stat}"] = round(final_rate, 5)

    fp = fantasy_points(proj, DEFAULT_WEIGHTS)
    out["fantasyPoints_default"] = round(fp, 2)
    out["fppg_default"] = round(fp / gp_proj, 3) if gp_proj > 0 else 0.0

    # --- Step 7: trend metrics ----------------------------------------------
    career_gp = sum(seasons[s]["gamesPlayed"] for s in present)
    out["careerGP"] = career_gp

    xs, per60_ys, hits60, blocks60 = [], [], [], []
    for idx, season in enumerate(SEASONS):
        s = seasons.get(season)
        label = f"y{idx + 1}"          # y1 = 2023-24, y3 = 2025-26
        if not s or s["gamesPlayed"] <= 0:
            out[f"fppg_{label}"] = ""
            out[f"gp_{label}"] = 0
            out[f"toi_{label}"] = ""
            continue
        s_fp = fantasy_points(s, DEFAULT_WEIGHTS)
        out[f"fppg_{label}"] = round(s_fp / s["gamesPlayed"], 3)
        out[f"gp_{label}"] = s["gamesPlayed"]
        out[f"toi_{label}"] = round(s["toiPerGame"], 2)
        xs.append(float(idx))
        minutes = s["toiPerGame"] * s["gamesPlayed"]
        per60_ys.append(s_fp / (minutes / 60.0) if minutes > 0 else 0.0)
        hits60.append(s["hits"] / (minutes / 60.0) if minutes > 0 else 0.0)
        blocks60.append(s["blocks"] / (minutes / 60.0) if minutes > 0 else 0.0)

    def _delta(a, b):
        return round(a - b, 3) if a != "" and b != "" else ""

    out["fppg_delta_y2_y1"] = _delta(out["fppg_y2"], out["fppg_y1"])
    out["fppg_delta_y3_y2"] = _delta(out["fppg_y3"], out["fppg_y2"])

    slope_per60 = ols_slope(xs, per60_ys)
    role_vol = (coeff_of_variation(hits60) + coeff_of_variation(blocks60)) / 2.0
    out["toi_trend"] = round(toi_trend, 3)
    out["slope_per60"] = round(slope_per60, 4)
    out["role_volatility"] = round(role_vol, 4)

    if slope_per60 > 0 and toi_trend > 0.5:
        out["trend_label"] = "Rising"
    elif slope_per60 < 0 and toi_trend < -0.5:
        out["trend_label"] = "Declining"
    elif role_vol > 0.25:
        out["trend_label"] = "Volatile"
    else:
        out["trend_label"] = "Stable"

    qualifying = sum(1 for s in present if seasons[s]["gamesPlayed"] >= CONFIDENCE_GP)
    if career_gp < 82:
        out["confidence"] = "Low"
    elif qualifying >= 3 and len(present) == 3:
        out["confidence"] = "High"
    elif qualifying == 2:
        out["confidence"] = "Medium"
    else:
        out["confidence"] = "Low"

    return out


# ---------------------------------------------------------------------------
# Step 8 -- tiers and replacement level, applied across the pool
# ---------------------------------------------------------------------------

def assign_tiers_and_vorp(rows: list[dict], teams: int,
                          f_slots: int, d_slots: int) -> dict[str, float]:
    baselines = {}
    for pg, slots in (("F", f_slots), ("D", d_slots)):
        pool = [r for r in rows if r["positionGroup"] == pg and r["gp_proj"] >= TIER_MIN_GP]

        # Tiers: 1-D k-means on projected FPPG so breaks land on real gaps.
        if pool:
            clusters = kmeans_1d([r["fppg_default"] for r in pool],
                                 k=len(KMEANS_TIER_NAMES))
            for r, c in zip(pool, clusters):
                r["tier_default"] = KMEANS_TIER_NAMES[c]

        # Replacement level: the (teams x slots)-th player by projected season total.
        n = teams * slots
        ranked = sorted(pool, key=lambda r: -r["fantasyPoints_default"])
        baseline = ranked[min(n, len(ranked)) - 1]["fantasyPoints_default"] if ranked else 0.0

        # The same idea in rate terms: the FPPG of the player who occupies the
        # last startable slot. VORP answers "how much does he add over a season",
        # VORP/G answers "how much better is he on any given night" -- which is
        # the one that matters if you stream around injuries or off-nights.
        ranked_rate = sorted(pool, key=lambda r: -r["fppg_default"])
        base_rate = ranked_rate[min(n, len(ranked_rate)) - 1]["fppg_default"] if ranked_rate else 0.0

        baselines[pg] = {"total": baseline, "fppg": base_rate}
        for r in rows:
            if r["positionGroup"] == pg:
                r["replacement_baseline"] = round(baseline, 2)
                r["vorp_default"] = round(r["fantasyPoints_default"] - baseline, 2)
                r["replacement_fppg"] = round(base_rate, 3)
                r["vorpPG_default"] = round(r["fppg_default"] - base_rate, 3)

    for r in rows:
        r.setdefault("tier_default", "NR")

    # S++ overlay: one draft board across F and D, ranked by projected season
    # total, cut at the size of the first round.
    eligible = sorted((r for r in rows if r["gp_proj"] >= TIER_MIN_GP),
                      key=lambda r: -r["fantasyPoints_default"])
    for r in eligible[:teams]:
        r["tier_default"] = "S++"

    return baselines


# ---------------------------------------------------------------------------

CSV_HEADER = [
    "playerId", "playerName", "position", "positionGroup", "team", "age",
    "gp_proj", "injury_risk",
    "proj_goals", "proj_assists", "proj_shots", "proj_hits", "proj_blocks",
    "rate_goals", "rate_assists", "rate_shots", "rate_hits", "rate_blocks",
    "ageMult_scoring", "ageMult_physical",
    "toiHist", "toiProj", "toiMult",
    "shPct_last", "shPct_proj", "shPct_flag",
    "fantasyPoints_default", "fppg_default", "vorp_default", "vorpPG_default",
    "replacement_baseline", "replacement_fppg", "tier_default",
    "fppg_y1", "fppg_y2", "fppg_y3", "fppg_delta_y2_y1", "fppg_delta_y3_y2",
    "gp_y1", "gp_y2", "gp_y3", "toi_y1", "toi_y2", "toi_y3",
    "toi_trend", "slope_per60", "role_volatility",
    "trend_label", "confidence", "seasonsPlayed", "careerGP",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teams", type=int, default=12, help="league size (default 12)")
    ap.add_argument("--f-slots", type=int, default=9, help="forward slots per team")
    ap.add_argument("--d-slots", type=int, default=4, help="defense slots per team")
    ap.add_argument("--in", dest="in_csv", default=IN_CSV)
    ap.add_argument("--out", dest="out_csv", default=OUT_CSV)
    args = ap.parse_args()

    try:
        players = load_players(args.in_csv)
        lg_rates, lg_shpct = league_rates(players)

        print(f"\nLeague weighted rates per game ({PROJECTION_SEASON} projection basis):")
        for pg in sorted({p["positionGroup"] for p in players.values()}):
            parts = "  ".join(f"{s}={lg_rates[(pg, s)]:.3f}" for s in STATS)
            print(f"  {pg}: {parts}   sh%={100 * lg_shpct[pg]:.2f}")

        no_age = sum(1 for p in players.values() if p["age"] is None)
        if no_age:
            print(f"\n{no_age} players have no age -> age multiplier forced to 1.00")

        rows = [project_player(p, lg_rates, lg_shpct) for p in players.values()]
        baselines = assign_tiers_and_vorp(rows, args.teams, args.f_slots, args.d_slots)
        rows.sort(key=lambda r: -r["fantasyPoints_default"])

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_HEADER, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} projections -> {args.out_csv}")

        print(f"\nReplacement level ({args.teams} teams x {args.f_slots}F / "
              f"{args.d_slots}D), default scoring:")
        for pg in ("F", "D"):
            b = baselines.get(pg, {"total": 0.0, "fppg": 0.0})
            print(f"  {pg}: {b['total']:.1f} fantasy points over the season, "
                  f"{b['fppg']:.3f} per game")

        counts = defaultdict(int)
        for r in rows:
            counts[(r["positionGroup"], r["tier_default"])] += 1
        print("\nTier counts (k-means on projected FPPG, per position group):")
        for pg in ("F", "D"):
            line = "  ".join(f"{t}={counts[(pg, t)]}" for t in TIER_NAMES + ["NR"])
            print(f"  {pg}: {line}")

        labels = defaultdict(int)
        conf = defaultdict(int)
        for r in rows:
            labels[r["trend_label"]] += 1
            conf[r["confidence"]] += 1
        print(f"\nTrend labels: {dict(labels)}")
        print(f"Confidence:   {dict(conf)}")

        print("\nTop 20 by projected fantasy points (default scoring):")
        hdr = (f"{'#':>3} {'Player':<24}{'Pos':<4}{'Tm':<5}{'Tier':<5}{'GP':>5}"
               f"{'FPPG':>7}{'Total':>8}{'VORP':>8}  {'G':>5}{'A':>5}{'S':>6}"
               f"{'H':>6}{'B':>6}  {'Trend':<10}{'Conf':<7}")
        print(hdr)
        for i, r in enumerate(rows[:20], 1):
            print(f"{i:>3} {r['playerName'][:23]:<24}{r['position']:<4}"
                  f"{(r['team'] or '-')[:4]:<5}{r['tier_default']:<5}"
                  f"{r['gp_proj']:>5.1f}{r['fppg_default']:>7.2f}"
                  f"{r['fantasyPoints_default']:>8.1f}{r['vorp_default']:>8.1f}  "
                  f"{r['proj_goals']:>5.1f}{r['proj_assists']:>5.1f}"
                  f"{r['proj_shots']:>6.1f}{r['proj_hits']:>6.1f}"
                  f"{r['proj_blocks']:>6.1f}  {r['trend_label']:<10}{r['confidence']:<7}")
        return 0
    except DataError as e:
        print(f"\nDATA ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
