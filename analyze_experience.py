#!/usr/bin/env python3
"""
analyze_experience.py -- does production really jump most in a player's first
3-5 NHL seasons?

Uses the career season-by-season history already cached under cache/landing_*.json
(no network calls). Career history carries goals/assists/points/shots/TOI but NOT
hits or blocked shots, so this tests *scoring*, which is the age-sensitive part of
fantasy value.

    python3 analyze_experience.py
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
from collections import defaultdict

import pandas as pd

DEBUT_MIN_GP = 10     # a 1-9 game cup of coffee does not start the experience clock
SEASON_MIN_GP = 20    # a season must be this long to count as an observation
RECENT = ["20232024", "20242025", "20252026"]


def load_careers() -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join("cache", "landing_*.json")))
    if not files:
        raise SystemExit("no cached landing files -- run fetch_nhl.py first")

    for path in files:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        pid = d.get("playerId")
        pos = d.get("position")
        if not pid or pos == "G":
            continue
        birth = d.get("birthDate")
        draft = (d.get("draftDetails") or {}).get("overallPick")

        # Aggregate NHL regular season by season (traded players get two rows).
        per_season: dict[str, dict] = defaultdict(
            lambda: {"gp": 0, "g": 0, "a": 0, "p": 0, "shots": 0, "toi_sec": 0.0})
        for r in d.get("seasonTotals", []):
            if r.get("leagueAbbrev") != "NHL" or r.get("gameTypeId") != 2:
                continue
            s = str(r.get("season"))
            gp = int(r.get("gamesPlayed") or 0)
            acc = per_season[s]
            acc["gp"] += gp
            acc["g"] += int(r.get("goals") or 0)
            acc["a"] += int(r.get("assists") or 0)
            acc["p"] += int(r.get("points") or 0)
            acc["shots"] += int(r.get("shots") or 0)
            toi = r.get("avgToi")           # "mm:ss"
            if toi and ":" in str(toi):
                m, sec = str(toi).split(":")[:2]
                acc["toi_sec"] += (int(m) * 60 + int(sec)) * gp

        if not per_season:
            continue

        seasons = sorted(per_season)
        debut = next((s for s in seasons if per_season[s]["gp"] >= DEBUT_MIN_GP), None)
        if debut is None:
            continue
        debut_start = int(debut[:4])

        for s in seasons:
            acc = per_season[s]
            if acc["gp"] <= 0:
                continue
            start_year = int(s[:4])
            if start_year < debut_start:
                continue                      # pre-debut cup of coffee
            age = None
            if birth:
                try:
                    bd = dt.date.fromisoformat(birth[:10])
                    age = round((dt.date(start_year + 1, 2, 1) - bd).days / 365.2425, 2)
                except ValueError:
                    pass
            rows.append({
                "playerId": pid, "position": pos,
                "positionGroup": "D" if pos == "D" else "F",
                "season": s, "startYear": start_year,
                "expYear": start_year - debut_start + 1,
                "age": age, "draftPick": draft,
                "gp": acc["gp"], "goals": acc["g"], "assists": acc["a"],
                "points": acc["p"], "shots": acc["shots"],
                "ppg": acc["p"] / acc["gp"],
                "toiPerGame": (acc["toi_sec"] / acc["gp"] / 60.0) if acc["gp"] else 0.0,
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} NHL player-seasons for {df.playerId.nunique()} skaters "
          f"({df.startYear.min()}-{df.startYear.max()})")
    return df


def era_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Index each season's P/GP to that season's mean, separately for F and D."""
    df = df[df.gp >= SEASON_MIN_GP].copy()
    league = (df.groupby(["season", "positionGroup"])["ppg"].mean()
                .rename("leaguePPG").reset_index())
    df = df.merge(league, on=["season", "positionGroup"])
    df["relPPG"] = df["ppg"] / df["leaguePPG"]
    return df


def transitions(df: pd.DataFrame, seasons_filter=None) -> pd.DataFrame:
    """Year-over-year changes for consecutive seasons played by the same player."""
    df = df.sort_values(["playerId", "startYear"])
    out = []
    for pid, g in df.groupby("playerId", sort=False):
        recs = g.to_dict("records")
        for a, b in zip(recs, recs[1:]):
            if b["startYear"] != a["startYear"] + 1:
                continue                      # skip gap years (injury, KHL, AHL)
            if seasons_filter and b["season"] not in seasons_filter:
                continue
            out.append({
                "playerId": pid, "positionGroup": a["positionGroup"],
                "fromExp": a["expYear"], "toSeason": b["season"],
                "age": a["age"], "draftPick": a["draftPick"],
                "relFrom": a["relPPG"], "relTo": b["relPPG"],
                "dRel": b["relPPG"] - a["relPPG"],
                "pctChange": (b["relPPG"] / a["relPPG"] - 1) * 100 if a["relPPG"] > 0 else None,
                "dTOI": b["toiPerGame"] - a["toiPerGame"],
            })
    return pd.DataFrame(out)


def summarize(t: pd.DataFrame, label: str, by="fromExp", max_bucket=14) -> pd.DataFrame:
    t = t.copy()
    t[by] = t[by].clip(upper=max_bucket)
    g = t.groupby(by).agg(
        n=("dRel", "size"),
        mean_change=("dRel", "mean"),
        median_change=("dRel", "median"),
        pct_improving=("dRel", lambda s: 100 * (s > 0).mean()),
        mean_dTOI=("dTOI", "mean"),
    ).reset_index()
    g = g[g.n >= 25]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    hdr = f"{'exp yr':>7} {'->':^4} {'n':>5} {'mean Δ':>9} {'median Δ':>10} {'% up':>7} {'mean ΔTOI':>11}"
    print(hdr)
    for _, r in g.iterrows():
        arrow = f"{int(r[by])}→{int(r[by]) + 1}"
        star = "  *" if r.mean_change > 0.03 else ""
        print(f"{arrow:>7} {'':^4} {int(r.n):>5} {r.mean_change:>+9.3f} "
              f"{r.median_change:>+10.3f} {r.pct_improving:>6.0f}% {r.mean_dTOI:>+10.2f}{star}")
    return g


if __name__ == "__main__":
    careers = load_careers()
    adj = era_adjust(careers)

    print(f"\nAfter the GP>={SEASON_MIN_GP} filter: {len(adj)} player-seasons")
    print("Relative P/GP is each player's points-per-game divided by the mean for "
          "their position group that season, so league scoring changes cancel out.\n"
          "Δ is the change in that relative figure from one season to the next.")

    # A. Recent, era-clean, pool-complete: the two most recent transitions.
    recent = transitions(adj, seasons_filter={"20242025", "20252026"})
    summarize(recent, "A. RECENT TRANSITIONS ONLY (2023-24→24-25, 24-25→25-26)\n"
                      "   Every active skater is in the pool for these seasons.")

    # B. Full career history: many more observations, more survivorship bias.
    full = transitions(adj)
    summarize(full, "B. ALL CAREER TRANSITIONS (more data, more survivorship bias)")

    # C. Same data grouped by age instead of experience -- does experience add
    #    anything the existing age curve does not already capture?
    full_age = full.dropna(subset=["age"]).copy()
    full_age["ageBucket"] = full_age["age"].round().astype(int).clip(19, 36)
    g = full_age.groupby("ageBucket").agg(
        n=("dRel", "size"), mean_change=("dRel", "mean"),
        pct_improving=("dRel", lambda s: 100 * (s > 0).mean())).reset_index()
    g = g[g.n >= 25]
    print(f"\n{'=' * 78}\nC. THE SAME TRANSITIONS GROUPED BY AGE\n{'=' * 78}")
    print(f"{'age':>5} {'n':>6} {'mean Δ':>9} {'% up':>7}")
    for _, r in g.iterrows():
        print(f"{int(r.ageBucket):>5} {int(r.n):>6} {r.mean_change:>+9.3f} {r.pct_improving:>6.0f}%")

    # D. Forwards vs defensemen.
    for pg in ("F", "D"):
        summarize(full[full.positionGroup == pg], f"D. ALL TRANSITIONS -- {pg} only")

    # E. Survivorship check: restrict to players who went on to play >=3 more
    #    seasons, so early-career busts cannot inflate the early-year jumps.
    counts = adj.groupby("playerId").size().rename("careerSeasons")
    full2 = full.merge(counts, on="playerId")
    summarize(full2[full2.careerSeasons >= 8],
              "E. SURVIVORSHIP CHECK -- only players with 8+ NHL seasons\n"
              "   (removes the 'improved or got cut' selection effect)")

    # F. Cumulative: how much of a career's total gain happens by year 5?
    print(f"\n{'=' * 78}\nF. CUMULATIVE GAIN BY EXPERIENCE YEAR (all transitions)\n{'=' * 78}")
    cum = full.groupby(full.fromExp.clip(upper=14))["dRel"].mean().cumsum()
    base = cum.loc[cum.index <= 14]
    peak = base.max()
    for yr, v in base.items():
        share = 100 * v / peak if peak else 0
        bar = "#" * max(0, int(round(share / 4)))
        print(f"  through yr {int(yr) + 1:>2}: {v:>+7.3f}  ({share:>5.0f}% of peak) {bar}")

    # G. The decisive test for the projection: hold age constant, vary experience.
    #    If experience only proxies for youth, these rows will be flat.
    print(f"\n{'=' * 78}\nG. AGE HELD CONSTANT, EXPERIENCE VARIED (mean Δ relative P/GP)\n{'=' * 78}")
    fa = full.dropna(subset=["age"]).copy()
    fa["ageB"] = pd.cut(fa.age, [20, 22, 24, 26, 28, 31, 40],
                        labels=["21-22", "23-24", "25-26", "27-28", "29-31", "32+"])
    fa["expB"] = pd.cut(fa.fromExp, [0, 2, 5, 9, 40],
                        labels=["exp 1-2", "exp 3-5", "exp 6-9", "exp 10+"])
    piv = fa.pivot_table(index="ageB", columns="expB", values="dRel",
                         aggfunc="mean", observed=True)
    cnt = fa.pivot_table(index="ageB", columns="expB", values="dRel",
                         aggfunc="size", observed=True)
    print(f"{'age':>7}  " + "".join(f"{c:>16}" for c in piv.columns))
    for idx in piv.index:
        cells = []
        for c in piv.columns:
            v, n = piv.loc[idx, c], cnt.loc[idx, c]
            cells.append(f"{v:>+9.3f} (n={int(n):<3d})" if pd.notna(v) and n >= 25
                         else f"{'--':>16}")
        print(f"{str(idx):>7}  " + "".join(cells))
    print("\nRead across a row: same age, more NHL experience. If the numbers fall")
    print("from left to right, experience carries information age does not.")

    # H. How much of the 2026-27 pool would an experience term actually touch?
    proj_path = os.path.join("data", "projections_2026_27.csv")
    if os.path.exists(proj_path):
        proj = pd.read_csv(proj_path)
        debut = careers.groupby("playerId").startYear.min().rename("debutYear")
        proj = proj.merge(debut, left_on="playerId", right_index=True, how="left")
        proj["expIn2026"] = 2026 - proj.debutYear + 1
        pool = proj[proj.gp_proj >= 40]
        print(f"\n{'=' * 78}\nH. THE 2026-27 POOL BY EXPERIENCE\n{'=' * 78}")
        for lo, hi, lbl in [(1, 1, "rookie (yr 1)"), (2, 3, "yr 2-3"),
                            (4, 5, "yr 4-5"), (6, 9, "yr 6-9"), (10, 99, "yr 10+")]:
            sub = pool[(pool.expIn2026 >= lo) & (pool.expIn2026 <= hi)]
            print(f"  {lbl:<15} {len(sub):>4} players  "
                  f"mean projected FPPG {sub.fppg_default.mean():.2f}")
        rising = pool[(pool.expIn2026 <= 5) & (pool.expIn2026 >= 1)]
        print(f"\n  {len(rising)} of {len(pool)} projectable players "
              f"({100 * len(rising) / len(pool):.0f}%) are in the yr 1-5 window "
              f"where production is still climbing.")
