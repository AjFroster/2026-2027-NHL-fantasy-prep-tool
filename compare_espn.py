#!/usr/bin/env python3
"""
compare_espn.py -- compare these projections against ESPN's for 2026-27.

ESPN's per-league endpoints need authentication, but the league-independent
players endpoint is public and carries their own projections:

  lm-api-reads.fantasy.espn.com/apis/v3/games/fhl/seasons/2027/players
      ?scoringPeriodId=0&view=kona_player_info

Stat ids were decoded by matching a known 2025-26 season against data/:
13 goals, 14 assists, 16 points, 29 shots, 30 games, 31 hits, 32 blocks.

ESPN's own "applied total" uses ESPN default scoring, so it is ignored: both
sides are re-scored with the weights below to make the comparison fair.

    python3 compare_espn.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import urllib.request

import pandas as pd

CACHE = os.path.join("cache", "espn_projections_2027.json")
URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/fhl/seasons/2027/"
       "players?scoringPeriodId=0&view=kona_player_info")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

STAT = {"goals": "13", "assists": "14", "shots": "29",
        "gp": "30", "hits": "31", "blocks": "32"}
POS = {1: "C", 2: "L", 3: "R", 4: "D", 5: "G"}
W = {"goals": 2.0, "assists": 1.5, "shots": 0.15, "hits": 0.2, "blocks": 0.35}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace(".", "").replace("-", " ").replace("'", "").strip()


def fetch(refresh: bool) -> list:
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def espn_frame(raw: list) -> pd.DataFrame:
    rows = []
    for p in raw:
        pos = POS.get(p.get("defaultPositionId"))
        if pos in (None, "G"):
            continue
        proj = next((s for s in p.get("stats", [])
                     if s.get("seasonId") == 2027 and s.get("statSourceId") == 1
                     and s.get("statSplitTypeId") == 0), None)
        if not proj or not proj.get("stats"):
            continue
        st = proj["stats"]
        gp = st.get(STAT["gp"], 0)
        if not gp:
            continue
        rows.append({
            "name": p["fullName"], "key": norm(p["fullName"]), "pos": pos,
            "pg": "D" if pos == "D" else "F",
            "gp": gp,
            **{k: st.get(v, 0.0) for k, v in STAT.items() if k != "gp"},
        })
    df = pd.DataFrame(rows)
    df["fp"] = sum(W[s] * df[s] for s in W)
    df["fppg"] = df.fp / df.gp
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    espn = espn_frame(fetch(a.refresh))
    mine = pd.read_csv(os.path.join("data", "projections_2026_27.csv"))
    mine["key"] = mine.playerName.map(norm)
    mine = mine.rename(columns={"gp_proj": "gp", "fantasyPoints_default": "fp",
                                "fppg_default": "fppg"})

    # Two active players share a name, so join on name + position group.
    m = mine.merge(espn, on=["key", "positionGroup"] if False else ["key"],
                   suffixes=("_me", "_espn"), how="inner")
    m = m[m.positionGroup == m.pg]
    m = m.drop_duplicates(subset=["playerId"])

    print(f"ESPN skaters with a 2026-27 projection : {len(espn)}")
    print(f"Mine                                   : {len(mine)}")
    print(f"Matched on name + position group       : {len(m)}\n")

    print("=" * 74)
    print("GAMES PLAYED: the biggest philosophical difference")
    print("=" * 74)
    full = (espn.gp >= 82).sum()
    print(f"  ESPN projects exactly 82 games for {full} of {len(espn)} skaters "
          f"({100*full/len(espn):.0f}%); mean {espn.gp.mean():.1f}")
    print(f"  Mine, same {len(m)} players: mean {m.gp_me.mean():.1f}, "
          f"max {m.gp_me.max():.1f} (82-game seasons: {(m.gp_me >= 82).sum()})")
    print(f"  ESPN, same {len(m)} players: mean {m.gp_espn.mean():.1f}")
    print("  ESPN projects health; this tool projects availability, so ESPN's")
    print("  season totals run higher almost everywhere. Per-game rates are the")
    print("  fair comparison.\n")

    for label, col in (("FPPG (rate)", "fppg"), ("Season total", "fp")):
        me, es = m[f"{col}_me"], m[f"{col}_espn"]
        # rank-then-pearson == spearman, without needing scipy
        spearman = me.rank().corr(es.rank())
        print(f"{label:<14} pearson r = {me.corr(es):.3f}   "
              f"spearman = {spearman:.3f}   "
              f"mine/ESPN mean ratio = {me.mean()/es.mean():.3f}")

    print("\n" + "=" * 74)
    print("TOP 20 BY MY FPPG, SIDE BY SIDE (both scored with my weights)")
    print("=" * 74)
    print(f"{'player':<22}{'pos':<4}{'my GP':>6}{'ESPN GP':>8}{'my FPPG':>9}"
          f"{'ESPN':>7}{'diff':>7}{'my rank':>9}{'ESPN rk':>8}")
    m["rank_me"] = m.fppg_me.rank(ascending=False)
    m["rank_espn"] = m.fppg_espn.rank(ascending=False)
    for _, r in m.nlargest(20, "fppg_me").iterrows():
        print(f"{r.playerName[:21]:<22}{r.position:<4}{r.gp_me:>6.1f}{r.gp_espn:>8.0f}"
              f"{r.fppg_me:>9.2f}{r.fppg_espn:>7.2f}{r.fppg_me-r.fppg_espn:>+7.2f}"
              f"{r.rank_me:>9.0f}{r.rank_espn:>8.0f}")

    m["gap"] = m.rank_espn - m.rank_me
    pool = m[(m.gp_me >= 40) & (m.gp_espn >= 40)]
    print("\n" + "=" * 74)
    print("WHERE WE DISAGREE MOST (rank by FPPG, both on my scoring)")
    print("=" * 74)
    print("\n  I am much higher on:")
    for _, r in pool.nlargest(10, "gap").iterrows():
        print(f"    {r.playerName[:23]:<24}{r.position:<3} mine #{r.rank_me:>4.0f}  "
              f"ESPN #{r.rank_espn:>4.0f}   {r.fppg_me:.2f} vs {r.fppg_espn:.2f}")
    print("\n  ESPN is much higher on:")
    for _, r in pool.nsmallest(10, "gap").iterrows():
        print(f"    {r.playerName[:23]:<24}{r.position:<3} mine #{r.rank_me:>4.0f}  "
              f"ESPN #{r.rank_espn:>4.0f}   {r.fppg_me:.2f} vs {r.fppg_espn:.2f}")

    print("\n" + "=" * 74)
    print("STAT BY STAT, PER GAME (matched players, mine / ESPN)")
    print("=" * 74)
    for s in W:
        a_ = (m[f"proj_{s}"] / m.gp_me).mean()
        b_ = (m[s] / m.gp_espn).mean()
        flag = "  <-- ESPN much higher" if b_ / a_ > 1.15 else (
               "  <-- I am much higher" if a_ / b_ > 1.15 else "")
        print(f"  {s:<8} mine {a_:.3f}   ESPN {b_:.3f}   "
              f"mine/ESPN {a_/b_:.3f}{flag}")


if __name__ == "__main__":
    main()
