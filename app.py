#!/usr/bin/env python3
"""
app.py -- Streamlit front end for the NHL fantasy projections.

Two tabs: the 2026-27 projection and the three-year history behind it.
Scoring weights live in the sidebar and every number on both tabs recomputes
when you change them -- nothing is baked in.

    streamlit run app.py

Reads data/projections_2026_27.csv and data/skaters_3yr.csv, and reuses the
projection module's k-means and constants so the tiers match project.py
exactly.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from project import (
    DEFAULT_WEIGHTS,
    KMEANS_TIER_NAMES,
    STATS,
    TIER_MIN_GP,
    TIER_NAMES,
    kmeans_1d,
)

PROJ_CSV = os.path.join("data", "projections_2026_27.csv")
HIST_CSV = os.path.join("data", "skaters_3yr.csv")

SEASON_LABEL = {20232024: "2023-24", 20242025: "2024-25", 20252026: "2025-26"}
POS_CHOICES = ["All", "Forwards", "Defense"]
POS_CODE = {"All": None, "Forwards": "F", "Defense": "D"}

STAT_LABEL = {"goals": "Goal", "assists": "Assist", "shots": "Shot on goal",
              "hits": "Hit", "blocks": "Blocked shot"}
STAT_STEP = {"goals": 0.05, "assists": 0.05, "shots": 0.01, "hits": 0.05, "blocks": 0.05}
HIST_STAT_COL = {"goals": "goals", "assists": "assists", "shots": "shots",
                 "hits": "hits", "blocks": "blockedShots"}

TIER_COLORS = {"S++": "#e6e9f0", "S+": "#f2d675", "S": "#c9a227", "A": "#57a773", "B": "#4a86c5",
               "C": "#7d7f94", "D": "#6a5060", "NR": "#2a2f3a"}

st.set_page_config(page_title="NHL Fantasy Projections 2026-27",
                   page_icon="🏒", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (PROJ_CSV, HIST_CSV):
        if not os.path.exists(path):
            st.error(f"Missing `{path}`. Run `python3 fetch_nhl.py` then "
                     f"`python3 project.py` first.")
            st.stop()
    proj = pd.read_csv(PROJ_CSV)
    hist = pd.read_csv(HIST_CSV)
    hist["season"] = hist["seasonId"].map(SEASON_LABEL)
    return proj, hist


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def weight_inputs() -> dict[str, float]:
    for s in STATS:
        st.session_state.setdefault(f"w_{s}", float(DEFAULT_WEIGHTS[s]))

    st.sidebar.subheader("Scoring")
    for s in STATS:
        st.sidebar.number_input(STAT_LABEL[s], step=STAT_STEP[s],
                                format="%.2f", key=f"w_{s}")

    def reset():
        for s in STATS:
            st.session_state[f"w_{s}"] = float(DEFAULT_WEIGHTS[s])

    st.sidebar.button("Reset to defaults", on_click=reset, width="stretch")
    return {s: float(st.session_state[f"w_{s}"]) for s in STATS}


def weight_slug(w: dict[str, float]) -> str:
    return (f"g{w['goals']:g}_a{w['assists']:g}_s{w['shots']:g}"
            f"_h{w['hits']:g}_b{w['blocks']:g}")


def score(df: pd.DataFrame, w: dict[str, float], cols: dict[str, str]) -> pd.Series:
    total = 0.0
    for s in STATS:
        total = total + w[s] * df[cols[s]]
    return total


# ---------------------------------------------------------------------------
# Tiers and replacement level, recomputed for the current weights
# ---------------------------------------------------------------------------

def add_tiers_and_vorp(df: pd.DataFrame, teams: int, f_slots: int,
                       d_slots: int) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.copy()
    df["tier"] = "NR"
    df["vorp"] = 0.0
    df["vorpPG"] = 0.0
    baselines: dict[str, dict[str, float]] = {}

    for pg, slots in (("F", f_slots), ("D", d_slots)):
        mask = (df["positionGroup"] == pg) & (df["gp"] >= TIER_MIN_GP)
        pool = df.loc[mask]
        if pool.empty:
            baselines[pg] = {"total": 0.0, "fppg": 0.0}
            continue

        clusters = kmeans_1d(pool["fppg"].tolist(), k=len(KMEANS_TIER_NAMES))
        df.loc[mask, "tier"] = [KMEANS_TIER_NAMES[c] for c in clusters]

        n = teams * slots
        ranked = pool["fp"].sort_values(ascending=False).tolist()
        baseline = ranked[min(n, len(ranked)) - 1]
        # The same idea in rate terms: the FPPG of the player holding the last
        # startable slot. VORP is season-long value; VORP/G is the per-night edge.
        ranked_rate = pool["fppg"].sort_values(ascending=False).tolist()
        base_rate = ranked_rate[min(n, len(ranked_rate)) - 1]

        baselines[pg] = {"total": baseline, "fppg": base_rate}
        sel = df["positionGroup"] == pg
        df.loc[sel, "vorp"] = df.loc[sel, "fp"] - baseline
        df.loc[sel, "vorpPG"] = df.loc[sel, "fppg"] - base_rate

    # S++ overlay: top `teams` players overall by projected season total.
    eligible = df.index[df["gp"] >= TIER_MIN_GP]
    first_round = df.loc[eligible, "fp"].nlargest(teams).index
    df.loc[first_round, "tier"] = "S++"

    return df, baselines


def build_projection_view(proj: pd.DataFrame, w: dict[str, float], teams: int,
                          f_slots: int, d_slots: int) -> tuple[pd.DataFrame, dict]:
    df = proj.copy()
    df["fp"] = score(df, w, {s: f"proj_{s}" for s in STATS})
    df["gp"] = df["gp_proj"]
    df["fppg"] = (df["fp"] / df["gp"]).where(df["gp"] > 0, 0.0)
    return add_tiers_and_vorp(df, teams, f_slots, d_slots)


def build_history_view(hist: pd.DataFrame, w: dict[str, float]) -> pd.DataFrame:
    df = hist.copy()
    df["fp"] = score(df, w, HIST_STAT_COL)
    df["fppg"] = (df["fp"] / df["gamesPlayed"]).where(df["gamesPlayed"] > 0, 0.0)
    minutes = df["toiPerGame"] * df["gamesPlayed"]
    df["fpPer60"] = (df["fp"] / (minutes / 60.0)).where(minutes > 0, 0.0)
    return df


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def tier_style(col: pd.Series):
    return [f"background-color: {TIER_COLORS.get(v, '#2a2f3a')}; color: #10131a; "
            f"font-weight: 700" for v in col]


def flag_style(col: pd.Series):
    out = []
    for v in col:
        if v == "positive regression":
            out.append("color: #66c98a")
        elif v == "negative regression":
            out.append("color: #e2756c")
        else:
            out.append("color: #8b95a7")
    return out


def download(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


def sparkline_frame(seasons: pd.DataFrame, proj_fppg: float | None) -> pd.DataFrame:
    rows = [{"Season": r["season"], "FPPG": r["fppg"], "Series": "actual"}
            for _, r in seasons.sort_values("seasonId").iterrows()]
    if proj_fppg is not None:
        rows.append({"Season": "2026-27 proj", "FPPG": proj_fppg, "Series": "projected"})
    return pd.DataFrame(rows)


def trend_chart(df: pd.DataFrame, y: str, y_title: str, series_col: str | None = None):
    """Line + points, y auto-scaled -- a zero baseline flattens three seasons."""
    base = alt.Chart(df).encode(
        x=alt.X("Season:N", sort=None, title=None),
        y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False), title=y_title),
        tooltip=list(df.columns))
    line = base.mark_line(color="#6b7688", strokeWidth=2)
    if series_col:
        points = base.mark_point(size=90, filled=True).encode(
            color=alt.Color(f"{series_col}:N",
                            scale=alt.Scale(domain=["actual", "projected"],
                                            range=["#d8dee9", "#61a0ff"]),
                            legend=None))
    else:
        points = base.mark_point(size=80, filled=True, color="#d8dee9")
    return (line + points).properties(height=240)

DETAIL_FMT = {
    "GP": "{:.1f}", "TOI/GP": "{:.2f}", "G": "{:.1f}", "A": "{:.1f}", "S": "{:.1f}",
    "H": "{:.1f}", "B": "{:.1f}", "Sh%": "{:.1f}", "FP": "{:.1f}", "FPPG": "{:.2f}",
    "FP/60": "{:.2f}", "G/60": "{:.2f}", "A/60": "{:.2f}", "S/60": "{:.2f}",
    "H/60": "{:.2f}", "B/60": "{:.2f}",
}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

proj_raw, hist_raw = load_data()

st.sidebar.title("🏒 Controls")
weights = weight_inputs()

st.sidebar.subheader("League")
teams = st.sidebar.number_input("Teams", min_value=1, max_value=32, value=12, step=1)
f_slots = st.sidebar.number_input("Forward slots", min_value=1, max_value=20, value=9, step=1)
d_slots = st.sidebar.number_input("Defense slots", min_value=1, max_value=20, value=4, step=1)
st.sidebar.caption(
    f"S++ is the top {teams} players overall by projected season total — the "
    "first round. S+ through D come from 1-D k-means on projected FPPG, "
    "computed separately for F and D. VORP is measured against the player "
    f"ranked {teams * f_slots}th (F) / {teams * d_slots}th (D) by season total; "
    "VORP/G is the same against that slot's FPPG. All of it "
    "recomputes whenever you change the scoring."
)

view, baselines = build_projection_view(proj_raw, weights, teams, f_slots, d_slots)
hist = build_history_view(hist_raw, weights)

tab_proj, tab_hist = st.tabs(["2026-27 Projection", "Three-Year History"])

# ---------------------------------------------------------------------------
# Tab 1 -- projection
# ---------------------------------------------------------------------------
with tab_proj:
    st.subheader("2026-27 projection")

    pos_pick = POS_CODE.get(st.session_state.get("p_pos") or "All")
    top_pool = view[view["gp"] >= TIER_MIN_GP]
    if pos_pick:
        top_pool = top_pool[top_pool["positionGroup"] == pos_pick]
    top10 = top_pool.nlargest(10, "fppg")[["playerName", "position", "fppg"]]
    cols = st.columns(5)
    for i, (_, r) in enumerate(top10.iterrows()):
        cols[i % 5].metric(f"{i + 1}. {r['playerName']} ({r['position']})",
                           f"{r['fppg']:.2f}")
    scope = {"F": "forwards", "D": "defensemen"}.get(pos_pick, "all skaters")
    st.caption(f"Top 10 {scope} by projected FPPG at the current weights "
               f"(min {TIER_MIN_GP} projected GP). Replacement level: "
               f"{baselines['F']['total']:.1f} F / {baselines['D']['total']:.1f} D "
               f"over the season ({baselines['F']['fppg']:.2f} / "
               f"{baselines['D']['fppg']:.2f} per game).")

    f1, f2, f3, f4, f5, f6 = st.columns([1.9, 1, 1, 1, 1, 2])
    pos = POS_CODE.get(f1.segmented_control(
        "Position", POS_CHOICES, default="All", key="p_pos") or "All")
    all_teams = sorted({t for s in view["team"].fillna("") for t in str(s).split("/") if t})
    team = f2.selectbox("Team", ["all"] + all_teams, key="p_team")
    min_gp = f3.number_input("Min GP", 0, 82, 0, step=1, key="p_gp")
    tier = f4.selectbox("Tier", ["all"] + TIER_NAMES + ["NR"], key="p_tier")
    trend = f5.selectbox("Trend", ["all", "Rising", "Declining", "Volatile", "Stable"],
                         key="p_trend")
    name_q = f6.text_input("Name contains", key="p_name")

    m = pd.Series(True, index=view.index)
    if pos:
        m &= view["positionGroup"] == pos
    if team != "all":
        m &= view["team"].fillna("").str.split("/").apply(lambda ts: team in ts)
    m &= view["gp"] >= min_gp
    if tier != "all":
        m &= view["tier"] == tier
    if trend != "all":
        m &= view["trend_label"] == trend
    if name_q:
        m &= view["playerName"].str.contains(name_q, case=False, na=False)

    shown = view[m].sort_values("fppg", ascending=False).copy()
    # Board rank: position by projected FPPG at the current weights, within the
    # current filter. Tied to the player, so it does not renumber when you sort
    # by a different column -- a player's rank does not change because you
    # looked at his hits.
    shown.insert(0, "#", range(1, len(shown) + 1))

    table = shown[[
        "#",
        "playerName", "position", "team", "tier", "gp", "fppg", "fp", "vorp",
        "vorpPG", "proj_goals", "proj_assists", "proj_shots", "proj_hits", "proj_blocks",
        "trend_label", "confidence", "shPct_flag", "injury_risk",
    ]].rename(columns={
        "playerName": "Player", "position": "Pos", "team": "Team", "tier": "Tier",
        "gp": "GP", "fppg": "FPPG", "fp": "Total", "vorp": "VORP",
        "vorpPG": "VORP/G",
        "proj_goals": "G", "proj_assists": "A", "proj_shots": "S",
        "proj_hits": "H", "proj_blocks": "B", "trend_label": "Trend",
        "confidence": "Conf", "shPct_flag": "Sh% flag", "injury_risk": "Injury",
    })
    table["Injury"] = table["Injury"].map({True: "risk", False: ""}).fillna("")

    st.caption(f"{len(table)} of {len(view)} players — click any column header to sort, "
               f"click a row for the three-year breakdown.")

    styled = (table.style
              .apply(tier_style, subset=["Tier"])
              .apply(flag_style, subset=["Sh% flag"])
              .format({"GP": "{:.1f}", "FPPG": "{:.2f}", "Total": "{:.1f}",
                       "VORP": "{:+.1f}", "VORP/G": "{:+.2f}",
                       "G": "{:.1f}", "A": "{:.1f}",
                       "S": "{:.1f}", "H": "{:.1f}", "B": "{:.1f}"}))

    event = st.dataframe(styled, width="stretch", hide_index=True,
                         height=520, on_select="rerun", selection_mode="single-row",
                         key="proj_table")

    download(table, "Download this view as CSV",
             f"nhl_proj2026-27_{weight_slug(weights)}.csv", "dl_proj")

    # st.dataframe returns a nested dict: {"selection": {"rows": [...], ...}}
    rows = list((event or {}).get("selection", {}).get("rows", []) or [])
    if rows:
        r = shown.iloc[rows[0]]
        st.divider()
        st.markdown(f"### {r['playerName']} — {r['position']}, {r['team']}")

        c = st.columns(6)
        c[0].metric("Projected GP", f"{r['gp']:.1f}")
        c[1].metric("Projected FPPG", f"{r['fppg']:.2f}")
        c[2].metric("Projected total", f"{r['fp']:.1f}")
        c[3].metric("VORP", f"{r['vorp']:+.1f}", f"{r['vorpPG']:+.2f} per game",
                delta_color="off")
        c[4].metric("Tier", r["tier"])
        c[5].metric("Age on Feb 1", f"{r['age']:.1f}" if pd.notna(r["age"]) else "—")

        seasons = hist[hist["playerId"] == r["playerId"]].sort_values("seasonId")

        # Two tables rather than one with holes: the projection has no TOI or
        # per-60 figures, and Streamlit renders a missing numeric cell as "None".
        counting = seasons[[
            "season", "teams", "gamesPlayed", "goals", "assists", "shots",
            "hits", "blockedShots", "shootingPct", "fp", "fppg",
        ]].rename(columns={
            "season": "Season", "teams": "Team", "gamesPlayed": "GP", "goals": "G",
            "assists": "A", "shots": "S", "hits": "H", "blockedShots": "B",
            "shootingPct": "Sh%", "fp": "FP", "fppg": "FPPG",
        })
        proj_row = pd.DataFrame([{
            "Season": "2026-27 proj", "Team": r["team"], "GP": r["gp"],
            "G": r["proj_goals"], "A": r["proj_assists"], "S": r["proj_shots"],
            "H": r["proj_hits"], "B": r["proj_blocks"], "Sh%": r["shPct_proj"],
            "FP": r["fp"], "FPPG": r["fppg"],
        }])
        combined = pd.concat([counting, proj_row], ignore_index=True)
        st.caption("Counting stats and fantasy points at the current weights")
        st.dataframe(combined.style.format(DETAIL_FMT), width="stretch",
                     hide_index=True)

        rates = seasons[[
            "season", "toiPerGame", "goalsPer60", "assistsPer60", "shotsPer60",
            "hitsPer60", "blocksPer60",
        ]].rename(columns={
            "season": "Season", "toiPerGame": "TOI/GP", "goalsPer60": "G/60",
            "assistsPer60": "A/60", "shotsPer60": "S/60", "hitsPer60": "H/60",
            "blocksPer60": "B/60",
        })
        st.caption("Ice time and per-60 rates (actual seasons only)")
        st.dataframe(rates.style.format(DETAIL_FMT), width="stretch", hide_index=True)

        left, right = st.columns([2, 3])
        with left:
            st.caption("FPPG by season, then the projection")
            st.altair_chart(trend_chart(sparkline_frame(seasons, r["fppg"]),
                                        "FPPG", "FPPG", "Series"), width="stretch")
        with right:
            st.caption("Why the projection lands where it does")
            st.dataframe(pd.DataFrame({
                "Metric": ["Confidence", "Trend label", "TOI/GP history → projected",
                           "TOI trend (min/season)", "Ice-time multiplier",
                           "Age multiplier (scoring / physical)",
                           "Slope FP/60", "Role volatility", "Sh% last → proj",
                           "Sh% flag", "Injury risk", "Career GP (3yr)"],
                "Value": [r["confidence"], r["trend_label"],
                          f"{r['toiHist']:.2f} → {r['toiProj']:.2f}",
                          f"{r['toi_trend']:+.2f}", f"{r['toiMult']:.3f}",
                          f"{r['ageMult_scoring']:.3f} / {r['ageMult_physical']:.3f}",
                          f"{r['slope_per60']:+.3f}", f"{r['role_volatility']:.3f}",
                          f"{r['shPct_last']} → {r['shPct_proj']}", r["shPct_flag"],
                          "yes" if r["injury_risk"] else "no", int(r["careerGP"])],
            }), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 -- history
# ---------------------------------------------------------------------------
with tab_hist:
    st.subheader("Three-year history (actuals)")

    h1, h2, h3, h4, h5 = st.columns([1.4, 1.9, 1, 1, 2])
    seasons_sel = h1.multiselect("Seasons", list(SEASON_LABEL.values()),
                                 default=list(SEASON_LABEL.values()), key="h_seasons")
    hpos = POS_CODE.get(h2.segmented_control(
        "Position", POS_CHOICES, default="All", key="h_pos") or "All")
    hteams = sorted({t for s in hist["teams"].fillna("") for t in str(s).split("/") if t})
    hteam = h3.selectbox("Team", ["all"] + hteams, key="h_team")
    hmin = h4.number_input("Min GP", 0, 82, 0, step=1, key="h_gp")
    hname = h5.text_input("Name contains", key="h_name")

    hm = hist["season"].isin(seasons_sel)
    if hpos:
        hm &= hist["positionGroup"] == hpos
    if hteam != "all":
        hm &= hist["teams"].fillna("").str.split("/").apply(lambda ts: hteam in ts)
    hm &= hist["gamesPlayed"] >= hmin
    if hname:
        hm &= hist["playerName"].str.contains(hname, case=False, na=False)

    hshown = hist[hm].sort_values("fppg", ascending=False).copy()
    hshown.insert(0, "#", range(1, len(hshown) + 1))

    htable = hshown[[
        "#",
        "playerName", "season", "position", "teams", "changedTeams", "gamesPlayed",
        "toiPerGame", "goals", "assists", "shots", "hits", "blockedShots",
        "shootingPct", "fp", "fppg", "fpPer60", "goalsPer60", "assistsPer60",
        "shotsPer60", "hitsPer60", "blocksPer60",
    ]].rename(columns={
        "playerName": "Player", "season": "Season", "position": "Pos",
        "teams": "Team", "changedTeams": "Traded", "gamesPlayed": "GP",
        "toiPerGame": "TOI/GP", "goals": "G", "assists": "A", "shots": "S",
        "hits": "H", "blockedShots": "B", "shootingPct": "Sh%", "fp": "FP",
        "fppg": "FPPG", "fpPer60": "FP/60", "goalsPer60": "G/60",
        "assistsPer60": "A/60", "shotsPer60": "S/60", "hitsPer60": "H/60",
        "blocksPer60": "B/60",
    })
    htable["Traded"] = htable["Traded"].map({True: "traded", False: ""}).fillna("")

    st.caption(f"{len(htable)} player-seasons — click any column header to sort.")
    st.dataframe(
        htable.style.format({
            "TOI/GP": "{:.2f}", "Sh%": "{:.1f}", "FP": "{:.1f}", "FPPG": "{:.2f}",
            "FP/60": "{:.2f}", "G/60": "{:.2f}", "A/60": "{:.2f}", "S/60": "{:.2f}",
            "H/60": "{:.2f}", "B/60": "{:.2f}"}),
        width="stretch", hide_index=True, height=460)

    download(htable, "Download this view as CSV",
             f"nhl_history_{weight_slug(weights)}.csv", "dl_hist")

    st.divider()
    st.markdown("#### Season-over-season for one player")
    # Keyed on playerId, not name: the league currently has two Sebastian Ahos
    # (CAR forward, NYI defenseman) and two Elias Petterssons (both VAN, one
    # forward one defenseman). Selecting by name would merge their careers.
    ident = hist[["playerId", "playerName", "position"]].drop_duplicates("playerId")
    ambiguous = ident.playerName.duplicated(keep=False)
    ident["label"] = [f"{n} ({p})" if amb else n
                      for n, p, amb in zip(ident.playerName, ident.position, ambiguous)]
    label_to_id = dict(zip(ident.label, ident.playerId))
    labels = sorted(label_to_id)
    default_i = labels.index("Cale Makar") if "Cale Makar" in labels else 0
    who_label = st.selectbox("Player", labels, index=default_i, key="h_player")
    who_id = label_to_id[who_label]

    pdata = hist[hist["playerId"] == who_id].sort_values("seasonId")
    pproj = view[view["playerId"] == who_id]

    m1, m2, m3 = st.columns(3)
    st.caption(f"{who_label} — playerId {who_id}")
    m1.metric("Seasons in window", len(pdata))
    m2.metric("Career GP (3yr)", int(pdata["gamesPlayed"].sum()))
    if not pproj.empty:
        last = pdata["fppg"].iloc[-1]
        m3.metric("2026-27 projected FPPG", f"{pproj.iloc[0]['fppg']:.2f}",
                  f"{pproj.iloc[0]['fppg'] - last:+.2f} vs 2025-26")

    g1, g2 = st.columns(2)
    with g1:
        st.caption("FPPG by season, then the projection")
        st.altair_chart(
            trend_chart(sparkline_frame(pdata, None if pproj.empty else pproj.iloc[0]["fppg"]),
                        "FPPG", "FPPG", "Series"), width="stretch")
    with g2:
        st.caption("TOI per game by season")
        toi_df = pdata.rename(columns={"season": "Season", "toiPerGame": "TOI/GP"})[
            ["Season", "TOI/GP"]]
        st.altair_chart(trend_chart(toi_df, "TOI/GP", "minutes"), width="stretch")

    st.dataframe(
        pdata[["season", "teams", "gamesPlayed", "toiPerGame", "goals", "assists",
               "shots", "hits", "blockedShots", "shootingPct", "fp", "fppg",
               "fpPer60"]]
        .rename(columns={"season": "Season", "teams": "Team", "gamesPlayed": "GP",
                         "toiPerGame": "TOI/GP", "goals": "G", "assists": "A",
                         "shots": "S", "hits": "H", "blockedShots": "B",
                         "shootingPct": "Sh%", "fp": "FP", "fppg": "FPPG",
                         "fpPer60": "FP/60"})
        .style.format(DETAIL_FMT, na_rep="—"),
        width="stretch", hide_index=True)
