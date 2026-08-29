#!/usr/bin/env python3
"""
draft_sim.py -- 12-team snake draft, 5 rounds, picking from a given slot.

The other 11 teams take the best available player by projected season fantasy
points. That makes the board fully deterministic given my own picks, so instead
of guessing I search over the top-K available at each of my picks and keep the
combination with the most total VORP.

    python3 draft_sim.py --slot 6 --rounds 5
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

TEAMS, F_SLOTS, D_SLOTS = 12, 9, 4
TIER_MIN_GP = 40


def my_overall_picks(slot: int, rounds: int, teams: int = TEAMS) -> list[int]:
    picks = []
    for r in range(rounds):
        pos = slot if r % 2 == 0 else teams - slot + 1
        picks.append(r * teams + pos)
    return picks


def board_arrays(board):
    return (board.playerId.tolist(), board.playerName.tolist(),
            board.position.tolist(), board.positionGroup.tolist(),
            board.fantasyPoints_default.tolist(), board.vorp_default.tolist(),
            board.tier_default.tolist())


def run_draft(n_board, mine, my_picks, total_picks):
    """Fast walk of the draft. Returns (pick_log, available_index_at_my_picks)."""
    taken = set()
    log, avail_at = [], []
    ci = 0
    cursor = 0
    for overall in range(1, total_picks + 1):
        while cursor < n_board and cursor in taken:
            cursor += 1
        if overall in my_picks:
            avail_at.append(set(taken))
            if ci < len(mine):
                idx = mine[ci]; ci += 1
            else:
                idx = cursor
            log.append(("ME", idx))
        else:
            idx = cursor
            log.append(("BPA", idx))
        taken.add(idx)
        if idx == cursor:
            cursor += 1
    return log, avail_at


def first_available(taken, n, start=0):
    i = start
    while i < n and i in taken:
        i += 1
    return i


def optimal_roster(board, my_picks, k, beam_width, f_slots, d_slots):
    """
    Beam search over a full roster. Other teams take best available, so the
    board is deterministic given my picks; the objective is total VORP of a
    legal 9F/4D roster, not of the first five picks alone.
    """
    pgs = board.positionGroup.tolist()
    vorp = board.vorp_default.tolist()
    n = len(board)
    rounds = len(my_picks)

    beam = [([], 0.0, 0, 0)]          # picks(idx), vorp, nF, nD
    for r in range(rounds):
        nxt = []
        for chosen, score, nf, nd in beam:
            taken = set()
            ci = 0
            cursor = 0
            for overall in range(1, my_picks[r] + 1):
                cursor = first_available(taken, n, cursor)
                if overall in my_picks:
                    if ci < len(chosen):
                        idx = chosen[ci]; ci += 1
                    else:
                        break
                else:
                    idx = cursor
                taken.add(idx)
            cand = []
            i = 0
            while len(cand) < k and i < n:
                if i not in taken:
                    cand.append(i)
                i += 1
            for idx in cand:
                is_d = pgs[idx] == "D"
                nf2, nd2 = nf + (not is_d), nd + is_d
                if nd2 > d_slots or nf2 > f_slots:
                    continue
                left = rounds - (r + 1)
                if (f_slots - nf2) + (d_slots - nd2) < left:
                    continue
                nxt.append((chosen + [idx], score + vorp[idx], nf2, nd2))
        nxt.sort(key=lambda t: -t[1])
        beam = nxt[:beam_width]
        print(f"  round {r+1}/{rounds}: {len(nxt)} states -> best VORP so far "
              f"{beam[0][1]:+.1f}")
    return beam[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=13)
    ap.add_argument("--show", type=int, default=5, help="picks to detail")
    ap.add_argument("--beam", type=int, default=600)
    ap.add_argument("--k", type=int, default=14, help="candidates considered per pick")
    a = ap.parse_args()

    df = pd.read_csv("data/projections_2026_27.csv")
    board = (df[df.gp_proj >= TIER_MIN_GP]
             .sort_values("fantasyPoints_default", ascending=False)
             .reset_index(drop=True))
    print(f"Draftable pool: {len(board)} players (gp_proj >= {TIER_MIN_GP}), "
          f"ranked by projected season total at default scoring.")

    my_picks = my_overall_picks(a.slot, a.rounds)
    print(f"Picking {a.slot} of {TEAMS}, {a.rounds} rounds -> overall {my_picks}\n")

    best, best_vorp, nf, nd = optimal_roster(board, my_picks, a.k, a.beam,
                                             F_SLOTS, D_SLOTS)
    log, _ = run_draft(len(board), best, my_picks, my_picks[-1])
    name = board.playerName.tolist(); pos = board.position.tolist()
    pg = board.positionGroup.tolist(); fp = board.fantasyPoints_default.tolist()
    vp = board.vorp_default.tolist(); tier = board.tier_default.tolist()

    print(f"\nMY ROSTER ({nf}F / {nd}D)  total VORP {best_vorp:+.1f}, "
          f"{sum(fp[i] for i in best):.0f} projected points")
    for i, idx in enumerate(best, 1):
        star = "  <-- first five" if i <= a.show else ""
        print(f"  R{i:>2} (#{my_picks[i-1]:>3}): {name[idx][:22]:<23}{pos[idx]:<3} "
              f"{tier[idx]:<4}{fp[idx]:>7.1f}{vp[idx]:>+8.1f}{star}")

    print(f"\nHOW THE FIRST {a.show} ROUNDS PLAY OUT")
    print(f"{'#':>4} {'team':<5} {'player':<24}{'pos':<4}{'tier':<5}{'total':>8}{'vorp':>8}")
    for overall, (who, idx) in enumerate(log[:a.show * TEAMS], 1):
        tag = "ME" if who == "ME" else f"T{((overall - 1) % TEAMS) + 1}"
        mark = "  <<<" if who == "ME" else ""
        print(f"{overall:>4} {tag:<5} {name[idx][:23]:<24}{pos[idx]:<4}{tier[idx]:<5}"
              f"{fp[idx]:>8.1f}{vp[idx]:>+8.1f}{mark}")

    # Pure best-available comparison over the same 13 rounds.
    bpa_log, _ = run_draft(len(board), [], my_picks, my_picks[-1])
    bpa = [idx for who, idx in bpa_log if who == "ME"]
    bf = sum(1 for i in bpa if pg[i] == "F"); bd = len(bpa) - bf
    print(f"\nPure best-available for all 13 picks: {sum(fp[i] for i in bpa):.0f} pts, "
          f"VORP {sum(vp[i] for i in bpa):+.1f}  ({bf}F/{bd}D"
          f"{' -- ILLEGAL ROSTER' if bd > D_SLOTS or bf > F_SLOTS else ''})")
    print("  first five: " + ", ".join(name[i] for i in bpa[:a.show]))


if __name__ == "__main__":
    main()
