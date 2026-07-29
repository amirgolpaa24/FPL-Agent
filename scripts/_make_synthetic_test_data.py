"""
Not part of the real pipeline. Generates a tiny synthetic season that mimics
merged_gw.csv's schema, purely so build_dataset.py / backtest.py can be
sanity-tested without needing real internet access.
"""
import os
import random

import pandas as pd

random.seed(42)

TEAMS = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Newcastle", "Villa"]
TEAM_IDS = {t: i + 1 for i, t in enumerate(TEAMS)}

# name, position, team, base_minutes (higher = more nailed), skill (higher = scores more)
PLAYERS = [
    ("Star Striker", "FWD", "Arsenal", 90, 6),
    ("Star Winger", "MID", "Liverpool", 85, 5),
    ("Steady Def", "DEF", "Man City", 100, 4),
    ("Rotation Mid", "MID", "Chelsea", 45, 2),
    ("Bench Warmer", "FWD", "Newcastle", 15, 2),
    ("Solid GK", "GK", "Villa", 90, 3),
    ("Newcastle Fwd", "FWD", "Newcastle", 80, 4),
    ("Villa Mid", "MID", "Villa", 75, 3),
    ("Chelsea Def", "DEF", "Chelsea", 90, 3),
    ("City Mid", "MID", "Man City", 85, 5),
]

N_GW = 12
rows = []

for gw in range(1, N_GW + 1):
    shuffled = list(TEAMS)
    random.shuffle(shuffled)
    pairings = list(zip(shuffled[::2], shuffled[1::2]))

    goals_for, goals_against, opponent_of, home_of = {}, {}, {}, {}
    for home, away in pairings:
        h_score = random.choice([0, 1, 1, 2, 2, 3])
        a_score = random.choice([0, 0, 1, 1, 2])
        goals_for[home], goals_against[home] = h_score, a_score
        goals_for[away], goals_against[away] = a_score, h_score
        opponent_of[home], opponent_of[away] = away, home
        home_of[home], home_of[away] = True, False

    for name, pos, team, base_minutes, skill in PLAYERS:
        minutes = max(0, min(90, int(random.gauss(base_minutes, 15))))
        starts = 1 if minutes > 60 else 0
        total_points = max(0, int(random.gauss(skill * (minutes / 90), 2)))
        was_home = home_of[team]

        rows.append({
            "name": name,
            "position": pos,
            "team": team,
            "opponent_team": TEAM_IDS[opponent_of[team]],
            "was_home": was_home,
            "GW": gw,
            "value": base_minutes // 10 + 40,
            "minutes": minutes,
            "starts": starts,
            "total_points": total_points,
            "ict_index": round(total_points * 1.5 + random.random(), 1),
            "team_h_score": goals_for[team] if was_home else goals_against[team],
            "team_a_score": goals_against[team] if was_home else goals_for[team],
        })

df = pd.DataFrame(rows)

out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "test-season")
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, "merged_gw.csv"), index=False)

teams_df = pd.DataFrame({"id": list(TEAM_IDS.values()), "name": list(TEAM_IDS.keys())})
teams_df.to_csv(os.path.join(out_dir, "teams.csv"), index=False)

print(f"Wrote {len(df)} rows across {N_GW} gameweeks, {len(PLAYERS)} players, "
      f"{len(TEAMS)} teams to {out_dir}")
