import requests
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Initial API Key for access
key = "78aac9217e5b16253e383fb61661f079"

#Fetches available sports from the Odds-API
url = "https://api.the-odds-api.com/v4/sports"
params = {
    "apiKey": key
}

#List of all sporting event/games on a given day
response = requests.get(url, params=params)
sports_data = response.json()

# Key name for all sports
print("Basketball Key")
for sport in sports_data:
    print(f"{sport['title']} Key: {sport['key']}")

#Filter by Sport
SPORT = "basketball_nba"  
REGION = "us"

#Since we are interested in spreads, we can utilize this metric
MARKET = "spreads"

# Initial API Call to get odds for the selected sport
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "apiKey": key,
    "regions": REGION,
    "markets": MARKET,
    "oddsFormat": "american"
}

# Fetch odds data
response = requests.get(url, params=params)
data = response.json()

# Data Cleaning
cleansed = []

for match in data:
    time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
    home = match["home_team"]
    away = match["away_team"]

    for site in match["bookmakers"]:
        book = site["title"]
        
        for market in site["markets"]:
            if market["key"] != MARKET:
                continue

            for outcome in market["outcomes"]:
                cleansed.append({
                    "tip-off": time,
                    "home_team": home,
                    "away_team": away,
                    "sportsbook": book,
                    "team": outcome["name"],
                    "spread": outcome["point"],
                    "odds": outcome["price"]
                })

#Stores all of the data from the Cleansed Data
api_data = pd.DataFrame(cleansed)

# Sort properly
api_data = api_data.sort_values(by="tip-off").reset_index(drop=True)
#print(api_data)

# +
#Creates Matchup Column for Each Game
api_data["matchup"] = api_data["away_team"] + " @ " + api_data["home_team"]

#Calculate Win Probability
def implied_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

api_data["implied_prob"] = api_data["odds"].apply(implied_prob)

#Spread Score (Higher Negative Values = More Favorite)
api_data["spread_score"] = -api_data["spread"]

#Win Score based off of spread and probability
api_data["win_score"] = 0.5 * api_data["spread_score"] + 0.5 * api_data["implied_prob"]

#Average Scores across all sportsbooks
avg_scores = api_data.groupby(["matchup", "team", "home_team", "away_team"]).agg({
    "spread": "mean",
    "odds": "mean",
    "implied_prob": "mean",
    "spread_score": "mean",
    "win_score": "mean"
}).reset_index()

# Pick winner per matchup using averaged win scores
predictions = (
    avg_scores.loc[avg_scores.groupby("matchup")["win_score"].idxmax()]
    .rename(columns={"team": "predicted_winner"})
    .sort_values(by="matchup")
    .reset_index(drop=True)
)


# -

for matchup, subset in avg_scores.groupby("matchup"):
    if len(subset) < 2:
        continue

    teams = subset["team"].tolist()
    scores = subset["win_score"].tolist()
    winner_index = np.argmax(scores)

    x = np.arange(len(teams))
    colors = ["gray"] * len(scores)
    colors[winner_index] = "seagreen"

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, scores, width=0.4, color=colors, edgecolor="black")

    for i, bar in enumerate(bars):
        offset = 0.05 if scores[i] >= 0 else -0.1
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, f"{scores[i]:.2f}", ha="center", fontsize=10)

    plt.xticks(x, teams)
    plt.ylim(min(min(scores), 0) - 1, max(scores) + 1)
    plt.title(f"Matchup: {matchup}\nProjected Winner: {teams[winner_index]}")
    plt.ylabel("Win Score")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


#Odds Per Sportsbook
avg_odds = api_data.groupby("sportsbook")["odds"].mean().sort_values()
avg_odds.plot(kind="barh", color="mediumseagreen", edgecolor="black")
plt.title("Average Odds per Sportsbook")
plt.xlabel("Average Odds")
plt.ylabel("Sportsbook")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
