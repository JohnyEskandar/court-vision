# # Libraries

import requests
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# # API Key Generation

# Initial API Key for access
key = "78aac9217e5b16253e383fb61661f079"

# # General API that we Chose

#Fetches available sports from the Odds-API
url = "https://api.the-odds-api.com/v4/sports"
params = {
    "apiKey": key
}

# # Filtering API via NBA Basketball

#Specify Sport based on Key and Region
sport = "basketball_nba"  
region = "us"

#Since we are interested in spreads, we can utilize this metric
metric = "spreads"

# Initial API Call to get odds for the selected sport
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {
    "apiKey": key,
    "regions": region,
    "markets": metric,
    "oddsFormat": "american"
}

# Fetch odds data
response = requests.get(url, params=params)
data = response.json()


# # Now we have all the data we Need; however, it is very messy as seen below so uncomment at your choice

# +
#print(data)
# -

# # Beginning the Data Cleaning Process

#Stores Data in Cleansed
cleansed = []

# #### Taking in some additional metrics

for match in data:
    time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
    home = match["home_team"]
    away = match["away_team"]

    for site in match["bookmakers"]:
        book = site["title"]
        
        for metrics in site["markets"]:
            if metrics["key"] != metric:
                continue

            for outcome in metrics["outcomes"]:
                cleansed.append({
                    "tip-off": time,
                    "home_team": home,
                    "away_team": away,
                    "sportsbook": book,
                    "team": outcome["name"],
                    "spread": outcome["point"],
                    
                    #Minus = Favorite
                    #Plus = Underdog
                    "odds": outcome["price"]
                })

# #### Creating a dataframe of the data we just extracted

#Stores all of the data from the Cleansed Data
api_data = pd.DataFrame(cleansed)


# #### All Different Sportsbooks

#All of the Sportsbooks used within the API
api_data["sportsbook"].unique()

# # The Current Code is Using All of the Sportsbooks

# +
#Uncomment This line and change to desired Sportsbook if needed
#api_data = api_data[api_data["sportsbook"] == "DraftKings"]

#Sorts in chronological order of games
api_data.sort_values(by="tip-off").reset_index(drop=True)
# -

# # Appending Columns to our Data

# #### Appending each Game

#Away team at Home Team
api_data["matchup"] = api_data["away_team"] + " @ " + api_data["home_team"]

# #### Appending a Spread Score

#Negates the value of the spread to give a spread score
api_data["spread_score"] =  api_data["spread"]


# ###### Converts Spread Odds to a Probability of Hitting

# +
#What is the Percent Change that the Spread Odd is Covered?
def spread_odd_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else: 
        return -odds / (-odds + 100)

#Append the Spread Probability to the Data
api_data["spread_probability"] = api_data["odds"].apply(spread_odd_probability)
# -

# # Win Score Calculation

api_data["win_score"] = 0.5 * api_data["spread_score"] + 0.5 * api_data["spread_probability"]

# # Plotting

# ### Matchup Winners

# +
for matchup, game in avg_scores.groupby("matchup"):
    #Has to be 2 teams per matchup
    if len(game) != 2:
        continue

    #Extract Team, Win Scores, and Projected Winner
    teams = game["team"].values
    scores = game["win_score"].values
    winner = teams[np.argmax(scores)]

    colors = ["red", "red"]
    colors[np.argmax(scores)] = "green"

    x = np.arange(2)
    bars = plt.bar(x, scores, color=colors)

    for i, bar in enumerate(bars):
        y = bar.get_height()
        if y >=0:
            pos = 0.05
        else:
            pos = -0.1
        plt.text(bar.get_x() + bar.get_width()/2, y + pos, f"{scores[i]:.2f}", ha="center", fontsize=10)

    plt.xticks(x, teams)
    plt.ylim(min(min(scores), 0) - 1, max(scores) + 1)
    plt.title(f"{matchup}\nPredicted Winner: {winner}", fontsize=12)
    plt.ylabel("Win Score")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()




# -
# ### Average Spread Odds by Book

# +
sportsbook_odds= api_data.groupby("sportsbook")["odds"].mean().sort_values()

sportsbook_odds.plot(kind="barh", color="steelblue", edgecolor="black")

plt.axvline(-110, color='red', linestyle='--', linewidth=1.5, label='Industry Standard')

plt.title("Average Odds by Sportsbook")
plt.xlabel("Average Odds")
plt.ylabel("Sportsbook")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# -

# ### Bets placed Per Book

# +
book_counts = api_data["sportsbook"].value_counts().sort_values(ascending=True)

book_counts.plot(kind="barh", color="blue", edgecolor="black")
plt.title("Bets per Sportsbook")
plt.xlabel("Number of Bets")
plt.ylabel("Sportsbook")
plt.tight_layout()
plt.show()



