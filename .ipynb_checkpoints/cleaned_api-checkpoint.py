import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Initial API Key for access
API_KEY = "78aac9217e5b16253e383fb61661f079"

#Fetches available sports from the Odds-API
url = "https://api.the-odds-api.com/v4/sports"
params = {
    "apiKey": API_KEY
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
    "apiKey": API_KEY,
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

print("\nOdds Data for Selected Sport:")
# print(api_data)



#Initial Plotting 

# Odds Per Sportsbook
# avg_odds = api_data.groupby("sportsbook")["odds"].mean().sort_values()
# avg_odds.plot(kind="barh", color="mediumseagreen", edgecolor="black")
# plt.title("Average Odds per Sportsbook")
# plt.xlabel("Average Odds")
# plt.ylabel("Sportsbook")
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

game_odds = api_data.groupby("game")["odds"].mean().sort_values()

game_odds.plot(kind="barh", color="mediumseagreen", edgecolor="black")
plt.title("Average Odds per Game")
plt.xlabel("Odds")
plt.ylabel("Game")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()