import requests
import pandas as pd
from datetime import datetime

# Initial API Call
key = "78aac9217e5b16253e383fb61661f079"
sport = "basketball_nba"
region = "us"
attribute = "spreads"

url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
params = {
    "apiKey": key,
    "regions": region,
    "markets": attribute,
    "oddsFormat": "american"
}

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
            if market["key"] != "spreads":
                continue

            for outcome in market["outcomes"]:
                cleansed.append({
                    "tip off": time,
                    "game": away + " at " + home,
                    "book": book,
                    "favorite": outcome["name"],
                    "spread": outcome["point"],
                    "odds": outcome["price"]
                })

# Build DataFrame
api_data = pd.DataFrame(cleansed)

# Sort properly
api_data = api_data.sort_values(by="tip off").reset_index(drop=True)

# Show result
print(api_data)
