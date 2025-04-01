import requests
import pandas as pd
from datetime import datetime

# Initial API Call
key = "78aac9217e5b16253e383fb61661f079"
region = "us"
attribute = "spreads"

#url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"

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
    league = match["sport_title"]

    for site in match["bookmakers"]:
        book = site["title"]
        
        for market in site["markets"]:
            if market["key"] != "spreads":
                continue

            for outcome in market["outcomes"]:
                cleansed.append({
                    "league" : league,
                    "prop" :  away.split()[-1] + " at " + home.split()[-1] + " spread",
                    "game-time": time.astimezone().strftime("%-I:%M %p"),
                    "game": away + " at " + home,
                    "book": book,
                    "favorite": outcome["name"],
                    "spread": outcome["point"],
                    "odds": outcome["price"]
                })

api_data = pd.DataFrame(cleansed)

api_data = api_data.sort_values(by="game-time").reset_index(drop=True)

#print(api_data)

# If we want to only look at one sport 
NHL_spreads= api_data[api_data["league"] == "NHL"]
print(NHL_spreads)
