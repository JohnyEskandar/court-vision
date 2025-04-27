import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from functools import reduce
import statsmodels.api as sm
from scipy.stats import ttest_ind

#
st.set_page_config(page_title="Court Vision", layout="wide")
st.title("Court Vision: Revamping Sports Betting")

#API Key and URL Declaration
a_key = "Your api key here " #Create an API Key from the Odds API
a_url = "https://api.the-odds-api.com/v4/sports"

# helper methods

#Converts Live Odds to Implied Probability
def odds_conversion(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

#American Odds to Decimal Format
def decimal_conversion(odds):
    return odds / 100 + 1 if odds > 0 else -100 / odds + 1

#Fetches and Connects API While Retrieving all Available Sports
def fetch_and_connect():
    try:
        response = requests.get(a_url, params={"apiKey": a_key})
        response.raise_for_status()
        all_sports = response.json()
        st.success("Successfully connected to sports data API")
        return [sport for sport in all_sports if sport["title"] in {"NBA", "MLB", "NHL"}]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {str(e)}")
        return []

#Gets the Key for NBA, MLB, and NHL
def fetch_odds(sport_key):
    url = f"{a_url}/{sport_key}/odds"
    params = {
        "apiKey": a_key,
        "regions": "us",
        "markets": "spreads,h2h,totals",
        "oddsFormat": "american"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch odds data: {str(e)}")
        return []

#Data Cleaning 
def raw_cleaning(raw_data):
    processed_matches = []
    
    for game in raw_data:
        game_time = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
        home_team = game["home_team"]
        away_team = game["away_team"]
        matchup = f"{away_team} @ {home_team}"
        
        for bookmaker in game.get("bookmakers", []):
            bookmaker_name = bookmaker["title"]
            
            for market in bookmaker.get("markets", []):
                for outcome in market["outcomes"]:
                    entry = {
                        "game_time": game_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "matchup": matchup,
                        "market_type": market["key"],
                        "sportsbook": bookmaker_name,
                        "team": outcome["name"],
                        "odds": outcome["price"]
                    }
                    
                    if "point" in outcome:
                        entry["point"] = outcome["point"]
                        
                    processed_matches.append(entry)
                    
    return pd.DataFrame(processed_matches)

#Win Score Metric
def win_score_calculation(spread_data):
    if spread_data.empty:
        return spread_data
        
    spreads = spread_data.copy()
    spreads["spread_score"] = -spreads["point"]
    spreads["implied_prob"] = spreads["odds"].apply(odds_conversion)
    spreads["win_score"] = 0.5 * spreads["spread_score"] + 0.5 * spreads["implied_prob"]
    
    # Normalize scores to probability range
    spreads["estimated_win_prob"] = (
        (spreads["win_score"] - spreads["win_score"].min()) / 
        (spreads["win_score"].max() - spreads["win_score"].min())
    )
    
    return spreads

#Finding Value Bets
def value_bets(spread_data, moneyline_data, stake=100):
    if moneyline_data.empty:
        return pd.DataFrame()
    
    value_calc = moneyline_data[["team", "matchup", "odds", "win_probability"]].merge(
        spread_data[["team", "win_score"]], 
        on="team", 
        how="left"
    )
    
    value_calc["value_indicator"] = value_calc["win_score"] - value_calc["win_probability"]
    
    #Measuring Payouts from Odds
    def calculate_expected_return(prob, odds, stake=100):
        payout = (odds / 100) * stake if odds > 0 else (100 / abs(odds)) * stake
        return (prob * payout) + ((1 - prob) * (-stake))

    value_calc["expected_value"] = value_calc.apply(
        lambda row: round(calculate_expected_return(row["win_probability"], row["odds"], stake), 2),
        axis=1
    )
    
    value_calc["ev_ratio"] = (value_calc["expected_value"] / stake).round(4)
    
    best_bets = value_calc.loc[value_calc.groupby("team")["value_indicator"].idxmax()]
    
    #Filtering
    return best_bets[best_bets["odds"] > -200].sort_values("ev_ratio", ascending=False)

#Average Bias Per Book
def average_book_bias(totals_data):
    if totals_data.empty:
        return pd.Series(dtype=float)
        
    totals_data["implied_prob"] = totals_data["odds"].apply(odds_conversion)
    
    prob_comparison = totals_data.pivot_table(
        index=["matchup", "sportsbook", "point"], 
        columns="team", 
        values="implied_prob"
    ).reset_index()
    
    prob_comparison["bias_score"] = prob_comparison.get("Over", np.nan) - prob_comparison.get("Under", np.nan)
    filtered_data = prob_comparison.dropna(subset=["bias_score"])
    
    return filtered_data.groupby("sportsbook")["bias_score"].mean().sort_values()

#Calculating Payouts
def parlay_payouts(bets, stake=100):
    decimal_odds = [decimal_conversion(bet['odds']) for bet in bets]
    combined_odds = reduce(lambda x, y: x * y, decimal_odds)
    total_payout = stake * combined_odds
    net_profit = total_payout - stake
    return combined_odds, total_payout, net_profit

#Classifying Teams as Safe, Risky, or Upsets
def recommended_bets(moneyline_data):
    if moneyline_data.empty:
        st.warning("Moneyline Data Not Available")
        return
    
    avg_probabilities = moneyline_data.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
    categories = {
        "Safe": [], 
        "50/50": [], 
        "Upsets": []
    }
    
    for matchup, team_data in avg_probabilities.groupby("matchup"):
        if len(team_data) != 2:
            continue
            
        # Normalize Implied Probabilities to sum to 100%
        total_prob = team_data["win_probability"].sum()
        team_data["normalized_prob"] = (team_data["win_probability"] / total_prob) * 100
        
        for _, row in team_data.iterrows():
            pick = {
                "Team": row["team"], 
                "Matchup": matchup, 
                "Win %": round(row["normalized_prob"], 2)
            }
            prob = row["normalized_prob"]
            
            if prob > 65:
                categories["Safe"].append(pick)
            elif 48 <= prob <= 52:
                categories["50/50"].append(pick)
            elif 40 <= prob < 48:
                categories["Upsets"].append(pick)
    
    for category, picks in categories.items():
        st.markdown(f"### {category}")
        if picks:
            st.dataframe(
                pd.DataFrame(picks).sort_values("Win %", ascending=False),
                column_config={"Win %": st.column_config.NumberColumn(format="%.1f%%")}
            )
        else:
            st.info(f"No {category.lower()} identified")

#Displaying All Value Bets
def display_value(value_bets):
    st.subheader("Top Value Bets")
    if not value_bets.empty:
        st.dataframe(
            value_bets[["team", "matchup", "odds", "win_probability", "win_score", "value_indicator"]].head(10),
            height=400,
            column_config={
                "win_probability": "Win Probability",
                "win_score": "Win score",
                "value_indicator": "Value Score"
            }
        )
    else:
        st.warning("No Strong Bets")

#Team Win Score Display
def display_teams(spread_data):
    st.subheader("Win Score Analysis")
    
    if not spread_data.empty:
        grouped = spread_data.groupby(["matchup", "team"]).agg({"win_score": "mean"}).reset_index()
        
        for matchup, team_data in grouped.groupby("matchup"):
            if len(team_data) != 2:
                continue
                
            teams = team_data["team"].values
            scores = team_data["win_score"].values
            predicted_winner = teams[np.argmax(scores)]
            
            fig, ax = plt.subplots()
            colors = ["#4CAF50" if score == max(scores) else "#F44336" for score in scores]
            bars = ax.bar(teams, scores, color=colors)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + 0.05, 
                    f"{height:.2f}", 
                    ha="center",
                    va="bottom"
                )
                
            ax.set_title(f"{matchup}\nPredicted Winner: {predicted_winner}")
            ax.set_ylabel("Strength Score")
            plt.xticks(rotation=15)
            st.pyplot(fig)
    else:
        st.warning("Spread Data Unavailable")

#Display Win Probabilites per Team
def display_win_probs(moneyline_data):
    st.subheader("Win Probability Breakdown")
    
    if not moneyline_data.empty:   
        avg_probabilities = moneyline_data.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
        
        for matchup, team_data in avg_probabilities.groupby("matchup"):
            if len(team_data) != 2:
                continue
                
            teams = team_data["team"].values
            probs = (team_data["win_probability"].values / team_data["win_probability"].sum()) * 100
            
            fig, ax = plt.subplots()
            bars = ax.bar(teams, probs, color="#2196F3")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + 0.01, 
                    f"{height:.1f}%", 
                    ha="center",
                    va="bottom"
                )
                
            ax.set_title(f"{matchup} - Win Probability Comparison")
            ax.set_ylabel("Probability (%)")
            plt.xticks(rotation=15)
            st.pyplot(fig)
    else:
        st.warning("Moneyline Data Unavailable")

#Display the Average Book Bias for each Sports
def display_bias(bias_data):
    st.subheader("Sportsbook Tendencies")
    
    if not bias_data.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        bias_data.plot(kind="barh", color="#FF9800", edgecolor="black", ax=ax)
        ax.axvline(0, linestyle="--", color="black", alpha=0.7)
        ax.set_title("Sportsbook Over/Under Biases")
        ax.set_xlabel("Bias Score (Positive = Favors Over)")
        ax.set_ylabel("Sportsbook")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        st.pyplot(fig)
    else:
        st.warning("Totals Data Unavailable")

#Interactive Parlay for Users
def display_parlay_creator(moneyline_data):
    st.subheader("Parlay Constructor")
    
    if not moneyline_data.empty:
        available_books = moneyline_data["sportsbook"].unique()
        selected_book = st.selectbox(
            "Choose Sportsbook:", 
            available_books, 
            key="parlay_book_selector"
        )
        
        book_data = moneyline_data[
            (moneyline_data["sportsbook"] == selected_book) & 
            (moneyline_data["market_type"] == "h2h")
        ].copy()
        
        if book_data.empty:
            st.warning(f"Moneyline data Unavailable from {selected_book}")
            return
            
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown(f"### {selected_book} Game Lines")
            
            st.info("""
                **Note:Some games show multiple odds for the same team.  
                - First odds are current live odds  
                - Second set shows opening odds  
                - This helps track line movement
            """)
            
            matchups = book_data[["matchup", "team", "odds"]].drop_duplicates()
            matchup_options = matchups["matchup"].unique()
            
            selected_matchup = st.selectbox(
                "Select Game:", 
                matchup_options, 
                key="parlay_game_selector"
            )
            
            matchup_odds = matchups[matchups["matchup"] == selected_matchup]
            st.dataframe(
                matchup_odds[["team", "odds"]].reset_index(drop=True), 
                hide_index=True,
                column_config={
                    "team": "Team",
                    "odds": "Odds"
                }
            )
            
            selected_team = st.selectbox(
                "Pick Team:", 
                matchup_odds["team"].unique(), 
                key="parlay_team_selector"
            )
            
            selected_odds = matchup_odds[matchup_odds["team"] == selected_team]["odds"].iloc[0]
            
            if st.button("Add to Parlay", key="add_parlay_leg"):
                if 'parlay_bets' not in st.session_state:
                    st.session_state.parlay_bets = []
                    
                if any(
                    bet["team"] == selected_team and bet["matchup"] == selected_matchup 
                    for bet in st.session_state.parlay_bets
                ):
                    st.warning("Cannot Add Same Team Again")
                else:
                    st.session_state.parlay_bets.append({
                        "sportsbook": selected_book,
                        "matchup": selected_matchup,
                        "team": selected_team,
                        "odds": selected_odds
                    })
                    st.rerun()
        
        with col2:
            st.markdown("### Bet Slip")
            
            if not st.session_state.get('parlay_bets'):
                st.info("Empty Parlay")
            else:
                unique_books = {bet["sportsbook"] for bet in st.session_state.parlay_bets}
                if len(unique_books) > 1:
                    st.error("Error: All Bets Must Be from the Same Sportsbook")
                
                for i, bet in enumerate(st.session_state.parlay_bets):
                    cols = st.columns([4, 1])
                    cols[0].write(f"{bet['team']} ({bet['odds']}) - {bet['matchup']}")
                    if cols[1].button(" ", key=f"remove_{i}"):
                        st.session_state.parlay_bets.pop(i)
                        st.rerun()
                
                stake = st.number_input(
                    "Enter Wager Amount:", 
                    min_value=1, 
                    value=100, 
                    step=1, 
                    key="parlay_stake"
                )
                
                if st.session_state.parlay_bets:
                    odds, payout, profit = parlay_payouts(
                        st.session_state.parlay_bets, 
                        stake
                    )
                    
                    st.markdown("---")
                    st.markdown(f"**Sportsbook:** {selected_book}")
                    st.markdown(f"**Number of Legs:** {len(st.session_state.parlay_bets)}")
                    st.markdown(f"**Combined Odds:** {odds:.2f}x")
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Potential Payout", f"${payout:,.2f}")
                    col_b.metric("Potential Profit", f"${profit:,.2f}")
                
                if st.button("Clear Parlay", key="clear_parlay"):
                    st.session_state.parlay_bets = []
                    st.rerun()
    else:
        st.warning("No Moneyline Data Available")

#EV Score Display
def display_EV_analysis(value_bets):
    st.subheader("Expected Value Opportunities")
    
    if not value_bets.empty:
        value_bets["EV Rating"] = value_bets["expected_value"].apply(
            lambda ev: "Positive" if ev > 0 else "Negative"
        )
        
        st.dataframe(
            value_bets[[
                "team", "matchup", "odds", "win_probability", 
                "expected_value", "ev_ratio", "EV Rating"
            ]].round({
                "win_probability": 2,
                "expected_value": 2,
                "ev_ratio": 4
            }),
            height=600,
            column_config={
                "team": "Team",
                "matchup": "Game",
                "odds": "Odds",
                "win_probability": st.column_config.NumberColumn("Win %", format="%.2f"),
                "expected_value": st.column_config.NumberColumn("Expected Value", format="%.2f"),
                "ev_ratio": st.column_config.NumberColumn("EV Ratio", format="%.4f"),
                "EV Rating": "EV Assessment"
            }
        )
    else:
        st.warning("No +EV opportunities")

#Executing Main Method and Beginning the Session
def main():
    if 'parlay_bets' not in st.session_state:
        st.session_state.parlay_bets = []
    
    st.sidebar.header("Sport Selection")
    available_sports = fetch_and_connect()
    if not available_sports:
        return
        
    sport_names = [sport['title'] for sport in available_sports]
    selected_sport = st.sidebar.selectbox("Choose Sport:", sport_names)
    sport_key = next(sport['key'] for sport in available_sports if sport['title'] == selected_sport)
    
    raw_data = fetch_odds(sport_key)
    if not raw_data:
        return
        
    processed_data = raw_cleaning(raw_data)
    spread_analysis = win_score_calculation(processed_data[processed_data["market_type"] == "spreads"])
    
    moneyline_data = processed_data[processed_data['market_type'] == 'h2h'].copy()
    moneyline_data = moneyline_data.merge(
        spread_analysis[["team", "estimated_win_prob"]], 
        on="team", 
        how="left"
    )
    moneyline_data["win_probability"] = moneyline_data["estimated_win_prob"]
    
    top_value_bets = value_bets(spread_analysis, moneyline_data)
    bookmaker_biases = average_book_bias(processed_data[processed_data["market_type"] == "totals"])
    
    analysis_tabs = st.tabs([
        "Value Bets", 
        "Win Score Analysis", 
        "Win Probabilities", 
        "Book Biases",
        "Our Recommendations", 
        "EV Analysis",
        "Mock Parlay Builder"
    ])
    
    with analysis_tabs[0]:
        display_value(top_value_bets)
    with analysis_tabs[1]:
        display_teams(spread_analysis)
    with analysis_tabs[2]:
        display_win_probs(moneyline_data)
    with analysis_tabs[3]:
        display_bias(bookmaker_biases)
    with analysis_tabs[4]:
        recommended_bets(moneyline_data)
    with analysis_tabs[5]:
        display_EV_analysis(top_value_bets)
    with analysis_tabs[6]:
        display_parlay_creator(moneyline_data)

if __name__ == "__main__":
    main()