import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from functools import reduce

# Initialize Streamlit app
st.set_page_config(page_title="Sports Odds & Value Model", layout="wide")
st.title("Stay Ahead and Make Some Bread")

# API configuration
api_key = "827e4300b8d71acf49cac34c9a405670"
api_url = "https://api.the-odds-api.com/v4/sports"

# Helper functions
def odds_to_probability(odds):
    """Convert American odds to implied probability"""
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

def american_to_decimal(odds):
    """Convert American odds to decimal odds"""
    return odds / 100 + 1 if odds > 0 else -100 / odds + 1

def fetch_sports_data():
    """Fetch available sports data from API"""
    try:
        response = requests.get(api_url, params={"apiKey": api_key})
        response.raise_for_status()
        sports_list = response.json()
        st.success("API Connected Successfully")
        return [sport for sport in sports_list if sport["title"] in {"NBA", "MLB", "NHL"}]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to Connect: {str(e)}")
        return []

def fetch_odds_data(sport_key):
    """Fetch odds data for selected sport"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,h2h,totals",
        "oddsFormat": "american"
    }
    response = requests.get(url, params=params)
    return response.json()

def clean_odds_data(data):
    """Clean and structure raw odds data"""
    cleaned_data = []
    for match in data:
        game_time = datetime.fromisoformat(match["commence_time"].replace("Z", "+00:00"))
        home = match["home_team"]
        away = match["away_team"]
        for site in match.get("bookmakers", []):
            book = site["title"]
            for market in site.get("markets", []):
                for outcome in market["outcomes"]:
                    entry = {
                        "time": game_time,
                        "home_team": home,
                        "away_team": away,
                        "matchup": f"{away} @ {home}",
                        "market": market["key"],
                        "sportsbook": book,
                        "team": outcome["name"],
                        "odds": outcome["price"]
                    }
                    if "point" in outcome:
                        entry["point"] = outcome["point"]
                    cleaned_data.append(entry)
    return pd.DataFrame(cleaned_data)

def calculate_win_scores(df):
    """Calculate win scores (combination of spread and odds probability)"""
    spreads = df[df["market"] == "spreads"].copy()
    spreads["spread_score"] = -spreads["point"]
    spreads["prob"] = spreads["odds"].apply(odds_to_probability)
    spreads["win_score"] = 0.5 * spreads["spread_score"] + 0.5 * spreads["prob"]
    return spreads

def calculate_value_bets(spreads, h2h):
    """Calculate value bets (difference between win score and moneyline probability)"""
    h2h["win_probability"] = h2h["odds"].apply(odds_to_probability)
    value_df = h2h.merge(spreads[["team", "win_score"]], on="team", how="left")
    value_df["value_score"] = value_df["win_score"] - value_df["win_probability"]
    best_value = value_df.loc[value_df.groupby("team")["value_score"].idxmax()].reset_index(drop=True)
    best_value = best_value[best_value["odds"] > -200]
    return best_value.sort_values(by="value_score", ascending=False)

def calculate_book_bias(df):
    """Calculate book bias for totals markets"""
    totals = df[df["market"] == "totals"].copy()
    totals["probability"] = totals["odds"].apply(odds_to_probability)
    pivot = totals.pivot_table(index=["matchup", "sportsbook", "point"], 
                             columns="team", values="probability").reset_index()
    pivot["public_bias"] = pivot.get("Over", np.nan) - pivot.get("Under", np.nan)
    bias = pivot.dropna(subset=["public_bias"])
    return bias.groupby("sportsbook")["public_bias"].mean().sort_values()

def calculate_parlay_payout(bets, stake=100):
    """Calculate parlay payout from selected bets"""
    if not bets:
        return 0, 0, 0
    decimal_odds = [american_to_decimal(bet['odds']) for bet in bets]
    combined_odds = reduce(lambda x, y: x * y, decimal_odds)
    payout = stake * combined_odds
    profit = payout - stake
    return combined_odds, payout, profit

# Main App Logic
sports_list = fetch_sports_data()

if sports_list:
    # Sport selection
    sport_names = [sport['title'] for sport in sports_list]
    selected_sport = st.selectbox("Select a sport:", sport_names)
    sport_key = [sport['key'] for sport in sports_list if sport['title'] == selected_sport][0]

    # Fetch and process data
    data = fetch_odds_data(sport_key)
    final_df = clean_odds_data(data)
    spreads = calculate_win_scores(final_df)
    h2h = final_df[final_df['market'] == 'h2h'].copy()
    best_value_bets = calculate_value_bets(spreads, h2h)
    book_bias = calculate_book_bias(final_df)

    # Initialize session state for parlay builder
    if 'parlay_bets' not in st.session_state:
        st.session_state.parlay_bets = []

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Value Bets", 
        "Win Score Analysis", 
        "Win Probabilities", 
        "Book Bias",
        "Favorite Picks", 
        "Parlay Builder"
    ])

    # Tab 1: Value Bets
    with tab1:
        st.subheader("Top Value Bets")
        st.dataframe(best_value_bets[["team", "matchup", "odds", "win_probability", "win_score", "value_score"]].head(10))

    # Tab 2: Win Score Analysis
    with tab2:
        st.subheader("Win-Score Analysis")
        grouped = spreads.groupby(["matchup", "team"]).agg({"win_score": "mean"}).reset_index()
        
        for matchup, metrics in grouped.groupby("matchup"):
            if len(metrics) != 2:
                continue
                
            teams = metrics["team"].values
            scores = metrics["win_score"].values
            winner = teams[np.argmax(scores)]

            fig, ax = plt.subplots()
            bars = ax.bar(teams, scores, color=["red" if score != max(scores) else "green" for score in scores])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha="center")
                
            ax.set_title(f"{matchup}\nPredicted Winner: {winner}")
            ax.set_ylabel("Win Score")
            st.pyplot(fig)

    # Tab 3: Win Probabilities
    with tab3:
        st.subheader("Moneyline Win Probabilities")
        if not h2h.empty:   
            h2h_avg = h2h.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
            
            for matchup, metrics in h2h_avg.groupby("matchup"):
                if len(metrics) != 2:
                    continue
                    
                teams = metrics["team"].values
                probs = metrics["win_probability"].values
                probs = probs / probs.sum()  # Normalize to 100%
                
                fig, ax = plt.subplots()
                bars = ax.bar(teams, probs, color="skyblue")
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height*100:.1f}%", ha="center")
                    
                ax.set_title(f"{matchup} - Moneyline Probabilities")
                ax.set_ylabel("Probability")
                st.pyplot(fig)

    # Tab 4: Book Bias
    with tab4:
        st.subheader("Sportsbook Public Bias")
        fig, ax = plt.subplots(figsize=(8, 5))
        book_bias.plot(kind="barh", color="salmon", edgecolor="black", ax=ax)
        ax.axvline(0, linestyle="--", color="black")
        ax.set_title("Book Bias (Over vs Under)")
        ax.set_xlabel("Bias")
        ax.set_ylabel("Sportsbook")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)

    # Tab 5: Favorite Picks
    with tab5:
        st.subheader("Safe Picks and Upset Alerts")
        
        if not h2h.empty:
            h2h_avg = h2h.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
            
            safe_picks = []
            risky_calls = []
            upset_alerts = []
            
            for matchup, metrics in h2h_avg.groupby("matchup"):
                if len(metrics) != 2:
                    continue
                    
                teams = metrics["team"].values
                probs = metrics["win_probability"].values
                probs = probs / probs.sum()  # Normalize to 100%
                
                for i in range(2):
                    team = teams[i]
                    prob = probs[i] * 100
                    
                    if prob > 65:
                        safe_picks.append({"Team": team, "Matchup": matchup, "Win Probability (%)": round(prob, 2)})
                    elif 48 <= prob <= 52:
                        risky_calls.append({"Team": team, "Matchup": matchup, "Win Probability (%)": round(prob, 2)})
                    elif 40 <= prob < 48:
                        upset_alerts.append({"Team": team, "Matchup": matchup, "Win Probability (%)": round(prob, 2)})
            
            st.markdown("### Safe Picks (65%+ chance to win)")
            st.dataframe(pd.DataFrame(safe_picks).sort_values(by="Win Probability (%)", ascending=False))
            
            st.markdown("### Risky Calls (48-52% win probability)")
            st.dataframe(pd.DataFrame(risky_calls).sort_values(by="Win Probability (%)", ascending=False))
            
            st.markdown("### Upset Picks (40-47.9% win probability)")
            st.dataframe(pd.DataFrame(upset_alerts).sort_values(by="Win Probability (%)", ascending=False))

    # Tab 6: Parlay Builder
    with tab6:
        st.subheader("Parlay Builder")
        
        if not h2h.empty:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Available Bets")
                unique_matchups = h2h["matchup"].unique()
                selected_matchup = st.selectbox("Select a matchup:", unique_matchups, key="matchup_select")
                
                filtered = h2h[h2h["matchup"] == selected_matchup].copy()
                avg_odds = (
                    filtered.groupby(["sportsbook", "team"])
                    .agg({"odds": "mean"})
                    .reset_index()
                    .pivot(index="sportsbook", columns="team", values="odds")
                )
                
                st.dataframe(avg_odds.style.format("{:.0f}"))
                
                st.markdown("### Add to Parlay")
                selected_book = st.selectbox("Sportsbook:", filtered["sportsbook"].unique())
                selected_team = st.selectbox("Team:", filtered[filtered["sportsbook"] == selected_book]["team"].unique())
                
                selected_odds = filtered[(filtered["sportsbook"] == selected_book) & 
                                        (filtered["team"] == selected_team)]["odds"].values[0]
                
                if st.button("Add Bet to Parlay"):
                    st.session_state.parlay_bets.append({
                        "matchup": selected_matchup,
                        "sportsbook": selected_book,
                        "team": selected_team,
                        "odds": selected_odds
                    })
                    st.success(f"Added {selected_team} ({selected_odds}) to parlay")
            
            with col2:
                st.markdown("### Current Parlay")
                if not st.session_state.parlay_bets:
                    st.info("No bets in parlay yet")
                else:
                    for i, bet in enumerate(st.session_state.parlay_bets):
                        cols = st.columns([3, 1])
                        cols[0].write(f"{bet['team']} ({bet['odds']})")
                        if cols[1].button("Remove", key=f"remove_{i}"):
                            st.session_state.parlay_bets.pop(i)
                            st.rerun()
                    
                    stake = st.number_input("Stake Amount ($)", min_value=1, value=100, step=10)
                    
                    if st.session_state.parlay_bets:
                        combined_odds, payout, profit = calculate_parlay_payout(st.session_state.parlay_bets, stake)
                        
                        st.markdown("---")
                        st.markdown(f"**Number of Bets:** {len(st.session_state.parlay_bets)}")
                        st.markdown(f"**Combined Odds:** {combined_odds:.2f}")
                        st.markdown(f"**Stake:** ${stake:.2f}")
                        st.markdown(f"**Potential Payout:** ${payout:.2f}")
                        st.markdown(f"**Potential Profit:** ${profit:.2f}")
                    
                    if st.button("Clear Parlay"):
                        st.session_state.parlay_bets = []
                        st.rerun()
                    
                    else:
                        st.warning("No moneyline data available for parlay building")