import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from functools import reduce

# Configuration and Setup

# Initialize Streamlit app
st.set_page_config(page_title="Court Vison", layout="wide")
st.title("Stay Ahead and Make some Bread")

# API configuration
API_KEY = "23460d3aaebd465dfc9eebfe77c12c21"
API_URL = "https://api.the-odds-api.com/v4/sports"

# Helper Functions


#Converts Odds to a Probability of Hitting
def odds_to_probability(odds):
    """Convert American odds to implied probability"""
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

#Converts
def american_odds(odds):
    """Convert American odds to decimal odds"""
    return odds / 100 + 1 if odds > 0 else -100 / odds + 1

def fetch_sports_data():
    """Fetch available sports data from API"""
    try:
        response = requests.get(API_URL, params={"apiKey": API_KEY})
        response.raise_for_status()
        sports_list = response.json()
        st.success("API Connected Successfully")
        return [sport for sport in sports_list if sport["title"] in {"NBA", "MLB", "NHL"}]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to Connect: {str(e)}")
        return []

def fetch_odds_data(sport_key):
    """Fetch odds data for selected sport"""
    url = f"{API_URL}/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
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
    if spreads.empty:
        return spreads
    
    spreads["spread_score"] = -spreads["point"]
    spreads["prob"] = spreads["odds"].apply(odds_to_probability)
    spreads["win_score"] = 0.5 * spreads["spread_score"] + 0.5 * spreads["prob"]
    return spreads

def calculate_value_bets(spreads, h2h):
    """Calculate value bets (difference between win score and moneyline probability)"""
    if h2h.empty:
        return pd.DataFrame()
    
    h2h["win_probability"] = h2h["odds"].apply(odds_to_probability)
    value_df = h2h.merge(spreads[["team", "win_score"]], on="team", how="left")
    value_df["value_score"] = value_df["win_score"] - value_df["win_probability"]
    best_value = value_df.loc[value_df.groupby("team")["value_score"].idxmax()].reset_index(drop=True)
    return best_value[best_value["odds"] > -200].sort_values("value_score", ascending=False)

def calculate_book_bias(df):
    """Calculate book bias for totals markets"""
    totals = df[df["market"] == "totals"].copy()
    if totals.empty:
        return pd.Series(dtype=float)
    
    totals["probability"] = totals["odds"].apply(odds_to_probability)
    pivot = totals.pivot_table(index=["matchup", "sportsbook", "point"], 
                             columns="team", values="probability").reset_index()
    pivot["public_bias"] = pivot.get("Over", np.nan) - pivot.get("Under", np.nan)
    bias = pivot.dropna(subset=["public_bias"])
    return bias.groupby("sportsbook")["public_bias"].mean().sort_values()

def calculate_parlay_payout(bets, stake=100):
    """Calculate parlay payout from selected bets"""

    decimal_odds = [american_odds(bet['odds']) for bet in bets]
    combined_odds = reduce(lambda x, y: x * y, decimal_odds)
    payout = stake * combined_odds
    profit = payout - stake
    return combined_odds, payout, profit

def display_favorite_picks(h2h_data):
    """Display categorized picks based on win probability"""
    if h2h_data.empty:
        st.warning("No moneyline data available")
        return
    
    h2h_avg = h2h_data.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
    
    # Initialize lists to store picks
    categories = {
        "Safe Picks (65%+ chance)": [],
        "Risky Calls (48-52% chance)": [],
        "Upset Alerts (40-47.9% chance)": []
    }
    
    for matchup, metrics in h2h_avg.groupby("matchup"):
        if len(metrics) != 2:
            continue
            
        # Normalize probabilities to sum to 100%
        total_prob = metrics["win_probability"].sum()
        metrics["normalized_prob"] = (metrics["win_probability"] / total_prob) * 100
        
        for _, row in metrics.iterrows():
            pick = {
                "Team": row["team"],
                "Matchup": matchup,
                "Win Probability (%)": round(row["normalized_prob"], 2)
            }
            
            prob = row["normalized_prob"]
            if prob > 65:
                categories["Safe Picks (65%+ chance)"].append(pick)
            elif 48 <= prob <= 52:
                categories["Risky Calls (48-52% chance)"].append(pick)
            elif 40 <= prob < 48:
                categories["Upset Alerts (40-47.9% chance)"].append(pick)
    
    # Display each category
    for category_name, picks in categories.items():
        st.markdown(f"### {category_name}")
        if picks:
            df = pd.DataFrame(picks)
            st.dataframe(df.sort_values("Win Probability (%)", ascending=False))
        else:
            st.info(f"No {category_name.lower()} found")



# Application Interface

def display_value_bets_tab(best_value_bets):
    """Display content for Value Bets tab"""
    st.subheader("Top Value Bets")
    if not best_value_bets.empty:
        st.dataframe(
            best_value_bets[["team", "matchup", "odds", "win_probability", "win_score", "value_score"]].head(10),
            height=400
        )
    else:
        st.warning("No value bets found")

def display_win_score_tab(spreads):
    """Display content for Win Score Analysis tab"""
    st.subheader("Win-Score Analysis")
    if not spreads.empty:
        grouped = spreads.groupby(["matchup", "team"]).agg({"win_score": "mean"}).reset_index()
        
        for matchup, metrics in grouped.groupby("matchup"):
            if len(metrics) != 2:
                continue
                
            teams = metrics["team"].values
            scores = metrics["win_score"].values
            winner = teams[np.argmax(scores)]

            fig, ax = plt.subplots()
            colors = ["green" if score == max(scores) else "red" for score in scores]
            bars = ax.bar(teams, scores, color=colors)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha="center")
                
            ax.set_title(f"{matchup}\nPredicted Winner: {winner}")
            ax.set_ylabel("Win Score")
            st.pyplot(fig)
    else:
        st.warning("No spread data available")

def display_win_probabilities_tab(h2h):
    """Display content for Win Probabilities tab"""
    st.subheader("Moneyline Win Probabilities")
    if not h2h.empty:   
        h2h_avg = h2h.groupby(["matchup", "team"])["win_probability"].mean().reset_index()
        
        for matchup, metrics in h2h_avg.groupby("matchup"):
            if len(metrics) != 2:
                continue
                
            teams = metrics["team"].values
            probs = metrics["win_probability"].values
            probs = (probs / probs.sum()) * 100  # Normalize to 100%
            
            fig, ax = plt.subplots()
            bars = ax.bar(teams, probs, color="skyblue")
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.1f}%", ha="center")
                
            ax.set_title(f"{matchup} - Moneyline Probabilities")
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)
    else:
        st.warning("No moneyline data available")

def display_book_bias_tab(book_bias):
    """Display content for Book Bias tab"""
    st.subheader("Sportsbook Public Bias")
    if not book_bias.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        book_bias.plot(kind="barh", color="salmon", edgecolor="black", ax=ax)
        ax.axvline(0, linestyle="--", color="black")
        ax.set_title("Book Bias (Over vs Under)")
        ax.set_xlabel("Bias")
        ax.set_ylabel("Sportsbook")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)
    else:
        st.warning("No totals data available")

def display_parlay_builder_tab(h2h):
    """Display content for Parlay Builder tab"""
    st.subheader("Parlay Builder")
    
    if not h2h.empty:
        # First select the sportsbook
        available_books = h2h["sportsbook"].unique()
        selected_book = st.selectbox(
            "Select Sportsbook to Build Parlay From:", 
            available_books,
            key="parlay_book_select"
        )
        
        # Filter to only show matchups from selected book
        book_h2h = h2h[h2h["sportsbook"] == selected_book].copy()
        
        if book_h2h.empty:
            st.warning(f"No moneyline data available for {selected_book}")
        else:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"### {selected_book} Matchups")
                
                # Group by matchup and show teams/odds
                matchups = book_h2h.groupby("matchup").agg({
                    'team': list,
                    'odds': list
                }).reset_index()
                
                selected_matchup = st.selectbox(
                    "Select Matchup:", 
                    matchups["matchup"].unique(),
                    key="parlay_matchup_select"
                )
                
                # Display teams and odds for selected matchup
                matchup_data = book_h2h[book_h2h["matchup"] == selected_matchup]
                st.dataframe(
                    matchup_data[["team", "odds"]].reset_index(drop=True),
                    hide_index=True
                )
                
                selected_team = st.selectbox(
                    "Select Team:", 
                    matchup_data["team"].unique(),
                    key="parlay_team_select"
                )
                
                selected_odds = matchup_data[matchup_data["team"] == selected_team]["odds"].iloc[0]
                
                if st.button("Add to Parlay"):
                    if any(bet["team"] == selected_team and bet["matchup"] == selected_matchup 
                          for bet in st.session_state.get('parlay_bets', [])):
                        st.warning("This team is already in your parlay for this matchup")
                    else:
                        if 'parlay_bets' not in st.session_state:
                            st.session_state.parlay_bets = []
                            
                        st.session_state.parlay_bets.append({
                            "sportsbook": selected_book,
                            "matchup": selected_matchup,
                            "team": selected_team,
                            "odds": selected_odds
                        })
                        st.rerun()
            
            with col2:
                st.markdown("### Your Parlay")
                
                if not st.session_state.get('parlay_bets'):
                    st.info("No bets added yet")
                else:
                    # Verify all bets are from same sportsbook
                    unique_books = {bet["sportsbook"] for bet in st.session_state.parlay_bets}
                    if len(unique_books) > 1:
                        st.error("Error: Parlay contains bets from multiple sportsbooks")
                    
                    # Display current parlay
                    for i, bet in enumerate(st.session_state.parlay_bets):
                        cols = st.columns([4, 1])
                        cols[0].write(f"{bet['team']} ({bet['odds']}) - {bet['matchup']}")
                        if cols[1].button("Remove", key=f"remove_{i}"):
                            st.session_state.parlay_bets.pop(i)
                            st.rerun()
                    
                    # Parlay calculation
                    stake = st.number_input(
                        "Enter Stake ($):",
                        min_value=1,
                        value=100,
                        step=1,
                        key="parlay_stake"
                    )
                    
                    if st.session_state.parlay_bets:
                        combined_odds, payout, profit = calculate_parlay_payout(
                            st.session_state.parlay_bets,
                            stake
                        )
                        
                        st.markdown("---")
                        st.markdown(f"**Sportsbook:** {selected_book}")
                        st.markdown(f"**Number of Legs:** {len(st.session_state.parlay_bets)}")
                        st.markdown(f"**Combined Odds:** {combined_odds:.2f}")
                        st.metric("Potential Payout", f"${payout:.2f}")
                        st.metric("Potential Profit", f"${profit:.2f}")
                    
                    if st.button("Clear Parlay"):
                        st.session_state.parlay_bets = []
                        st.rerun()
    else:
        st.warning("No moneyline data available for parlay building")


# Application Execution

def main():
    """Main application execution flow"""
    # Initialize session state for parlay builder
    if 'parlay_bets' not in st.session_state:
        st.session_state.parlay_bets = []

    # Fetch sports data
    sports_list = fetch_sports_data()
    if not sports_list:
        return

    # Sport selection
    sport_names = [sport['title'] for sport in sports_list]
    selected_sport = st.selectbox("Select a sport:", sport_names)
    sport_key = next(sport['key'] for sport in sports_list if sport['title'] == selected_sport)

    # Fetch and process data
    data = fetch_odds_data(sport_key)
    if not data:
        return
        
    final_df = clean_odds_data(data)
    spreads = calculate_win_scores(final_df)
    h2h = final_df[final_df['market'] == 'h2h'].copy()
    best_value_bets = calculate_value_bets(spreads, h2h)
    book_bias = calculate_book_bias(final_df)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Value Bets", 
        "Win Score Analysis", 
        "Win Probabilities", 
        "Book Bias",
        "Favorite Picks", 
        "Parlay Builder"
    ])

    # Display each tab's content
    with tab1:
        display_value_bets_tab(best_value_bets)

    with tab2:
        display_win_score_tab(spreads)

    with tab3:
        display_win_probabilities_tab(h2h)

    with tab4:
        display_book_bias_tab(book_bias)

    with tab5:
        st.subheader("Safe Picks and Upset Alerts")
        display_favorite_picks(h2h)

    with tab6:
        display_parlay_builder_tab(h2h)

if __name__ == "__main__":
    main()