import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

# --- CONFIGURATION & DISCLAIMER ---
APP_TITLE = "SBW Predictive Analytics (Super App)"
DATA_FILE = "historical_match_data.csv"
API_PLACEHOLDER_KEY = "key" 

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"⚽ {APP_TITLE}")
st.markdown("""
> ⚠️ **DISCLAIMER:** This tool provides mathematical probabilities based on historical data (Poisson Distribution). 
> There is **no guarantee** of accuracy. Please use responsibly.
""")

# ==========================================
# 1. CORE ENGINE & DATA ACQUISITION
# ==========================================

class SBWPredictor:
    def __init__(self):
        self.df = pd.DataFrame()
        # Initialize dictionaries for team strengths
        self.strengths = {}

    def download_data_to_csv(self, api_key, league_id="39", season="2024"):
        """
        Simulates fetching data from a real football API (like API-Football).
        NOTE: This requires a valid, active API key and specific league/season IDs.
        """
        if api_key == API_PLACEHOLDER_KEY:
            st.error("key")
            return

        st.info(f"Attempting to download Premier League ({season}) data...")
        url = "https://v3.football.api-football.com/fixtures"
        headers = {
            'x-rapidapi-host': "v3.football.api-football.com",
            'x-rapidapi-key': api_key
        }
        params = {"league": league_id, "season": season, "status": "FT"} 

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status() # Check for HTTP errors
            data = response.json()
            
            matches = []
            for fixture in data['response']:
                if fixture['fixture']['status']['short'] == 'FT': # Full Time only
                    matches.append({
                        'home_team': fixture['teams']['home']['name'],
                        'away_team': fixture['teams']['away']['name'],
                        'home_goals': fixture['goals']['home'],
                        'away_goals': fixture['goals']['away'],
                    })

            self.df = pd.DataFrame(matches)
            if self.df.empty:
                st.warning("Download successful, but no finished matches found for the criteria.")
                return

            self.df.to_csv(DATA_FILE, index=False)
            st.success(f"✅ Data downloaded and saved to **{DATA_FILE}** ({len(self.df)} matches).")
            self.train_model()
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Request Failed: {e}. Check your API Key or subscription.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")


    def load_data(self):
        """Loads local CSV data or provides dummy data."""
        try:
            self.df = pd.read_csv(DATA_FILE)
            st.sidebar.success(f"Loaded **{len(self.df)}** matches from **{DATA_FILE}**.")
            self.train_model()
        except FileNotFoundError:
            st.sidebar.error("Local data file not found. Loading dummy data.")
            self.df = self._create_dummy_data()
            self.train_model()

    def _create_dummy_data(self):
        """Synthetic data for initial testing."""
        data = {
            'home_team': ['Arsenal', 'Man City', 'Liverpool', 'Chelsea', 'Arsenal', 'Man City', 'Man Utd', 'Liverpool'],
            'away_team': ['Man Utd', 'Chelsea', 'Arsenal', 'Man Utd', 'Liverpool', 'Spurs', 'Man Utd', 'Chelsea'],
            'home_goals': [3, 2, 1, 0, 2, 3, 1, 2],
            'away_goals': [1, 1, 1, 2, 2, 1, 2, 1]
        }
        return pd.DataFrame(data)

    def train_model(self):
        """Calculates Attack, Defense, and H2H Strengths."""
        if self.df.empty: return

        # 1. Global Averages
        avg_home_goals = self.df['home_goals'].mean()
        avg_away_goals = self.df['away_goals'].mean()

        teams = set(self.df['home_team'].unique()) | set(self.df['away_team'].unique())

        for team in teams:
            df_home = self.df[self.df['home_team'] == team]
            df_away = self.df[self.df['away_team'] == team]
            
            # --- Attack/Defense Strength (Poisson Model Foundation) ---
            home_att = df_home['home_goals'].mean() / avg_home_goals if avg_home_goals > 0 else 1.0
            home_def = df_home['away_goals'].mean() / avg_away_goals if avg_away_goals > 0 else 1.0
            away_att = df_away['away_goals'].mean() / avg_away_goals if avg_away_goals > 0 else 1.0
            away_def = df_away['home_goals'].mean() / avg_home_goals if avg_home_goals > 0 else 1.0
            
            # Combine home and away performance for a single rating
            attack_strength = (home_att + away_att) / 2
            defense_strength = (home_def + away_def) / 2
            
            self.strengths[team] = {
                'attack': attack_strength,
                'defense': defense_strength,
                'home_goals': avg_home_goals, # Store for prediction
                'away_goals': avg_away_goals  # Store for prediction
            }

    def get_h2h_modifier(self, home_team, away_team, num_games=5):
        """Calculates a head-to-head modifier based on recent meetings."""
        h2h_df = self.df[
            ((self.df['home_team'] == home_team) & (self.df['away_team'] == away_team)) |
            ((self.df['home_team'] == away_team) & (self.df['away_team'] == home_team))
        ].tail(num_games)
        
        if h2h_df.empty:
            return 1.0, 1.0 # No modifier

        # Calculate goals scored by each team in H2H matches
        home_goals_h2h = h2h_df.apply(
            lambda row: row['home_goals'] if row['home_team'] == home_team else row['away_goals'], axis=1
        ).mean()
        away_goals_h2h = h2h_df.apply(
            lambda row: row['away_goals'] if row['away_team'] == away_team else row['home_goals'], axis=1
        ).mean()

        # Compare H2H average goals to overall team average goals
        # A modifier > 1.0 means they score more than expected vs this opponent
        try:
            home_mod = home_goals_h2h / (self.strengths[home_team]['attack'] * self.strengths[home_team]['home_goals'])
            away_mod = away_goals_h2h / (self.strengths[away_team]['attack'] * self.strengths[away_team]['away_goals'])
        except (KeyError, ZeroDivisionError):
            return 1.0, 1.0 # Fallback

        # Cap modifier to prevent extreme results from low sample size H2H
        return np.clip(home_mod, 0.5, 1.5), np.clip(away_mod, 0.5, 1.5)


    def predict_match(self, home_team, away_team):
        """Predicts score probabilities using enhanced Poisson model."""
        if home_team not in self.strengths or away_team not in self.strengths:
            return None, "One or both teams not found in data."

        s_home = self.strengths[home_team]
        s_away = self.strengths[away_team]

        # Get H2H Modifier
        h2h_mod_home, h2h_mod_away = self.get_h2h_modifier(home_team, away_team)
        
        # Calculate Expected Goals (Lambda) - applying H2H modifier
        home_xg = s_home['attack'] * s_away['defense'] * s_home['home_goals'] * h2h_mod_home
        away_xg = s_away['attack'] * s_home['defense'] * s_away['away_goals'] * h2h_mod_away

        # Calculate Probabilities for scores 0-0 up to 5-5
        score_probs = []
        for h in range(6): 
            for a in range(6):
                # Poisson Probability Mass Function (PMF)
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                score_probs.append({'score': f"{h}-{a}", 'probability': prob})

        predictions = sorted(score_probs, key=lambda x: x['probability'], reverse=True)
        
        # Calculate Match Outcomes (Win/Draw/Loss probability)
        home_win_prob = sum(p['probability'] for p in predictions if int(p['score'].split('-')[0]) > int(p['score'].split('-')[1]))
        draw_prob = sum(p['probability'] for p in predictions if int(p['score'].split('-')[0]) == int(p['score'].split('-')[1]))
        away_win_prob = sum(p['probability'] for p in predictions if int(p['score'].split('-')[0]) < int(p['score'].split('-')[1]))
        
        outcomes = {
            'home_win': home_win_prob, 
            'draw': draw_prob, 
            'away_win': away_win_prob
        }
        
        return predictions, (home_xg, away_xg), outcomes

# ==========================================
# 2. FRONTEND UI (STREAMLIT)
# ==========================================

# Initialize Engine
if 'engine' not in st.session_state:
    st.session_state.engine = SBWPredictor()

st.sidebar.header("🛠️ 1. Data Management")
api_key_input = st.sidebar.text_input("key", API_PLACEHOLDER_KEY, type="password")

if st.sidebar.button("⬇️ Download Current Data (API)"):
    st.session_state.engine.download_data_to_csv(api_key_input)
    st.session_state.teams = sorted(list(set(st.session_state.engine.df['home_team'].unique()) | set(st.session_state.engine.df['away_team'].unique())))
    st.experimental_rerun()

if st.sidebar.button("📂 Load Local Data"):
    st.session_state.engine.load_data()
    st.session_state.teams = sorted(list(set(st.session_state.engine.df['home_team'].unique()) | set(st.session_state.engine.df['away_team'].unique())))
    st.experimental_rerun()


# --- Main Area ---
st.header("2. Match Analysis")

if st.session_state.engine.df.empty:
    st.warning("⚠️ Please load or download data from the sidebar to start analysis.")
else:
    teams = st.session_state.engine.df['home_team'].unique().tolist()
    
    col_home, col_vs, col_away = st.columns([2, 1, 2])
    with col_home:
        home_team = st.selectbox("Home Team", teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 0)
    with col_away:
        away_team = st.selectbox("Away Team", teams, index=teams.index('Chelsea') if 'Chelsea' in teams else (1 if len(teams)>1 else 0))

    if st.button(f"🔍 Analyze: {home_team} vs {away_team}"):
        if home_team == away_team:
            st.error("Teams must be different.")
        else:
            with st.spinner('Calculating advanced probabilities...'):
                predictions, (h_xg, a_xg), outcomes = st.session_state.engine.predict_match(home_team, away_team)

                st.subheader(f"Model Results: {home_team} vs {away_team}")
                
                # --- XG and Outcome Probability ---
                col_xgh, col_draw, col_xga = st.columns([1, 1, 1])
                col_xgh.metric(f"{home_team} Expected Goals (xG)", f"{h_xg:.2f}", delta=f"{outcomes['home_win']*100:.1f}% Win Prob")
                col_draw.metric("Draw Expected Goals", f"{h_xg:.2f}", delta=f"{outcomes['draw']*100:.1f}% Draw Prob")
                col_xga.metric(f"{away_team} Expected Goals (xG)", f"{a_xg:.2f}", delta=f"{outcomes['away_win']*100:.1f}% Win Prob")

                st.divider()

                # --- Top 5 Correct Scores ---
                st.subheader("🎯 Top 5 Correct Score Predictions")
                
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        score = predictions[i]['score']
                        prob = predictions[i]['probability'] * 100
                        st.markdown(f"**{score}**")
                        st.progress(prob / predictions[0]['probability'] * 100) # Progress bar relative to top score
                        st.caption(f"{prob:.2f}%")
                
                st.divider()

                # --- Super App Feature: Value Bet Check ---
                st.subheader("💰 3. Super App Feature: Value Bet Check (SBW Permission)")
                st.markdown("""
                *To proceed, SBW requires your 'permission/desire' to proceed based on potential value. 
                Enter the bookmaker odds to check if this is a **Value Bet** (Positive EV).*
                """)
                
                # Assume the highest probability score is the user's focus
                top_score_prob = predictions[0]['probability']
                top_score = predictions[0]['score']

                col_prob, col_odds, col_ev = st.columns(3)
                
                col_prob.metric("Model Probability for " + top_score, f"{top_score_prob*100:.2f}%")
                
                with col_odds:
                    bookie_odds = st.number_input(f"Bookmaker Odds for {top_score}", min_value=1.01, value=3.50, step=0.01)

                # Value Bet Calculation: (Odds * Model Prob) - 1 > 0
                implied_prob = 1 / bookie_odds
                value_index = (bookie_odds * top_score_prob) - 1
                
                if value_index > 0:
                    col_ev.metric("Value Check (EV)", f"✅ VALUE BET", delta=f"{value_index*100:.2f}% Edge", delta_color="green")
                    st.success(f"**Permission Granted!** The bookmaker odds ({bookie_odds:.2f}) are higher than the odds implied by the model ({1/top_score_prob:.2f}). This is a **Positive Expected Value (+EV)** bet.")
                elif value_index == 0:
                    col_ev.metric("Value Check (EV)", "NEUTRAL", delta="0.00% Edge")
                    st.info("The model and bookmaker agree. No edge found.")
                else:
                    col_ev.metric("Value Check (EV)", f"❌ NO VALUE", delta=f"{value_index*100:.2f}% Edge", delta_color="red")
                    st.error("The bookmaker odds are lower than the model's implied odds. This is a **Negative Expected Value (-EV)** bet.")

                
                with st.expander("Full Probability Matrix"):
                    st.dataframe(pd.DataFrame(predictions))          