from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os

app = Flask(__name__)

# Global variables to hold data
df_players = None
df_teams = None
df_match = None
df_events = None

PKL_TEAMS = [
    'Jaipur Pink Panthers', 'U Mumba', 'Dabang Delhi K.C.', 'Dabang Delhi',
    'Bengal Warriors', 'Patna Pirates', 'Puneri Paltan', 'Telugu Titans',
    'Bengaluru Bulls', 'Haryana Steelers', 'Gujarat Fortunegiants',
    'U.P. Yoddha', 'Tamil Thalaivas'
]

TACTICAL_INTERPRETATION = {
    "Primary Raider": {
        "style": "Aggressive Spearhead",
        "strengths": ["High point scoring", "Do-or-Die reliability", "Pressure handling"],
        "weaknesses": ["Prone to advanced tackles", "High time off-court if targeted"],
        "recommendation": "Ensure they are on the mat during crucial Do-or-Die situations. Build defensive setups to revive them quickly."
    },
    "High-Risk Raider": {
        "style": "Disruptive / Unorthodox",
        "strengths": ["Unpredictable", "Forces defensive errors", "High pace"],
        "weaknesses": ["Low success rate", "Often concedes easy tackle points"],
        "recommendation": "Use sparingly or for shock tactics when trailing significantly. Do not use to defend narrow leads."
    },
    "Support Raider": {
        "style": "Safe / Tactical",
        "strengths": ["Low error rate", "Bonus point accumulation", "Clock management"],
        "weaknesses": ["Struggles against heavy defensive formations", "Low multi-point raids"],
        "recommendation": "Deploy to keep the scoreboard ticking and to revive primary raiders safely."
    },
    "Defensive Specialist": {
        "style": "Anchor Defender",
        "strengths": ["High tackle success", "Super Tackle capability", "Positional discipline"],
        "weaknesses": ["Zero attacking threat", "Can be targeted by fast raiders"],
        "recommendation": "Anchor the defensive chain around them. Crucial for situations with 3 or fewer defenders on court."
    },
    "Support Defender": {
        "style": "Assisting Defender",
        "strengths": ["Chain tackles", "Dashes and blocks", "Support speed"],
        "weaknesses": ["Poor solo tackle success", "Vulnerable to advanced raids"],
        "recommendation": "Pair closely with Defensive Specialists. Use them to execute chain tackles and blocks."
    },
    "All-Round Impact Player": {
        "style": "Versatile Core",
        "strengths": ["Contributes in all phases", "High total impact", "Never a liability"],
        "weaknesses": ["Fatigue from high workload", "May lack elite specialization in one specific skill"],
        "recommendation": "Your most valuable asset. Keep them on the mat as much as possible to balance the squad."
    },
    "Consistent Performer": {
        "style": "Reliable Core",
        "strengths": ["Very few unforced errors", "Steady contribution", "High stamina"],
        "weaknesses": ["Low explosive impact", "Rarely turns matches around single-handedly"],
        "recommendation": "Trust them to execute the game plan without making risky errors."
    },
    "Developing Player": {
        "style": "Unknown / Rookie",
        "strengths": ["Unpredictable to opponents", "High energy"],
        "weaknesses": ["Lack of data", "Inexperienced"],
        "recommendation": "Give them game time during comfortable leads to build confidence and data profiles."
    }
}

def classify_player(row):
    raids = row.get('player_raids_total', 0)
    tackles = row.get('player_tackles_total', 0)
    raid_pts = row.get('player_raid_points_total', 0)
    tackle_pts = row.get('player_tackle_points_total', 0)
    raid_sr = row.get('raid_success_rate', 0)
    tackle_sr = row.get('tackle_success_rate', 0)
    
    total_actions = raids + tackles
    if total_actions < 10:
        return "Developing Player"
        
    if raids > tackles * 2:
        if raid_sr > 50:
            return "Primary Raider"
        elif raid_sr < 30:
            return "High-Risk Raider"
        else:
            return "Support Raider"
    elif tackles > raids * 2:
        if tackle_sr > 50:
            return "Defensive Specialist"
        else:
            return "Support Defender"
    else:
        if raid_pts > 20 and tackle_pts > 15:
            return "All-Round Impact Player"
        else:
            return "Consistent Performer"

def load_data():
    global df_players, df_teams, df_match, df_events
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        df_p = pd.read_csv(os.path.join(data_dir, "DS_players.csv"), low_memory=False)
        df_t = pd.read_csv(os.path.join(data_dir, "DS_team.csv"), low_memory=False)
        df_m = pd.read_csv(os.path.join(data_dir, "DS_match.csv"), low_memory=False)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Clean columns
    for df in [df_p, df_t, df_m]:
        df.columns = [str(c).strip().lower() for c in df.columns]

    # Filter strictly for PKL Teams
    # df_t 'name' column contains team names
    if 'name' in df_t.columns:
        df_t = df_t[df_t['name'].isin(PKL_TEAMS)].copy()
        
    team_map = {}
    if 'id' in df_t.columns and 'name' in df_t.columns:
        team_map = dict(zip(df_t['id'], df_t['name']))
        valid_team_ids = set(df_t['id'])
        # Filter match dataset based on PKL teams (using team_id if available, otherwise just use team_map indirectly)
        if 'home_team_id' in df_m.columns:
            df_m = df_m[df_m['home_team_id'].isin(valid_team_ids)].copy()
    
    player_cols = df_p.columns
    agg_cols = [c for c in player_cols if pd.api.types.is_numeric_dtype(df_p[c]) and c not in ['index', 'match_id', 'team_id', 'player_id', 'player_jersey']]
    
    df_players_agg = df_p.groupby(['player_id', 'player_name']).agg(
        {col: 'sum' for col in agg_cols}
    ).reset_index()
    
    player_team = df_p.groupby('player_id')['team_id'].agg(lambda x: x.value_counts().index[0] if not x.empty else None).reset_index()
    player_team['team_name'] = player_team['team_id'].map(team_map).fillna("Unknown Team")
    
    df_players_agg = pd.merge(df_players_agg, player_team, on='player_id', how='left')
    
    # Filter players to only those in PKL teams
    df_players_agg = df_players_agg[df_players_agg['team_name'] != "Unknown Team"].copy()
    
    if 'player_raids_total' in df_players_agg.columns:
        valid_raids = df_players_agg['player_raids_successful'] + df_players_agg['player_raids_unsuccessful']
        df_players_agg['raid_success_rate'] = (df_players_agg['player_raids_successful'] / valid_raids.replace(0, 1)) * 100
        
        valid_tackles = df_players_agg['player_tackles_successful'] + df_players_agg['player_tackles_unsuccessful']
        df_players_agg['tackle_success_rate'] = (df_players_agg['player_tackles_successful'] / valid_tackles.replace(0, 1)) * 100
    
    df_players_agg['tactical_class'] = df_players_agg.apply(classify_player, axis=1)

    df_teams_agg = df_players_agg.groupby(['team_id', 'team_name']).agg(
        total_raids=pd.NamedAgg(column='player_raids_total', aggfunc='sum'),
        successful_raids=pd.NamedAgg(column='player_raids_successful', aggfunc='sum'),
        unsuccessful_raids=pd.NamedAgg(column='player_raids_unsuccessful', aggfunc='sum'),
        total_raid_points=pd.NamedAgg(column='player_raid_points_total', aggfunc='sum'),
        total_tackles=pd.NamedAgg(column='player_tackles_total', aggfunc='sum'),
        successful_tackles=pd.NamedAgg(column='player_tackles_successful', aggfunc='sum'),
        unsuccessful_tackles=pd.NamedAgg(column='player_tackles_unsuccessful', aggfunc='sum'),
        total_tackle_points=pd.NamedAgg(column='player_tackle_points_total', aggfunc='sum')
    ).reset_index()
    
    valid_team_raids = df_teams_agg['successful_raids'] + df_teams_agg['unsuccessful_raids']
    df_teams_agg['raid_success_rate'] = (df_teams_agg['successful_raids'] / valid_team_raids.replace(0, 1)) * 100
    
    valid_team_tackles = df_teams_agg['successful_tackles'] + df_teams_agg['unsuccessful_tackles']
    df_teams_agg['tackle_success_rate'] = (df_teams_agg['successful_tackles'] / valid_team_tackles.replace(0, 1)) * 100

    df_players = df_players_agg
    df_teams = df_teams_agg
    df_match = df_m

# Call it once at startup
load_data()


@app.route('/')
def index():
    total_players = df_players['player_id'].nunique() if df_players is not None else 0
    total_teams = df_teams['team_id'].nunique() if df_teams is not None else 0
    total_matches = df_match['match_id'].nunique() if df_match is not None and 'match_id' in df_match.columns else len(df_match)
    highest_raid_pts = int(df_players['player_raid_points_total'].max()) if df_players is not None else 0
    
    best_raider = df_players.loc[df_players['player_raid_points_total'].idxmax()].to_dict() if df_players is not None else {}
    best_defender = df_players.loc[df_players['player_tackle_points_total'].idxmax()].to_dict() if df_players is not None else {}
    top_team = df_teams.loc[df_teams['total_raid_points'].idxmax()].to_dict() if df_teams is not None else {}
    
    # Most consistent (lowest variance or high total actions with good rates, simplifying to All-round impact player)
    consistent_players = df_players[df_players['tactical_class'] == "All-Round Impact Player"] if df_players is not None else []
    consistent_player = consistent_players.iloc[0].to_dict() if len(consistent_players) > 0 else {}
    
    return render_template('index.html', 
        kpis={'players': total_players, 'teams': total_teams, 'matches': total_matches, 'highest_raid_pts': highest_raid_pts},
        insights={'raider': best_raider, 'defender': best_defender, 'team': top_team, 'consistent': consistent_player}
    )

@app.route('/players')
def players():
    if df_players is None:
        return render_template('players.html', featured={}, top_raiders=[], top_defenders=[])
    
    # Ensure total points exist for sorting
    if 'player_total_points' not in df_players.columns:
        df_players['player_total_points'] = df_players.get('player_raid_points_total', 0) + df_players.get('player_tackle_points_total', 0)

    # Provide data for the page (Popular Players / Discovery)
    featured = df_players[df_players['tactical_class'] == "All-Round Impact Player"].sort_values(by='player_total_points', ascending=False).head(1)
    featured = featured.iloc[0].to_dict() if not featured.empty else df_players.iloc[0].to_dict()
    
    popular_players = df_players.sort_values(by='player_total_points', ascending=False).head(8).to_dict('records')
    
    return render_template('players.html', featured=featured, popular_players=popular_players)

@app.route('/api/players/search')
def api_players_search():
    query = request.args.get('q', '').lower()
    if not query or df_players is None:
        return jsonify([])
    matches = df_players[df_players['player_name'].str.lower().str.contains(query, na=False)]
    return jsonify(matches['player_name'].unique().tolist()[:10])

@app.route('/player/<path:player_name>')
def player_profile(player_name):
    if df_players is None:
        return "Data not loaded", 500
        
    p_data = df_players[df_players['player_name'] == player_name]
    if p_data.empty:
        return "Player not found", 404
        
    p_data = p_data.iloc[0].to_dict()
    
    # Calculate Ranks and League Averages
    active_raiders = df_players[df_players['player_raids_total'] > 0].copy()
    active_defenders = df_players[df_players['player_tackles_total'] > 0].copy()
    
    active_raiders['raid_rank'] = active_raiders['player_raid_points_total'].rank(ascending=False, method='min')
    active_defenders['tackle_rank'] = active_defenders['player_tackle_points_total'].rank(ascending=False, method='min')
    
    player_raid_rank = active_raiders.loc[active_raiders['player_id'] == p_data['player_id'], 'raid_rank'].values
    p_data['raid_rank'] = int(player_raid_rank[0]) if len(player_raid_rank) > 0 else "N/A"
    
    player_tackle_rank = active_defenders.loc[active_defenders['player_id'] == p_data['player_id'], 'tackle_rank'].values
    p_data['tackle_rank'] = int(player_tackle_rank[0]) if len(player_tackle_rank) > 0 else "N/A"
    
    p_data['league_avg_raid_pts'] = int(active_raiders['player_raid_points_total'].mean())
    p_data['league_avg_tackle_pts'] = int(active_defenders['player_tackle_points_total'].mean())
    
    # Calculate Contribution
    total_pts = p_data.get('player_raid_points_total', 0) + p_data.get('player_tackle_points_total', 0)
    p_data['attack_contrib'] = (p_data.get('player_raid_points_total', 0) / total_pts * 100) if total_pts > 0 else 0
    p_data['defense_contrib'] = (p_data.get('player_tackle_points_total', 0) / total_pts * 100) if total_pts > 0 else 0

    # Create charts
    raid_json = None
    tackle_json = None
    if p_data.get('player_raids_total', 0) > 0:
        # Fix for Plotly JS 50-50 bug: natively construct the JSON config to avoid binary encoding of numpy arrays
        raid_json = json.dumps({
            "data": [{
                "type": "pie",
                "labels": ["Successful", "Unsuccessful"],
                "values": [int(p_data.get('player_raids_successful', 0)), int(p_data.get('player_raids_unsuccessful', 0))],
                "marker": {"colors": ['#EF4444', '#94A3B8']},
                "hole": 0.7,
                "textinfo": "none",
                "hoverinfo": "label+percent"
            }],
            "layout": {
                "annotations": [{
                    "text": f"<span style='font-weight:bold; font-size:24px; color:#1E293B;'>{p_data.get('raid_success_rate', 0):.1f}%</span><br><span style='font-size:12px; color:#64748B;'>Raid Success</span>",
                    "x": 0.5, "y": 0.5, "showarrow": False, "align": "center"
                }],
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "margin": {"t": 10, "b": 10, "l": 10, "r": 10},
                "showlegend": False
            }
        })
    
    if p_data.get('player_tackles_total', 0) > 0:
        tackle_json = json.dumps({
            "data": [{
                "type": "pie",
                "labels": ["Successful", "Unsuccessful"],
                "values": [int(p_data.get('player_tackles_successful', 0)), int(p_data.get('player_tackles_unsuccessful', 0))],
                "marker": {"colors": ['#EF4444', '#94A3B8']},
                "hole": 0.7,
                "textinfo": "none",
                "hoverinfo": "label+percent"
            }],
            "layout": {
                "annotations": [{
                    "text": f"<span style='font-weight:bold; font-size:24px; color:#1E293B;'>{p_data.get('tackle_success_rate', 0):.1f}%</span><br><span style='font-size:12px; color:#64748B;'>Tackle Success</span>",
                    "x": 0.5, "y": 0.5, "showarrow": False, "align": "center"
                }],
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "margin": {"t": 10, "b": 10, "l": 10, "r": 10},
                "showlegend": False
            }
        })

    interpretation = TACTICAL_INTERPRETATION.get(p_data.get('tactical_class'), {})

    return render_template('player_profile.html', player=p_data, raid_json=raid_json, tackle_json=tackle_json, interpretation=interpretation)

@app.route('/teams')
def teams():
    if df_teams is None:
        return render_template('teams.html', featured={}, popular_teams=[])
        
    valid_teams = df_teams[df_teams['team_name'] != "Unknown Team"]
    featured = valid_teams.sort_values(by='total_raid_points', ascending=False).head(1)
    featured = featured.iloc[0].to_dict() if not featured.empty else {}
    
    popular_teams = valid_teams.sort_values(by='total_raid_points', ascending=False).head(8).to_dict('records')
    
    return render_template('teams.html', featured=featured, popular_teams=popular_teams)

@app.route('/api/teams/search')
def api_teams_search():
    query = request.args.get('q', '').lower()
    if not query or df_teams is None:
        return jsonify([])
    # Filter out Unknown Team
    valid_teams = df_teams[df_teams['team_name'] != "Unknown Team"]
    matches = valid_teams[valid_teams['team_name'].str.lower().str.contains(query, na=False)]
    return jsonify(matches['team_name'].unique().tolist()[:10])

@app.route('/team/<path:team_name>')
def team_profile(team_name):
    if df_teams is None:
        return "Data not loaded", 500
        
    t_data = df_teams[df_teams['team_name'] == team_name]
    if t_data.empty:
        return "Team not found", 404
        
    t_data = t_data.iloc[0].to_dict()
    team_id = t_data['team_id']
    
    team_players = df_players[df_players['team_id'] == team_id]
    
    # Team Leaders
    top_raider = team_players.sort_values(by='player_raid_points_total', ascending=False).head(1).to_dict('records')
    top_defender = team_players.sort_values(by='player_tackle_points_total', ascending=False).head(1).to_dict('records')
    most_active = team_players.sort_values(by='player_raids_total', ascending=False).head(1).to_dict('records')
    most_succ_def = team_players.sort_values(by='tackle_success_rate', ascending=False).head(1).to_dict('records')
    
    top_raider = top_raider[0] if len(top_raider) > 0 else None
    top_defender = top_defender[0] if len(top_defender) > 0 else None
    most_active = most_active[0] if len(most_active) > 0 else None
    most_succ_def = most_succ_def[0] if len(most_succ_def) > 0 else None
    
    # Calculate Ratings (0-100)
    valid_teams = df_teams[df_teams['team_name'] != "Unknown Team"]
    max_raid_pts = valid_teams['total_raid_points'].max()
    max_tackle_pts = valid_teams['total_tackle_points'].max()
    max_raid_succ = valid_teams['raid_success_rate'].max()
    max_tackle_succ = valid_teams['tackle_success_rate'].max()
    
    attack_rating = int((t_data.get('total_raid_points', 0) / max_raid_pts) * 50 + (t_data.get('raid_success_rate', 0) / max_raid_succ) * 50) if max_raid_pts > 0 else 0
    defense_rating = int((t_data.get('total_tackle_points', 0) / max_tackle_pts) * 50 + (t_data.get('tackle_success_rate', 0) / max_tackle_succ) * 50) if max_tackle_pts > 0 else 0
    
    t_data['attack_rating'] = min(100, attack_rating)
    t_data['defense_rating'] = min(100, defense_rating)
    
    # Team Strength / Weakness logic
    strength = "Balanced Output"
    if t_data.get('raid_success_rate', 0) > 40: strength = "Highly Efficient Raiding"
    elif t_data.get('tackle_success_rate', 0) > 45: strength = "Impenetrable Defense"
    
    weakness = "Consistency Issues"
    if t_data.get('raid_success_rate', 0) < 30: weakness = "Struggling Attack"
    elif t_data.get('tackle_success_rate', 0) < 35: weakness = "Leaky Defense"
    
    t_data['strength'] = strength
    t_data['weakness'] = weakness

    return render_template('team_profile.html', team=t_data, 
                           top_raider=top_raider, top_defender=top_defender,
                           most_active=most_active, most_succ_def=most_succ_def)

@app.route('/performers')
def performers():
    if df_players is None:
        return "Data not loaded", 500
        
    top_r = df_players.sort_values(by='player_raid_points_total', ascending=False).head(10).to_dict('records')
    top_d = df_players.sort_values(by='player_tackle_points_total', ascending=False).head(10).to_dict('records')
    
    df_players['total_impact_points'] = df_players['player_raid_points_total'] + df_players['player_tackle_points_total']
    all_rounders = df_players[(df_players['player_raid_points_total'] > 15) & (df_players['player_tackle_points_total'] > 15)]
    top_a = all_rounders.sort_values(by='total_impact_points', ascending=False).head(10).to_dict('records')
    
    # Team Rankings
    valid_teams = df_teams[df_teams['team_name'] != "Unknown Team"]
    top_teams = valid_teams.sort_values(by='total_raid_points', ascending=False).to_dict('records')
    
    return render_template('performers.html', raiders=top_r, defenders=top_d, allrounders=top_a, teams=top_teams)

@app.route('/insights')
def match_insights():
    if df_match is None:
        return "Data not loaded", 500
        
    df_m = df_match.copy()
    score_pattern = r'\((\d+)\s*-\s*(\d+)\)'
    extracted = df_m['result'].str.extract(score_pattern)
    df_m['score_1'] = pd.to_numeric(extracted[0], errors='coerce')
    df_m['score_2'] = pd.to_numeric(extracted[1], errors='coerce')
    df_m['total_match_points'] = df_m['score_1'] + df_m['score_2']
    df_m['point_difference'] = abs(df_m['score_1'] - df_m['score_2'])
    
    highest = df_m.dropna(subset=['total_match_points']).sort_values(by='total_match_points', ascending=False).head(10).to_dict('records')
    closest = df_m.dropna(subset=['point_difference']).sort_values(by='point_difference', ascending=True).head(10).to_dict('records')
    biggest_margin = df_m.dropna(subset=['point_difference']).sort_values(by='point_difference', ascending=False).head(5).to_dict('records')
    
    # Top Scoring Teams calculation
    valid_teams = df_teams[df_teams['team_name'] != "Unknown Team"]
    top_scoring_teams = valid_teams.sort_values(by='total_raid_points', ascending=False).head(5).to_dict('records')
    
    return render_template('match_insights.html', highest=highest, closest=closest, biggest_margin=biggest_margin, top_teams=top_scoring_teams)

@app.route('/analytics')
def analytics():
    return render_template('predictive_analytics.html')

@app.route('/api/predict_raid', methods=['POST'])
def predict_raid():
    data = request.json
    try:
        time_rem = float(data.get('time', 20))
        defenders = int(data.get('defenders', 7))
        score_diff = int(data.get('score_diff', 0))
        raid_type = data.get('raid_type', 'Standard')
        
        prob = 50
        reasons = []
        
        # 1. Defenders Logic
        if defenders == 1:
            prob += 35
            reasons.append("✓ Only one defender remains. This creates maximum space for the raider.")
            reasons.append("✓ Very high probability of scoring points or securing an All-Out.")
        elif defenders <= 3:
            prob += 20
            reasons.append(f"✓ Reduced defense strength ({defenders} defenders) opens gaps for escapes.")
        elif defenders >= 6:
            prob -= 15
            reasons.append("⚠ Full defensive formation available, reducing scoring opportunities.")
        else:
            reasons.append(f"✓ Standard defense size ({defenders} defenders) offers balanced risk/reward.")
            
        # 2. Raid Type Logic
        if raid_type == 'Do-or-Die':
            prob -= 15
            if defenders == 1:
                reasons.append("⚠ However, the raider MUST score, which increases pressure and risk.")
            else:
                reasons.append("⚠ Do-or-Die situation forces risk-taking. Historically these raids have lower success rates.")
        elif raid_type == 'Bonus Attempt':
            if defenders >= 6:
                prob += 10
                reasons.append("✓ Bonus point is active (6+ defenders), providing an alternative scoring method.")
            else:
                prob -= 5
                reasons.append("⚠ Bonus is NOT active (requires 6+ defenders). Raider must take a touch point.")
        else:
            reasons.append("✓ Standard raid allows safe empty returns if no clear opening exists.")
            
        # 3. Game Context Logic (Time & Score)
        if score_diff < -5 and time_rem < 5:
            prob += 10
            reasons.append("✓ Late-game trailing situation. Aggressive attacking is highly likely.")
        elif score_diff > 5:
            prob -= 5
            reasons.append(f"✓ Team is leading by {score_diff} points, encouraging safer clock management.")
            
        # Clamp probability
        prob = max(5, min(95, prob))
        
        confidence = "High" if prob > 75 or prob < 25 else "Medium"
        if defenders == 1 and raid_type == 'Standard':
            confidence = "Very High"
            
        return jsonify({
            "probability": prob,
            "confidence": confidence,
            "reasons": reasons
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
