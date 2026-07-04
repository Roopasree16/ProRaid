# Kabaddi Player Analysis & Impact Prediction Dashboard

A comprehensive data-driven web application and machine learning pipeline designed to analyze, visualize, and predict player performance and match dynamics in the **Pro Kabaddi League (PKL)**. 

This project combines an interactive **Flask Web App** (with Plotly visualizations, player search, team profiling, and a live raid success simulation engine) with a robust **Machine Learning Pipeline** (training Linear Regression, Random Forest, and XGBoost models to predict player efficiency).

---

##  Key Features

### 1. Flask Web Application Dashboard
* **Dynamic Home Dashboard:** Showcases real-time key performance indicators (KPIs) including total active players, teams, matches analyzed, and top-tier player awards (Best Raider, Best Defender, Consistent Performer).
* **Interactive Player Profiles:** Displays detailed player-level performance statistics, overall offensive/defensive contributions, and interactive doughnut charts for raid/tackle success rates.
* **Team Profile & Ratings:** Evaluates PKL teams with dynamic ratings (0–100) for attack and defense, alongside automatic classification of tactical strengths and weaknesses.
* **Performers Leaderboard:** Leaderboards highlighting top performers across different categories: Top 10 Raiders, Top 10 Defenders, Top 10 All-Rounders, and Team Standings.
* **Match Insights:** Aggregates season data to identify the highest-scoring games, closest finishes, and largest victory margins.
* **Interactive Simulation Engine:** Evaluates the success probability of a raid in real-time based on live context (match time remaining, active defenders on the mat, score margin, and raid type such as standard, bonus attempt, or Do-or-Die).

### 2. Machine Learning Pipeline (`kabaddi_player_impact_predictor_(complete).py`)
* **Data Preparation & Aggregation:** Handles missing role values and aggregates raw event/match statistics to a season level.
* **Feature Engineering:** Calculates raid success rate, tackle success rate, and raid-to-tackle ratio to classify player style.
* **Efficiency Prediction:** Creates a target metric called `overall_efficiency_score` (calculated as `0.6 * raid_success_rate + 0.4 * tackle_success_rate`).
* **Model Selection & Tuning:** Trains and compares Linear Regression, Random Forest Regressor, and XGBoost Regressor. Uses `GridSearchCV` for hyperparameter tuning.

---

##  Project Directory Structure

```text
Kabaddi-Player-Analysis-main/
├── .gitignore
├── README.md                              # This root documentation file
└── Kabaddi-Player-Analysis-main/          # Subdirectory containing codebase
    ├── Code/                              # Backend app and dataset files
    │   ├── app.py                         # Main Flask application
    │   ├── kabaddi_player_impact_predictor_(complete).py  # Machine learning model pipeline
    │   ├── requirements.txt               # Python package dependencies
    │   ├── DS_players.csv                 # Raw player dataset
    │   ├── DS_team.csv                    # Raw team dataset
    │   ├── DS_match.csv                   # Raw match dataset
    │   ├── DS_events.csv                  # Raw play-by-play events dataset
    │   ├── templates/                     # Jinja2 HTML templates
    │   │   ├── base.html                  # Global layout skeleton
    │   │   ├── index.html                 # Homepage dashboard
    │   │   ├── players.html               # Player search & directory
    │   │   ├── player_profile.html        # Detailed player analytics page
    │   │   ├── teams.html                 # Team list page
    │   │   ├── team_profile.html          # Team detailed analysis page
    │   │   ├── performers.html            # Top performers leaderboard
    │   │   ├── match_insights.html        # Match-level analytics report
    │   │   └── predictive_analytics.html  # Live raid simulation dashboard
    │   └── static/                        # CSS styles, assets, and frontend Javascript
    │       ├── css/
    │       └── js/
    └── Screenshot 2025-11-18 152640.png   # Dashboard UI preview screenshot
```

---

##  Technology Stack

* **Frontend:** HTML5, CSS3 (Vanilla design tokens, custom layout animations, modern typography), JavaScript (Vanilla ES6, Plotly.js for charts)
* **Backend:** Flask, Python (>= 3.8)
* **Data & Machine Learning:** Pandas, NumPy, Scikit-Learn, XGBoost, Plotly

---

##  Installation & Getting Started

### Prerequisites
* Python 3.8 or higher installed on your system.

### Steps
1. **Clone or download** this repository to your local machine.
2. Open your terminal and navigate to the `Code` directory:
   ```bash
   cd Kabaddi-Player-Analysis-main/Code
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### 1. Running the Web Application
To start the Flask development server, execute the following command:
```bash
python app.py
```
After the server initializes, open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000) or [http://localhost:5000](http://localhost:5000)

### 2. Running the ML Pipeline
To train and tune the regression models on the player dataset, run:
```bash
python "kabaddi_player_impact_predictor_(complete).py"
```

*Note: Ensure the dataset CSV file paths within the Python files match your folder configuration.*
