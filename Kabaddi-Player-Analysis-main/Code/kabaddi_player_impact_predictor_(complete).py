# Part 1: Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error

# Part 2: Data Loading and Preparation


def load_and_prepare_data(players_path):
    """
    Loads and aggregates player data to a season level.
    The events data is no longer needed as we now predict efficiency from player stats.
    """
    print("--- Starting Part 1: Data Preparation ---")
    try:
        players_df = pd.read_csv(players_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your CSV files are uploaded to Colab and paths are correct.")
        return None

    # --- Handle missing player_role before aggregation ---
    print("Filling missing player roles...")
    players_df['player_role'] = players_df.groupby('player_id')['player_role'].transform(lambda x: x.ffill().bfill())
    players_df['player_role'] = players_df['player_role'].fillna('Unknown')

    # --- Aggregate player stats to season level ---
    print("Aggregating players_df to get season-level stats for each player...")
    agg_dict = {
        'player_name': 'first',
        'player_role': 'first',
        'player_raids_successful': 'sum',
        'player_tackles_successful': 'sum',
        'player_raids_total': 'sum',
        'player_tackles_total': 'sum',
    }
    players_df_agg = players_df.groupby('player_id').agg(agg_dict).reset_index()
    print(f"Aggregated players data into {len(players_df_agg)} unique players.")
    return players_df_agg

# Use your absolute path for the players CSV:
players_csv_path = r"C:\Users\Om Raut\Documents\Projects\Kabaddi Player Analysis\Code\DS_players.csv"

players_df = load_and_prepare_data(players_csv_path)

# You can do similar for other CSVs:
events_csv_path = r"C:\Users\Om Raut\Documents\Projects\Kabaddi Player Analysis\Code\DS_events.csv"
match_csv_path  = r"C:\Users\Om Raut\Documents\Projects\Kabaddi Player Analysis\Code\DS_match.csv"
team_csv_path   = r"C:\Users\Om Raut\Documents\Projects\Kabaddi Player Analysis\Code\DS_team.csv"

# For example, load match data:
match_df = pd.read_csv(match_csv_path)
# Load other files as needed in your analysis.


# Part 3: Model Training and Evaluation (Regression for Efficiency)
def train_and_evaluate_regressors(df):
    """
    Creates an efficiency score target, trains regression models, and evaluates them.
    Returns the best trained Random Forest model, the test target values, and the tuned model's predictions on the test set.
    """
    print("\n--- Starting Part 2: Model Training and Evaluation ---")

    # --- Create Efficiency Features AND the New Target Variable ---
    epsilon = 1e-6 # A small number to prevent division by zero
    df['raid_success_rate'] = df['player_raids_successful'] / (df['player_raids_total'] + epsilon)
    df['tackle_success_rate'] = df['player_tackles_successful'] / (df['player_tackles_total'] + epsilon)

    # NEW TARGET: A weighted score of a player's efficiency.
    # We give slightly more weight to raiding as it's the primary scoring method.
    df['overall_efficiency_score'] = (0.6 * df['raid_success_rate']) + (0.4 * df['tackle_success_rate'])

    # --- NEW: Engineer a 'Player Style' feature ---
    df['raid_to_tackle_ratio'] = df['player_raids_total'] / (df['player_tackles_total'] + epsilon)


    # 1. Define Features (X) and Target (y)
    target = 'overall_efficiency_score' # Our new regression target

    # Use ACTIVITY, STYLE, and ROLE as features to predict EFFICIENCY.
    numerical_features = [
        'player_raids_total',
        'player_tackles_total',
        'raid_to_tackle_ratio' # Add our new feature to the model
    ]
    categorical_features = ['player_role']

    df_cleaned = df.dropna(subset=numerical_features + categorical_features + [target])

    X = df_cleaned[numerical_features + categorical_features]
    y = df_cleaned[target]

    print(f"\nTraining with {len(X)} players after cleaning.")

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create a preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 4. Train and evaluate baseline models
    baseline_models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "XGBoost Regressor (Advanced)": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }

    for name, model in baseline_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\n--- {name} Results ---")
        print(f"  R-squared: {r2:.4f}")
        print(f"  Mean Absolute Error: {mae:.4f}")

    # --- 5. Hyperparameter Tuning for Random Forest ---
    print("\n--- Hyperparameter Tuning for Random Forest Regressor ---")

    # Create a pipeline with the preprocessor and the regressor
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(random_state=42))])

    # Define the 'dials' to search over
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Set up the search
    grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='r2', verbose=1)

    # Run the search on the training data
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {grid_search.best_params_}")

    # Evaluate the BEST model on the test data
    best_rf_model = grid_search.best_estimator_
    y_pred_tuned = best_rf_model.predict(X_test)

    r2_tuned = r2_score(y_test, y_pred_tuned)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

    print(f"\n--- Tuned Random Forest Results ---")
    print(f"  R-squared: {r2_tuned:.4f}")
    print(f"  Mean Absolute Error: {mae_tuned:.4f}")

    try:
        # Get feature importances from the tuned model's regressor step
        regressor_step = best_rf_model.named_steps['regressor']
        ohe_feature_names = best_rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)
        importances = regressor_step.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)

        print("  Top 5 Most Important Features:")
        print(feature_importance_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"  Could not compute feature importances due to: {e}")

    return best_rf_model, y_test, y_pred_tuned


# Part 4: Main Execution Block
if __name__ == '__main__':
    
    # We now only need the players CSV as the target is derived from it.
    PLAYERS_PATH = 'C:/Users/Om Raut/Documents/Projects/Kabaddi Player Analysis/Code/DS_players.csv'


    # Step 1: Run the data preparation
    ml_ready_df = load_and_prepare_data(PLAYERS_PATH)

    # Step 2: Run the model training and evaluation
    trained_model = None # Initialize to None
    y_test_results = None # Initialize to None
    y_pred_tuned_results = None # Initialize to None

    if ml_ready_df is not None and not ml_ready_df.empty:
        trained_model, y_test_results, y_pred_tuned_results = train_and_evaluate_regressors(ml_ready_df)
    else:
        print("\nModel training skipped because data preparation failed or resulted in an empty dataset.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Running Test 1: The 'Common Sense' Test ---")

# A very active, aggressive raider (should have a high score)
star_raider = pd.DataFrame({
    'player_raids_total': [250],
    'player_tackles_total': [20],
    'raid_to_tackle_ratio': [250 / (20 + 1e-6)],
    'player_role': ['Raider'],
})

# A solid, defense-focused player (should have a good score)
rock_defender = pd.DataFrame({
    'player_raids_total': [30],
    'player_tackles_total': [100],
    'raid_to_tackle_ratio': [30 / (100 + 1e-6)],
    'player_role': ['Defender'],
})

# A player who barely plays (should have a low score)
substitute_player = pd.DataFrame({
    'player_raids_total': [5],
    'player_tackles_total': [2],
    'raid_to_tackle_ratio': [5 / (2 + 1e-6)],
    'player_role': ['All-Rounder'],
})

# Check if trained_model exists before using it
if 'trained_model' in locals() and trained_model is not None:
    pred_raider = trained_model.predict(star_raider)
    pred_defender = trained_model.predict(rock_defender)
    pred_sub = trained_model.predict(substitute_player)

    print(f"\nPredicted Efficiency Score for the 'Star Raider': {pred_raider[0]:.4f}")
    print(f"Predicted Efficiency Score for the 'Rock Defender': {pred_defender[0]:.4f}")
    print(f"Predicted Efficiency Score for the 'Substitute Player': {pred_sub[0]:.4f}")
    print("\nSense Check Complete. Ask yourself: Do these scores seem logical?")
else:
    print("Model not trained. Cannot make predictions.")

print("\n--- Running Test 2: The 'Error Check' ---")

# Calculate the errors for all the predictions on the test set
if y_test_results is not None and y_pred_tuned_results is not None:
    residuals = y_test_results - y_pred_tuned_results

    # Create a scatter plot of the errors
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_tuned_results, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--') # The red line means "zero error"
    plt.xlabel("Model's Prediction")
    plt.ylabel("Prediction Error")
    plt.title("Picture of the Model's Mistakes (Residual Plot)")
    plt.grid(True)
    plt.show()

    print("\nError Check Complete. A random cloud of dots is a GOOD sign!")
else:
    print("\nError Check skipped as model training did not produce test results.")