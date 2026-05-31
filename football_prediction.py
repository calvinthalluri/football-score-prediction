import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("matches.csv")

print("Dataset Preview:")
print(df.head())

# Features
X = df[['home_shots', 'away_shots',
        'home_possession', 'away_possession']]

# Target
y = df[['home_score', 'away_score']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print("Mean Absolute Error:", round(mae, 2))

# Example prediction
new_match = [[16, 11, 58, 42]]

predicted_score = model.predict(new_match)

print("\nPredicted Score")
print("Home Goals:", round(predicted_score[0][0]))
print("Away Goals:", round(predicted_score[0][1]))