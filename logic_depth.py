import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Sample dataset (Replace with actual RTL data extraction)
data = {
    'Fan_in': [3, 5, 2, 6, 4, 8, 3, 7, 5, 6],
    'Fan_out': [2, 3, 1, 5, 4, 6, 2, 5, 3, 4],
    'Gate_Count': [10, 15, 5, 20, 12, 25, 8, 18, 14, 19],
    'Critical_Path_Length': [4, 7, 2, 9, 6, 10, 3, 8, 5, 7],
    'Logic_Depth': [3, 6, 2, 8, 5, 9, 3, 7, 4, 6]  # Target variable
}

df = pd.DataFrame(data)

# Splitting data into training and testing sets
X = df[['Fan_in', 'Fan_out', 'Gate_Count', 'Critical_Path_Length']]
y = df['Logic_Depth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
accuracy = (1 - mae / np.mean(y_test)) * 100

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Function to predict logic depth for new RTL parameters
def predict_logic_depth(fan_in, fan_out, gate_count, critical_path):
    input_data = np.array([[fan_in, fan_out, gate_count, critical_path]])
    predicted_depth = model.predict(input_data)[0]
    return round(predicted_depth, 2)

# Example prediction
example_prediction = predict_logic_depth(4, 3, 14, 6)
print(f"Predicted Logic Depth: {example_prediction}")
